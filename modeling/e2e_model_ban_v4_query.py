import itertools
import math

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch.nn.utils import weight_norm
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d
from transformers import BertConfig, BertLayer
from utils.distribute import is_main_process
from modeling.video_swin_transformer import video_swin_t, video_swin_b, video_swin_s
import timm
from modeling.transformer_layer import TransformerDecoderLayer, TransformerDecoder


def prepare_gaussian_targets(targets, sigma=1):
    gaussian_targets = []
    for batch_idx in range(targets.shape[0]):
        t = targets[batch_idx]
        axis = torch.arange(len(t), device=targets.device)
        gaussian_t = torch.zeros_like(t)
        indices, = torch.nonzero(t, as_tuple=True)
        for i in indices:
            g = torch.exp(-(axis - i) ** 2 / (2 * sigma * sigma))
            gaussian_t += g

        gaussian_t = gaussian_t.clamp(0, 1)
        # gaussian_t /= gaussian_t.max()
        gaussian_targets.append(gaussian_t)
    gaussian_targets = torch.stack(gaussian_targets, dim=0)
    return gaussian_targets


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=8):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (T, C)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, C)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x


class PositionalEncodingLearned(nn.Module):
    def __init__(self, dim, size=8):
        super(PositionalEncodingLearned, self).__init__()
        self.embed = nn.Embedding(size, dim)

    def forward(self, x):
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class GroupSimilarity(nn.Module):
    def __init__(self, dim, window_size, group=4, similarity_func='cosine', offset=0, use_contrastive=False):
        super(GroupSimilarity, self).__init__()
        self.out_channels = dim * 2
        self.group = group
        self.similarity_func = similarity_func
        self.offset = offset

        k = 5
        padding = (k - 1) // 2

        self.fcn = nn.Sequential(
            BasicConv2d(self.group, dim, kernel_size=k, stride=1, padding=padding),
            BasicConv2d(dim, dim, kernel_size=k, stride=1, padding=padding),
            BasicConv2d(dim, dim, kernel_size=k, stride=1, padding=padding),
            BasicConv2d(dim, dim, kernel_size=k, stride=1, padding=padding),
        )

        # self.pe = PositionalEncoding(dim, max_len=kernel_size * 2)
        encoder_config = BertConfig(
            hidden_size=dim,
            num_attention_heads=8,
            intermediate_size=2048,
        )
        # self.emb_cls = nn.Parameter(0.02 * torch.randn(1, dim))
        self.encoders = nn.ModuleList([BertLayer(encoder_config) for _ in range(6)])
        decoder_layer = nn.TransformerDecoderLayer(dim, nhead=8)
        # self.linear = nn.Linear(dim, dim)
        print('sim k={}, similarity-head: {} top2-224-aug'.format(k, self.group))
        self.use_contrastive = use_contrastive
        # if use_contrastive:
        #     self.predictor = self.predictor = nn.Sequential(nn.Conv1d(dim, dim, 1, 1, 0,bias=False),
        #                                    nn.BatchNorm1d(dim),
        #                                    nn.ReLU(inplace=True),  # hidden layer
        #                                    nn.Conv1d(dim, dim, 1, 1, 0))  # output layer
        decoder_layer = TransformerDecoderLayer(dim, nheads=8, norm=nn.LayerNorm(dim))
        self.decoder = TransformerDecoder(decoder_layer, num_layers=3)
        self.object_query = nn.Embedding(1, dim)

    def recognize_patterns(self, left_seq, mid_seq, right_seq, offset=0):
        k = left_seq.shape[1]
        assert k > offset

        left_seq = left_seq[:, offset:]
        right_seq = right_seq[:, :(None if offset == 0 else -offset)]
        assert left_seq.shape[1] == right_seq.shape[1] == (k - offset)
        b = left_seq.shape[0]

        x = torch.cat([left_seq, mid_seq, right_seq], dim=1)

        for encoder in self.encoders:
            x = encoder(x)[0]  # (B, L, C)

        B, L, C = x.shape

        tgt = torch.zeros([L, B, C], device=x.device)
        x_decoder = self.decoder(tgt, einops.rearrange(x, 'b t c -> t b c'), query_pos=einops.repeat(self.object_query.weight, 'n c -> n b c', b=b))[0][0]

        # cls_feature, x = x[:, -1, :], x[:, :-1, :]


        x = x.view(B, L, self.group, C // self.group)  # (B, L, G, C')
        # (B, L, L, H)
        similarity_func = self.similarity_func

        if similarity_func == 'cosine':
            sim = F.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)  # batch, T, T, G
        else:
            raise NotImplemented

        sim = sim.permute(0, 3, 1, 2)  # batch, G, T, T
        # print(sim.shape)
        # import numpy as np
        # global INDEX
        # np.save(f'similarity_maps{INDEX}', sim.detach().cpu().numpy())
        # INDEX += 1

        # sim2 = torch.cdist(x, x, p=2)[:, None]  # batch, 1, T, T
        # sim = F.instance_norm(sim)·
        # sim = torch.cat([sim, sim2], dim=1)  # batch, 2, T, T

        h = self.fcn(sim)  # batch, dim, T, T
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        h = torch.cat([x_decoder, h], dim=1)

        return h

    def forward(self, left_seq, mid_seq, right_seq):
        """
        left_seq = batch, T, dim
        mid_seq = batch, 1, dim
        right_seq = batch, T, dim
        """
        h = self.recognize_patterns(left_seq, mid_seq, right_seq, offset=self.offset)

        return h


def SPoS(inputs, ban, k):
    """(b c t)"""
    B = inputs.shape[0]
    L = inputs.shape[-1]
    C = inputs.shape[1]

    padded_inputs = F.pad(inputs, pad=(0, math.ceil(L / k) * k - L), mode='replicate')
    pad_L = padded_inputs.shape[-1]

    # outputs = torch.zeros_like(padded_inputs)
    outputs = torch.zeros(B, ban.out_channels, pad_L, dtype=inputs.dtype, device=inputs.device)
    for offset in range(k):
        left_x = F.pad(padded_inputs, pad=(k - offset, 0), mode='replicate')[:, :, :-(k - offset)]
        right_x = F.pad(padded_inputs, pad=(0, offset + 1), mode='replicate')[:, :, (offset + 1):]
        left_seq = einops.rearrange(left_x, 'b c (nw k) -> (b nw) k c', k=k)
        right_seq = einops.rearrange(right_x, 'b c (nw k) -> (b nw) k c', k=k)
        mid_seq = einops.rearrange(padded_inputs[:, :, offset::k], 'b c nw -> (b nw) 1 c')

        h = ban(left_seq, mid_seq, right_seq)  # (b nw) c
        hidden_state = einops.rearrange(h, '(b nw) c -> b c nw', b=B)

        outputs[:, :, offset::k] = hidden_state

    outputs = outputs[:, :, :L]  # (b c t)
    return outputs


class TemporalEmbedding(Module):
    def __init__(self, dim):
        super().__init__()
        model = []
        for _ in range(4):
            model += [
                nn.Conv1d(dim, dim, 3, 1, 1),
                nn.BatchNorm1d(dim),
                nn.PReLU(),
            ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        Args:
            x: (b c t)
        """
        x = self.model(x)
        return x


class E2EModelBANQuery(Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone_name = cfg.MODEL.BACKBONE.NAME
        self.weight_bce_loss = cfg.SOLVER.WEIGHT_BCE
        self.weight_smooth_l1_loss = cfg.SOLVER.WEIGHT_SMOOTH_L1
        self.weight_contrastive_loss = cfg.SOLVER.WEIGHT_CONTRASTIVE
        self.multi_level = cfg.MODEL.MULTI_LEVEL
        if self.backbone_name == 'csn':
            from .backbone import CSN
            self.backbone = CSN()
            in_feat_dim = 2048
            if self.multi_level:
                in_feat_dim = 2048 + 1024
        elif self.backbone_name == 'video_swin-t':
            self.backbone = video_swin_t('swin_tiny_patch244_window877_kinetics400_1k.pth')
            in_feat_dim = 768
        elif self.backbone_name == 'video_swin-s':
            self.backbone = video_swin_s('swin_small_patch244_window877_kinetics400_1k.pth')
            in_feat_dim = 768
        elif self.backbone_name == 'video_swin-b':
            self.backbone = video_swin_b('swin_base_patch244_window877_kinetics600_22k.pth')
            in_feat_dim = 1024
        elif self.backbone_name == 'swin-l':
            self.backbone = timm.create_model('swin_large_patch4_window7_224_in22k', pretrained=True, num_classes=0, global_pool='')
            in_feat_dim = 1536
        elif self.backbone_name == 'swin-b':
            self.backbone = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True, num_classes=0, global_pool='')
            in_feat_dim = 1024
        elif self.backbone_name == 'swin-s':
            self.backbone = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=0, global_pool='')
            in_feat_dim = 768
        else:
            self.backbone = getattr(models, self.backbone_name)(pretrained=True, norm_layer=FrozenBatchNorm2d)
            in_feat_dim = self.backbone.fc.in_features
            for param in itertools.chain(self.backbone.conv1.parameters(), self.backbone.bn1.parameters()):
                param.requires_grad = False

            del self.backbone.fc

        dim = cfg.MODEL.DIMENSION
        self.window_size = cfg.MODEL.WINDOW_SIZE
        self.reduce_dim = nn.Linear(in_feat_dim, dim)
        self.temporal_embedding = TemporalEmbedding(dim)
        self.group_similarity = GroupSimilarity(dim=dim,
                                                window_size=self.window_size,
                                                group=cfg.MODEL.SIMILARITY_GROUP,
                                                similarity_func=cfg.MODEL.SIMILARITY_FUNC)

        self.classifier = nn.Sequential(
            nn.Conv1d(self.group_similarity.out_channels, dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv1d(dim, 1, 1)
        )

        if self.weight_contrastive_loss > 0:
            self.predictor = nn.Sequential(nn.Conv1d(dim, dim, 1, 1, 0,bias=False),
                                           nn.BatchNorm1d(dim),
                                           nn.ReLU(inplace=True),  # hidden layer
                                           nn.Conv1d(dim, dim, 1, 1, 0))  # output layer

    def extract_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(self, inputs, targets=None):
        """
        Args:
            inputs(dict): imgs (B, T, C, H, W), frame_masks (B, T);
            targets: (B, T);
        Returns:
        """
        imgs = inputs['imgs']

        B = imgs.shape[0]
        if self.backbone_name == 'csn':
            x = self.backbone(imgs)

            x = [einops.rearrange(o, 'b c t h w -> (b t) c h w') for o in x]
        elif 'video_swin' in self.backbone_name:
            x = self.backbone(imgs)
            x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        elif self.backbone_name == 'swin-s' or self.backbone_name == 'swin-b' or self.backbone_name == 'swin-l':
            imgs = einops.rearrange(imgs, 'b t c h w -> (b t) c h w')
            x = self.backbone(imgs)[..., None, None]
        else:
            imgs = einops.rearrange(imgs, 'b t c h w -> (b t) c h w')
            x = self.extract_features(imgs)

        if self.multi_level:
            tmp_x = []
            # print(x.shape)
            for i in range(len(x)):
                tmp_x.append(F.adaptive_avg_pool2d(x[i], 1).flatten(1))
            x = torch.cat(tmp_x, dim=1)
        else:
            if isinstance(x, list):
                x = x[-1]
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.reduce_dim(x)

        x = einops.rearrange(x, '(b t) c -> b c t', b=B)  # (4, 512, 100)
        x = self.temporal_embedding(x) # b c t

        if self.weight_contrastive_loss > 0:
            x_proj = self.predictor(x) # b c t
            z = x

        feats = SPoS(x, self.group_similarity, self.window_size)  # b c t

        logits = self.classifier(feats)  # b 1 t

        if self.training:
            if isinstance(targets, dict):
                targets = targets['targets']
            targets = targets.to(logits.dtype)
            targets_gaussian = prepare_gaussian_targets(targets)

            logits = logits.view(-1)
            targets_gaussian = targets_gaussian.view(-1)

            if 'frame_masks' in inputs:
                masks = inputs['frame_masks'].view(-1)
                logits = logits[masks]
                targets_gaussian = targets_gaussian[masks]

            loss_dict = {}
            if self.weight_bce_loss > 0:
                loss = F.binary_cross_entropy_with_logits(logits, targets_gaussian)
                loss_dict.update({'bce_loss': loss * self.weight_bce_loss})
            if self.weight_smooth_l1_loss > 0:
                loss = F.smooth_l1_loss(torch.sigmoid(logits), targets_gaussian)
                loss_dict.update({'smooth_l1_loss': loss * self.weight_smooth_l1_loss})
            if self.weight_contrastive_loss > 0:
                contrastive_mask = self.get_mask(targets, inputs['frame_masks'])
                sim_mat = self.get_similarity_map(x_proj, z.detach(), contrastive_mask)
                loss_contrastive = - torch.sum(sim_mat) / torch.sum(contrastive_mask)
                loss_dict.update({'contrastive_loss': loss_contrastive * self.weight_contrastive_loss})
            return loss_dict

        scores = torch.sigmoid(logits).flatten(1)
        return scores

    def get_mask(self, targets, frame_masks):
        mask = torch.zeros(targets.shape[0], targets.shape[1], targets.shape[1], device=targets.device)
        # bound_pos = []
        b = targets.shape[0]
        for i in range(b):
            tmp_targets = targets[i].cpu().numpy()
            tmp_pos = [0]
            for j in range(len(tmp_targets)):
                if tmp_targets[j] == 1:
                    tmp_pos.append(j)
            end_pos = int(torch.sum(frame_masks[i]).item())
            tmp_pos.append(end_pos-1)
            for k in range(len(tmp_pos)-1):
                mask[i, tmp_pos[k]:tmp_pos[k+1], tmp_pos[k]:tmp_pos[k+1]] = torch.ones([tmp_pos[k+1]-tmp_pos[k], tmp_pos[k+1]-tmp_pos[k]])

        return mask


def get_similarity_map(x, z):
    x = F.normalize(x, dim=1)
    z = F.normalize(z, dim=1)
    x = einops.rearrange(x, 'b c t -> b t c')
    sim_mat = torch.bmm(x, z)

    return sim_mat



