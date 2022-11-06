import torch
from .config import _C as cfg

from .baseline_model import GEBDModel
from .e2e_model import E2EModel
from .e2e_model_ban_v4 import E2EModelBAN


def build_model(cfg):
    if cfg.MODEL.NAME == 'GEBDModel':
        model = GEBDModel(cfg)
    elif cfg.MODEL.NAME == 'E2EModel':
        model = E2EModel(cfg)
    elif cfg.MODEL.NAME == 'E2EModelBAN':
        model = E2EModelBAN(cfg)
    else:
        raise NotImplemented(f'No such model: {cfg.MODEL.NAME}')

    if cfg.MODEL.SYNC_BN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model
