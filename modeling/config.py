from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = 'GEBDModel'
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'resnet50'
_C.MODEL.SYNC_BN = True
_C.MODEL.DIMENSION = 512
_C.MODEL.WINDOW_SIZE = 8
_C.MODEL.SIMILARITY_GROUP = 4
_C.MODEL.SIMILARITY_FUNC = 'cosine'
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ('GEBD_train',)
_C.DATASETS.TEST = ('GEBD_minval',)

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.RESOLUTION = 224
_C.INPUT.ARGUMENT = True
_C.INPUT.ANNOTATORS = 2

_C.INPUT.FRAME_PER_SIDE = 5
_C.INPUT.DYNAMIC_DOWNSAMPLE = False

_C.INPUT.DOWNSAMPLE = 3
_C.INPUT.END_TO_END = False  # input whole video
_C.INPUT.SEQUENCE_LENGTH = 50  # input whole video
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 30
_C.SOLVER.WARMUP_EPOCHS = 2
_C.SOLVER.MILESTONES = [2, 3]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.AMPE = True  # automatic mixed precision training
_C.SOLVER.LR = 1e-2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.CLIP_GRAD = 0.0
_C.SOLVER.NUM_WORKERS = 8
_C.SOLVER.OPTIMIZER = 'SGD'

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.THRESHOLD = 0.5
_C.TEST.PRED_FILE = ''  # precomputed predictions
_C.TEST.RELDIS_THRESHOLD = 0.05

_C.OUTPUT_DIR = 'output'
