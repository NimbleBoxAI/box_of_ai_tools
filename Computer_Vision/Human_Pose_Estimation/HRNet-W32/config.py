# https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/tree/master/lib/config
import os
from yacs.config import CfgNode as CN

# pose_multi_resoluton_net related params
POSE_HIGHER_RESOLUTION_NET = CN()
POSE_HIGHER_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
POSE_HIGHER_RESOLUTION_NET.STEM_INPLANES = 64
POSE_HIGHER_RESOLUTION_NET.FINAL_CONV_KERNEL = 1

POSE_HIGHER_RESOLUTION_NET.STAGE1 = CN()
POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_MODULES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_BRANCHES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_BLOCKS = [4]
POSE_HIGHER_RESOLUTION_NET.STAGE1.NUM_CHANNELS = [64]
POSE_HIGHER_RESOLUTION_NET.STAGE1.BLOCK = 'BOTTLENECK'
POSE_HIGHER_RESOLUTION_NET.STAGE1.FUSE_METHOD = 'SUM'

POSE_HIGHER_RESOLUTION_NET.STAGE2 = CN()
POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
POSE_HIGHER_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [24, 48]
POSE_HIGHER_RESOLUTION_NET.STAGE2.BLOCK = 'BOTTLENECK'
POSE_HIGHER_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'

POSE_HIGHER_RESOLUTION_NET.STAGE3 = CN()
POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
POSE_HIGHER_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [24, 48, 92]
POSE_HIGHER_RESOLUTION_NET.STAGE3.BLOCK = 'BOTTLENECK'
POSE_HIGHER_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'

POSE_HIGHER_RESOLUTION_NET.STAGE4 = CN()
POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_MODULES = 1
POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
POSE_HIGHER_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [24, 48, 92, 192]
POSE_HIGHER_RESOLUTION_NET.STAGE4.BLOCK = 'BOTTLENECK'
POSE_HIGHER_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'

POSE_HIGHER_RESOLUTION_NET.DECONV = CN()
POSE_HIGHER_RESOLUTION_NET.DECONV.NUM_DECONVS = 2
POSE_HIGHER_RESOLUTION_NET.DECONV.NUM_CHANNELS = [32, 32]
POSE_HIGHER_RESOLUTION_NET.DECONV.NUM_BASIC_BLOCKS = 4
POSE_HIGHER_RESOLUTION_NET.DECONV.KERNEL_SIZE = [2, 2]
POSE_HIGHER_RESOLUTION_NET.DECONV.CAT_OUTPUT = [True, True]

# final
MODEL_EXTRAS = {
    'pose_multi_resolution_net_v16': POSE_HIGHER_RESOLUTION_NET,
}


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.VERBOSE = True
_C.DIST_BACKEND = 'nccl'
_C.MULTIPROCESSING_DISTRIBUTED = True

# FP16 training params
_C.FP16 = CN()
_C.FP16.ENABLED = False
_C.FP16.STATIC_LOSS_SCALE = 1.0
_C.FP16.DYNAMIC_LOSS_SCALE = False

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_multi_resolution_net_v16'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.EXTRA = POSE_HIGHER_RESOLUTION_NET
_C.MODEL.SYNC_BN = False

_C.LOSS = CN()
_C.LOSS.NUM_STAGES = 1
_C.LOSS.WITH_HEATMAPS_LOSS = (True,)
_C.LOSS.HEATMAPS_LOSS_FACTOR = (1.0,)
_C.LOSS.WITH_AE_LOSS = (True,)
_C.LOSS.AE_LOSS_TYPE = 'max'
_C.LOSS.PUSH_LOSS_FACTOR = (0.001,)
_C.LOSS.PULL_LOSS_FACTOR = (0.001,)

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'coco_kpt'
_C.DATASET.DATASET_TEST = 'coco'
_C.DATASET.NUM_JOINTS = 17
_C.DATASET.MAX_NUM_PEOPLE = 30
_C.DATASET.TRAIN = 'train2017'
_C.DATASET.TEST = 'val2017'
_C.DATASET.DATA_FORMAT = 'jpg'

# training data augmentation
_C.DATASET.MAX_ROTATION = 30
_C.DATASET.MIN_SCALE = 0.75
_C.DATASET.MAX_SCALE = 1.25
_C.DATASET.SCALE_TYPE = 'short'
_C.DATASET.MAX_TRANSLATE = 40
_C.DATASET.INPUT_SIZE = 512
_C.DATASET.OUTPUT_SIZE = [128, 256, 512]
_C.DATASET.FLIP = 0.5

# heatmap generator (default is OUTPUT_SIZE/64)
_C.DATASET.SIGMA = -1
_C.DATASET.SCALE_AWARE_SIGMA = False
_C.DATASET.BASE_SIZE = 256.0
_C.DATASET.BASE_SIGMA = 2.0
_C.DATASET.INT_SIGMA = False

_C.DATASET.WITH_CENTER = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.IMAGES_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
# _C.TEST.BATCH_SIZE = 32
_C.TEST.IMAGES_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.ADJUST = True
_C.TEST.REFINE = True
_C.TEST.SCALE_FACTOR = [1]
# group
_C.TEST.DETECTION_THRESHOLD = 0.2
_C.TEST.TAG_THRESHOLD = 1.
_C.TEST.USE_DETECTION_VAL = True
_C.TEST.IGNORE_TOO_MUCH = False
_C.TEST.MODEL_FILE = ''
_C.TEST.IGNORE_CENTER = True
_C.TEST.NMS_KERNEL = 3
_C.TEST.NMS_PADDING = 1
_C.TEST.PROJECT2IMAGE = False

_C.TEST.WITH_HEATMAPS = (True,)
_C.TEST.WITH_AE = (True,)

_C.TEST.LOG_PROGRESS = False

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = True
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = True
_C.DEBUG.SAVE_HEATMAPS_PRED = True
_C.DEBUG.SAVE_TAGMAPS_PRED = True

CONFIG = _C


# dataset dependent configuration for visualization
coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r'
]
coco_part_idx = {
    b: a for a, b in enumerate(coco_part_labels)
}
coco_part_orders = [
    ('nose', 'eye_l'), ('eye_l', 'eye_r'), ('eye_r', 'nose'),
    ('eye_l', 'ear_l'), ('eye_r', 'ear_r'), ('ear_l', 'sho_l'),
    ('ear_r', 'sho_r'), ('sho_l', 'sho_r'), ('sho_l', 'hip_l'),
    ('sho_r', 'hip_r'), ('hip_l', 'hip_r'), ('sho_l', 'elb_l'),
    ('elb_l', 'wri_l'), ('sho_r', 'elb_r'), ('elb_r', 'wri_r'),
    ('hip_l', 'kne_l'), ('kne_l', 'ank_l'), ('hip_r', 'kne_r'),
    ('kne_r', 'ank_r')
]

crowd_pose_part_labels = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'head', 'neck'
]
crowd_pose_part_idx = {
    b: a for a, b in enumerate(crowd_pose_part_labels)
}
crowd_pose_part_orders = [
    ('head', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle')
]

VIS_CONFIG = {
    'COCO': {
        'part_labels': coco_part_labels,
        'part_idx': coco_part_idx,
        'part_orders': coco_part_orders
    },
    'CROWDPOSE': {
        'part_labels': crowd_pose_part_labels,
        'part_idx': crowd_pose_part_idx,
        'part_orders': crowd_pose_part_orders
    }
}



def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.config)

    if not os.path.exists(cfg.DATASET.ROOT):
        cfg.DATASET.ROOT = os.path.join(
            cfg.DATA_DIR, cfg.DATASET.ROOT
        )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    if cfg.DATASET.WITH_CENTER:
        cfg.DATASET.NUM_JOINTS += 1
        cfg.MODEL.NUM_JOINTS = cfg.DATASET.NUM_JOINTS

    if not isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)):
        cfg.DATASET.OUTPUT_SIZE = [cfg.DATASET.OUTPUT_SIZE]
    if not isinstance(cfg.LOSS.WITH_HEATMAPS_LOSS, (list, tuple)):
        cfg.LOSS.WITH_HEATMAPS_LOSS = (cfg.LOSS.WITH_HEATMAPS_LOSS)

    if not isinstance(cfg.LOSS.HEATMAPS_LOSS_FACTOR, (list, tuple)):
        cfg.LOSS.HEATMAPS_LOSS_FACTOR = (cfg.LOSS.HEATMAPS_LOSS_FACTOR)

    if not isinstance(cfg.LOSS.WITH_AE_LOSS, (list, tuple)):
        cfg.LOSS.WITH_AE_LOSS = (cfg.LOSS.WITH_AE_LOSS)

    if not isinstance(cfg.LOSS.PUSH_LOSS_FACTOR, (list, tuple)):
        cfg.LOSS.PUSH_LOSS_FACTOR = (cfg.LOSS.PUSH_LOSS_FACTOR)

    if not isinstance(cfg.LOSS.PULL_LOSS_FACTOR, (list, tuple)):
        cfg.LOSS.PULL_LOSS_FACTOR = (cfg.LOSS.PULL_LOSS_FACTOR)

    cfg.freeze()


def check_config(cfg):
    assert cfg.LOSS.NUM_STAGES == len(cfg.LOSS.WITH_HEATMAPS_LOSS), \
        'LOSS.NUM_SCALE should be the same as the length of LOSS.WITH_HEATMAPS_LOSS'
    assert cfg.LOSS.NUM_STAGES == len(cfg.LOSS.HEATMAPS_LOSS_FACTOR), \
        'LOSS.NUM_SCALE should be the same as the length of LOSS.HEATMAPS_LOSS_FACTOR'
    assert cfg.LOSS.NUM_STAGES == len(cfg.LOSS.WITH_AE_LOSS), \
        'LOSS.NUM_SCALE should be the same as the length of LOSS.WITH_AE_LOSS'
    assert cfg.LOSS.NUM_STAGES == len(cfg.LOSS.PUSH_LOSS_FACTOR), \
        'LOSS.NUM_SCALE should be the same as the length of LOSS.PUSH_LOSS_FACTOR'
    assert cfg.LOSS.NUM_STAGES == len(cfg.LOSS.PULL_LOSS_FACTOR), \
        'LOSS.NUM_SCALE should be the same as the length of LOSS.PULL_LOSS_FACTOR'
    assert cfg.LOSS.NUM_STAGES == len(cfg.TEST.WITH_HEATMAPS), \
        'LOSS.NUM_SCALE should be the same as the length of TEST.WITH_HEATMAPS'
    assert cfg.LOSS.NUM_STAGES == len(cfg.TEST.WITH_AE), \
        'LOSS.NUM_SCALE should be the same as the length of TEST.WITH_AE'
