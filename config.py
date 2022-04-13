from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 2
_C.SYSTEM.NUM_WORKERS = 8

_C.MODEL = CN()
_C.MODEL.NAME = "unet++"
_C.MODEL.BACKBONE = "timm-efficientnet-b4"

_C.TRAIN = CN()
_C.TRAIN.BASE_LR = 0.003
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.NUM_EPOCHS = 50

_C.VAL = CN()
_C.VAL.BATCH_SIZE = 4

_C.DATASET = CN()
_C.DATASET.CLASSES = [41, 76, 90, 124, 161]
_C.DATASET.IMG_WIDTH = 2*18*32
_C.DATASET.IMG_HEIGHT = 2*14*32
_C.DATASET.ROOT = "/home/johann/sonstiges/comma10k-segmenation-pytorch/comma10k"
_C.DATASET.CHANNEL2CLASS = ["road", "lane markings", "my car", "undrivable", "movable"]
_C.DATASET.CHANNEL2COLOR = ["#402020", "#ff0000", "#cc00ff", "#808060", "#00ff66"]


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
