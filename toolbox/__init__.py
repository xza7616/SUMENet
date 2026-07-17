from .metrics import averageMeter, runningScore
from .log import get_logger
from .loss import MscCrossEntropyLoss
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, adjust_lr
from .ranger.ranger import Ranger
from .ranger.ranger913A import RangerVA
from .ranger.rangerqh import RangerQH

def get_dataset(cfg):
    assert cfg['dataset'] in ['SUIM', 'WE3Ds', 'UPLight']

    # if cfg['dataset'] == 'nyuv2':
    #     from .datasets.SUIM import NYUv2
    #     return NYUv2(cfg, mode='train'), NYUv2(cfg, mode='test')
    # if cfg['dataset'] == 'sunrgbd':
    #     from .datasets.sunrgbd import SUNRGBD
    #     return SUNRGBD(cfg, mode='train'), SUNRGBD(cfg, mode='test')
    if cfg['dataset'] == 'WE3Ds':
        from .datasets.WE3Ds import WE3Ds
        return WE3Ds(cfg, mode='train'), WE3Ds(cfg, mode='test')
    if cfg['dataset'] == 'SUIM':
        from .datasets.SUIM import SUIM
        return SUIM(cfg, mode='train'), SUIM(cfg, mode='test')
    if cfg['dataset'] == 'UPLight':
        from .datasets.UPLight_pga import UPLight
        return UPLight(cfg, mode='train'), UPLight(cfg, mode='test')


def get_model(cfg):
    if cfg['model_name'] == 'SUMENet':
        from toolbox.DINO.SUMENet import EncoderDecoder
        return EncoderDecoder(num_classes=8, freeze_backbone=True, topk=4)



