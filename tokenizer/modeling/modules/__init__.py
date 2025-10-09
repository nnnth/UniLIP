from .base_model import BaseModel
from .ema_model import EMAModel
from .losses import ReconstructionLoss_Stage1, ReconstructionLoss_Stage2
loss_map = {
    'ReconstructionLoss_Stage1': ReconstructionLoss_Stage1,
    'ReconstructionLoss_Stage2': ReconstructionLoss_Stage2,
}