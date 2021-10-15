import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmdet.utils import get_root_logger
from functools import partial

from ..builder import LOSSES


@LOSSES.register_module()
class BarlowTwinLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        logger = get_root_logger()
        logger.info(f"Using bt loss")

    def forward(self,
                features,
                labels,
                **kwargs):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)
        dot_prod = torch.matmul(features.t(), features)

        mask = torch.eye(dot_prod.shape[0], dot_prod.shape[1]).bool().to(device)
        mask_neg = (~mask).float()
        loss = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        return self.loss_weight * loss

