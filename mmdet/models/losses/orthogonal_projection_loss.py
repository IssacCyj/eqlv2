import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmdet.utils import get_root_logger
from functools import partial

from ..builder import LOSSES


@LOSSES.register_module()
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5,):
        super().__init__()
        self.gamma = gamma
        logger = get_root_logger()
        logger.info(f"Using feat loss")

    def forward(self,
                features,
                labels,
                **kwargs):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss

