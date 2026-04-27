# ultralytics_custom/utils/wiou.py
# WIoU Loss — fixed with IoU clamping for stable early training

import torch
from ultralytics.utils.loss import BboxLoss


class WIoUBboxLoss(BboxLoss):
    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask,
                *args, **kwargs):

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pb = pred_bboxes[fg_mask]
        tb = target_bboxes[fg_mask]

        # Standard IoU first
        iou = self._iou(pb, tb)

        with torch.no_grad():
            # Clamp IoU to prevent division by near-zero
            iou_clamped = iou.detach().clamp(min=0.01, max=1.0)
            iou_mean    = iou_clamped.mean().clamp(min=0.01)
            # Beta: hard examples get higher weight
            # clamp beta to prevent explosion
            beta = (iou_clamped / iou_mean).pow(4).clamp(min=0.1, max=10.0)

        loss_iou = ((1 - iou) * beta * weight).sum() / target_scores_sum

        _, loss_dfl = super().forward(
            pred_dist, pred_bboxes, anchor_points,
            target_bboxes, target_scores, target_scores_sum, fg_mask,
            *args, **kwargs
        )

        return loss_iou, loss_dfl

    def _iou(self, pred, target, eps=1e-7):
        ix1 = torch.max(pred[:, 0], target[:, 0])
        iy1 = torch.max(pred[:, 1], target[:, 1])
        ix2 = torch.min(pred[:, 2], target[:, 2])
        iy2 = torch.min(pred[:, 3], target[:, 3])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        pa = (pred[:, 2]-pred[:, 0]) * (pred[:, 3]-pred[:, 1])
        ta = (target[:, 2]-target[:, 0]) * (target[:, 3]-target[:, 1])
        return inter / (pa + ta - inter + eps)
