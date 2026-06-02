import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms.functional import rgb_to_grayscale
import imageio

'''https://github.com/ntcongvn/CCBANet/blob/main/libraries/CCBANet/utils/loss.py'''


import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

def compute_sdf(mask_tensor):

    masks_np = mask_tensor.detach().cpu().numpy()
    sdf_np = np.zeros_like(masks_np, dtype=np.float32)

    for i in range(masks_np.shape[0]):

        posmask = masks_np[i, 0] > 0.5
        negmask = ~posmask

        if posmask.any():
            posdis = distance_transform_edt(posmask)
        else:
            posdis = np.zeros_like(posmask, dtype=float)

        if negmask.any():
            negdis = distance_transform_edt(negmask)
        else:
            negdis = np.zeros_like(negmask, dtype=float)

        sdf_np[i, 0] = (negdis - posdis) / 50.0

    return torch.from_numpy(sdf_np).to(mask_tensor.device).float()
class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, reduction=reduction)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class IoULoss(nn.Module):
    def __init__(self, weight=None,  reduction='mean'):
        super(IoULoss, self).__init__()

    def forward(self, pred, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # pred = torch.sigmoid(pred)

        # flatten label and prediction tensors
        pred = pred.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (pred * targets).sum()
        total = (pred + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None,  reduction='mean'):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, reduction)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss


"""BCE + IoU Loss"""


class BceIoULoss(nn.Module):
    def __init__(self, weight=None,  reduction='mean'):
        super(BceIoULoss, self).__init__()
        self.bce = BCELoss(weight, reduction)
        self.iou = IoULoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        iouloss = self.iou(pred, target)

        loss = iouloss + bceloss

        return loss


""" Structure Loss: https://github.com/DengPingFan/PraNet/blob/master/MyTrain.py """


class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        mask = mask.float()
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred_prob = torch.sigmoid(pred)
        inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
        union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


class SDFEdgeLoss(nn.Module):
    def __init__(self, lambda_edge=0.1):
        super(SDFEdgeLoss, self).__init__()
        self.lambda_edge = lambda_edge
        self.l1_loss = nn.L1Loss()


        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]]).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_gradient(self, tensor):
        tensor = tensor.to(torch.float32)

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        device = tensor.device
        dtype = tensor.dtype


        sobel_x = self.sobel_x.to(device=device, dtype=dtype)
        sobel_y = self.sobel_y.to(device=device, dtype=dtype)
        grad_x = F.conv2d(tensor, sobel_x, padding=1)
        grad_y = F.conv2d(tensor, sobel_y, padding=1)
        return grad_x, grad_y

    def forward(self, pred_sdf, gt_sdf):

        loss_sdf = self.l1_loss(pred_sdf, gt_sdf)


        pred_gx, pred_gy = self.compute_gradient(pred_sdf)
        gt_gx, gt_gy = self.compute_gradient(gt_sdf)


        norm_pred = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)
        norm_gt = torch.sqrt(gt_gx ** 2 + gt_gy ** 2 + 1e-6)


        dot_product = (pred_gx * gt_gx) + (pred_gy * gt_gy)
        cos_sim = dot_product / (norm_pred * norm_gt + 1e-6)
        loss_edge = torch.mean(norm_gt * (1.0 - cos_sim))

        return loss_sdf + self.lambda_edge * loss_edge

class HybridSDFLoss(nn.Module):
        def __init__(self, lambda_edge=0.1, lambda_reg=1.0):
            super(HybridSDFLoss, self).__init__()
            self.sdf_edge_loss = SDFEdgeLoss(lambda_edge=1.0)
            self.region_loss = StructureLoss()
            self.lambda_reg = lambda_reg

        def forward(self, pred_sdf, gt_mask):
            gt_sdf = compute_sdf(gt_mask)
            gt_mask = gt_mask.float()

            area_ratio = gt_mask.sum() / (gt_mask.numel() + 1e-8)
            if area_ratio < 0.05:
                dynamic_tau = 10.0
                dynamic_lambda = 0.1
            else:
                dynamic_tau = 5.0
                dynamic_lambda = 0.05

            loss_geometry = self.sdf_edge_loss(pred_sdf, gt_sdf) * dynamic_lambda

            pred_logits = -pred_sdf * dynamic_tau
            loss_region = self.region_loss(pred_logits, gt_mask)

            return (self.lambda_reg * loss_region) + loss_geometry


""" Deep Supervision Loss"""

class DeepSupervisionLoss(nn.Module):
    def __init__(self, typeloss="BceDiceLoss"):
        super(DeepSupervisionLoss, self).__init__()
        self.typeloss = typeloss


        self.hybrid_loss = HybridSDFLoss(lambda_edge=0.1, lambda_reg=1.0)
        self.pure_region_loss = StructureLoss()
        if typeloss == "BceDiceLoss":
            self.criterion = BceDiceLoss()
        elif typeloss == "BceIoULoss":
            self.criterion = BceIoULoss()
        elif typeloss == "StructureLoss":
            self.criterion = StructureLoss()
        elif typeloss == "SDFEdgeLoss":

            self.criterion = HybridSDFLoss(lambda_edge=0.1, lambda_reg=1.0)
        else:
            raise Exception("Loss name is unvalid.")

    def forward(self, pred, gt):
        if isinstance(pred, torch.Tensor):
            if self.typeloss == "SDFEdgeLoss":

                return self.criterion(pred, gt)
            else:
                return self.criterion(torch.sigmoid(pred), gt)

        if isinstance(pred, torch.Tensor):
            return self.criterion(torch.sigmoid(pred), gt)

        n = len(pred)


        total_loss = 0
        for i in range(n):
            p = pred[i]

            if p.shape[2:] != gt.shape[2:]:
                p = F.interpolate(p, size=gt.shape[2:], mode='bilinear', align_corners=False)
            if self.typeloss == "SDFEdgeLoss":
                if i == 1 or i == 2:

                    pred_logits = -p * 5.0
                    total_loss += self.pure_region_loss(pred_logits, gt)

                else:

                    total_loss += self.hybrid_loss(p, gt)
            else:
                 total_loss += self.pure_region_loss(torch.sigmoid(p), gt)

        return total_loss