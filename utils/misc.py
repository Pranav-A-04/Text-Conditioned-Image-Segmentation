import torch

def dice_loss(predicted_mask, ground_truth_mask, smooth=1e-6):
    probs = torch.sigmoid(predicted_mask)
    predicted_mask = probs.view(probs.size(0), -1)
    ground_truth_mask = ground_truth_mask.view(probs.size(0), -1)
    
    intersection = (predicted_mask * ground_truth_mask).sum(dim=1)
    dice_score = (2. * intersection + smooth) / (predicted_mask.sum(dim=1) + ground_truth_mask.sum(dim=1) + smooth)
    # CRITICAL: clamp + nan guard
    dice = torch.clamp(dice_score, 0.0, 1.0)
    dice = torch.nan_to_num(dice, nan=1.0)
    return 1 - dice.mean()

def compute_iou_and_dice(pred, target, eps=1e-6):
    """
    pred, target: [B, H, W] binary tensors {0,1}
    """
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + eps)

    return iou, dice