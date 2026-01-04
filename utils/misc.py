import torch

def dice_loss(predicted_mask, ground_truth_mask, smooth=1e-6):
    probs = torch.sigmoid(predicted_mask)
    predicted_mask = probs.view(probs.size(0), -1)
    ground_truth_mask = ground_truth_mask.view(probs.size(0), -1)
    
    intersection = (predicted_mask * ground_truth_mask).sum(dim=1)
    dice_score = (2. * intersection + smooth) / (predicted_mask.sum(dim=1) + ground_truth_mask.sum(dim=1) + smooth)
    
    return 1 - dice_score.mean()