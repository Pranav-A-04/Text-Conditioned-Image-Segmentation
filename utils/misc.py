def dice_loss(predicted_mask, ground_truth_mask, smooth=1e-6):
    predicted_mask = predicted_mask.view(-1)
    ground_truth_mask = ground_truth_mask.view(-1)
    
    intersection = (predicted_mask * ground_truth_mask).sum()
    dice_score = (2. * intersection + smooth) / (predicted_mask.sum() + ground_truth_mask.sum() + smooth)
    
    return 1 - dice_score