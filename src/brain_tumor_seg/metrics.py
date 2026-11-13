import torch


@torch.no_grad()
def dice_iou(logits, targets, threshold=0.5, eps=1e-7):
    predictions = torch.sigmoid(logits) >= threshold
    targets = targets >= 0.5
    dims = (1, 2, 3)
    intersection = (predictions & targets).sum(dims).float()
    pred_sum, target_sum = predictions.sum(dims).float(), targets.sum(dims).float()
    union = pred_sum + target_sum - intersection
    # Empty prediction and empty reference scores 1 for that slice.
    dice = torch.where(pred_sum + target_sum == 0, 1.0, (2 * intersection + eps) / (pred_sum + target_sum + eps))
    iou = torch.where(union == 0, 1.0, (intersection + eps) / (union + eps))
    return dice.mean().item(), iou.mean().item()
