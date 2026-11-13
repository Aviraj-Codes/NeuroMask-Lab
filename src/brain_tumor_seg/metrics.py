import torch
@torch.no_grad()
def segmentation_metrics(logits, targets, threshold=0.5, eps=1e-7):
    predictions = torch.sigmoid(logits) >= threshold
    targets = targets >= 0.5
    dims = (1, 2, 3)
    intersection = (predictions & targets).sum(dims).float()
    pred_sum, target_sum = predictions.sum(dims).float(), targets.sum(dims).float()
    union = pred_sum + target_sum - intersection
    true_positive = intersection
    false_positive = pred_sum - intersection
    false_negative = target_sum - intersection
    dice = torch.where(pred_sum + target_sum == 0, 1.0, (2 * intersection + eps) / (pred_sum + target_sum + eps))
    iou = torch.where(union == 0, 1.0, (intersection + eps) / (union + eps))
    # Empty/empty slices score 1; recall is 1 when the reference contains no tumor.
    precision = torch.where(pred_sum == 0, (target_sum == 0).float(), true_positive / (pred_sum + eps))
    recall = torch.where(target_sum == 0, 1.0, true_positive / (target_sum + eps))
    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }