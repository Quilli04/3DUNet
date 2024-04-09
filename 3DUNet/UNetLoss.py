import torch


class DiceLoss(torch.nn.Module):

    def __init__(self, epsilon=1**-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, masks: torch.Tensor):
        # pred.shape: B x num_classes x 128 x 128 x 128 | target.shape: B x 128 x 128 x 128

        total_dice_score = 0
        num_classes = pred.shape[1]

        for i in range(num_classes):
            numerator = 2 * torch.sum(pred[:, i, ...] * masks[:, i, ...]) + self.epsilon
            denominator = torch.sum(pred[:, i, ...]) + torch.sum(masks[:, i, ...]) + self.epsilon

            total_dice_score += 1 - numerator/denominator

        return total_dice_score

