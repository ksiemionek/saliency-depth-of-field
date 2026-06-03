import torch


def kl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred / (pred.sum(dim=(1, 2, 3), keepdim=True))
    target = target / (target.sum(dim=(1, 2, 3), keepdim=True))
    div = (target * torch.log(1e-8 + target / (pred + 1e-8))).sum(dim=(1, 2, 3))
    return div.mean()


def cc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_c = pred - pred.mean(dim=(1, 2, 3), keepdim=True)
    target_c = target - target.mean(dim=(1, 2, 3), keepdim=True)
    cc = (pred_c * target_c).sum(dim=(1, 2, 3)) / torch.sqrt(
        (pred_c**2).sum(dim=(1, 2, 3)) * (target_c**2).sum(dim=(1, 2, 3)) + 1e-8
    )
    return cc.mean()


def saliency_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    kl = kl_loss(pred, target)
    cc = cc_loss(pred, target)

    total = kl - cc
    return total, {
        "kl": kl.item(),
        "cc": cc.item(),
        "total": total.item(),
    }
