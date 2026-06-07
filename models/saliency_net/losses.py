import torch

from models.saliency_net.model_config import W_CC, W_KL, W_NSS, W_SIM


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


def norm(x):
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)
    x = (x - x_min) / (x_max - x_min + 1e-8)
    return x / (x.sum(dim=(1, 2, 3), keepdim=True) + 1e-8)


def sim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = norm(pred)
    target = norm(target)
    return torch.minimum(pred, target).sum(dim=(1, 2, 3)).mean()


def nss_loss(pred: torch.Tensor, fixation: torch.Tensor) -> torch.Tensor:
    mean = pred.mean(dim=(1, 2, 3), keepdim=True)
    std = pred.std(dim=(1, 2, 3), keepdim=True)
    pred = (pred - mean) / (std + 1e-8)
    mask = fixation > 0.5
    num = (pred * mask).sum(dim=(1, 2, 3))
    den = mask.sum(dim=(1, 2, 3)).clamp(min=1)
    return (num / den).mean()


def saliency_loss(
    pred: torch.Tensor, target: torch.Tensor, fixation: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    kl = kl_loss(pred, target)
    cc = cc_loss(pred, target)
    sim = sim_loss(pred, target)
    nss = nss_loss(pred, fixation)

    total = W_KL * kl - W_CC * cc - W_SIM * sim - W_NSS * nss
    return total, {
        "kl": kl.item(),
        "cc": cc.item(),
        "sim": sim.item(),
        "nss": nss.item(),
        "total": total.item(),
    }
