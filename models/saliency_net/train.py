import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import config
from models.data.dataset import SALICONDataset
from models.saliency_net import model_config
from models.saliency_net.losses import saliency_loss
from models.saliency_net.model import SaliencyNet
from models.saliency_net.transforms import (
    fixation_transform,
    heatmap_transform,
    image_transform,
)
from models.utils.device import get_torch_device
from models.utils.image import save_saliency


class Metrics:
    def __init__(self):
        self.totals = {"loss": 0.0, "kl": 0.0, "cc": 0.0, "sim": 0.0, "nss": 0.0}
        self.steps = 0

    def update(self, metrics: dict[str, float]):
        self.totals["loss"] += metrics["total"]
        self.totals["kl"] += metrics["kl"]
        self.totals["cc"] += metrics["cc"]
        self.totals["sim"] += metrics["sim"]
        self.totals["nss"] += metrics["nss"]
        self.steps += 1

    def averages(self) -> dict[str, float]:
        if self.steps == 0:
            return {k: 0.0 for k in self.totals}
        return {k: v / self.steps for k, v in self.totals.items()}


def dataloaders() -> tuple[DataLoader, DataLoader]:
    train_dataset = SALICONDataset(
        split="train",
        img_transform=image_transform,
        heatmap_transform=heatmap_transform,
        fixation_transform=fixation_transform,
    )
    val_dataset = SALICONDataset(
        split="val",
        img_transform=image_transform,
        heatmap_transform=heatmap_transform,
        fixation_transform=fixation_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=model_config.BATCH_SIZE,
        num_workers=model_config.NUM_WORKERS,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=model_config.BATCH_SIZE,
        num_workers=model_config.NUM_WORKERS,
        persistent_workers=True,
    )
    return train_loader, val_loader


def net_optimizer(model: SaliencyNet) -> torch.optim.AdamW:
    for block in model.encoder.model.layer[-4:]:
        for p in block.parameters():
            p.requires_grad = True

    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": model_config.ENCODER_LR},
            {"params": model.decoder.parameters(), "lr": model_config.DECODER_LR},
            {"params": model.proj.parameters(), "lr": model_config.DECODER_LR},
            {"params": model.upsample.parameters(), "lr": model_config.DECODER_LR},
        ],
        weight_decay=model_config.WEIGHT_DECAY,
    )


@torch.no_grad()
def save_epoch_preview(model: SaliencyNet, device: torch.device, epoch: int):
    out_path = f"{config.SALIENCY_OUTPUTS}/saliency_epoch_{epoch}.png"

    img = Image.open(config.IMAGE).convert("RGB")

    original_size = img.size
    x = image_transform(img).unsqueeze(0).to(device)

    model.eval()
    pred = model(x).squeeze().cpu().numpy()
    save_saliency(pred, original_size, out_path)


def train_epoch(
    model: SaliencyNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Metrics:
    model.train()
    metrics = Metrics()
    desc = f"Epoch {epoch + 1}/{model_config.NUM_EPOCHS} [train]"

    for batch in tqdm(loader, desc=desc):
        imgs = batch["image"].to(device)
        heatmaps = batch["heatmap"].to(device)
        fixations = batch["fixation"].to(device)

        optimizer.zero_grad()

        preds = model(imgs)
        loss, step_metrics = saliency_loss(preds, heatmaps, fixations)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.GRAD_CLIP_NORM)
        optimizer.step()

        metrics.update(step_metrics)

    return metrics


@torch.no_grad()
def validate(
    model: SaliencyNet, loader: DataLoader, device: torch.device, epoch: int
) -> Metrics:
    model.eval()
    metrics = Metrics()
    desc = f"Epoch {epoch + 1}/{model_config.NUM_EPOCHS} [val]"

    for batch in tqdm(loader, desc=desc):
        imgs = batch["image"].to(device)
        heatmaps = batch["heatmap"].to(device)
        fixations = batch["fixation"].to(device)

        preds = model(imgs)
        _, step_metrics = saliency_loss(preds, heatmaps, fixations)

        metrics.update(step_metrics)

    return metrics


def main():
    device = get_torch_device()

    train_loader, val_loader = dataloaders()
    model = SaliencyNet(
        model_config.BACKBONE,
        dropout=model_config.DROPOUT,
        decoder_dim=model_config.DECODER_DIM,
    ).to(device)

    optimizer = net_optimizer(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, model_config.NUM_EPOCHS
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(model_config.NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, device, epoch)
        scheduler.step()

        train_avg = train_metrics.averages()
        val_avg = val_metrics.averages()

        print(
            f"[train] loss {train_avg['loss']:.4f} | "
            f"KL {train_avg['kl']:.4f} CC {train_avg['cc']:.4f} "
            f"SIM {train_avg['sim']:.4f} NSS {train_avg['nss']:.4f}"
        )
        print(
            f"[val]   loss {val_avg['loss']:.4f} | "
            f"KL {val_avg['kl']:.4f} CC {val_avg['cc']:.4f} "
            f"SIM {val_avg['sim']:.4f} NSS {val_avg['nss']:.4f}"
        )

        save_epoch_preview(model, device, epoch + 1)

        if val_avg["loss"] < best_val_loss - model_config.MIN_DELTA:
            best_val_loss = val_avg["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), config.CHECKPOINT_BEST)
            print(f"Saved best model: {config.CHECKPOINT_BEST}\n")
        else:
            patience_counter += 1
            print(f"No improvement {patience_counter}/{model_config.PATIENCE}\n")

        if patience_counter >= model_config.PATIENCE:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
