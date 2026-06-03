from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import config
from models.saliency_net import train_config
from models.saliency_net.data.dataset import SALICONDataset
from models.saliency_net.losses import saliency_loss
from models.saliency_net.model import SaliencyNet
from models.saliency_net.transforms import image_transform, heatmap_transform
from models.utils.device import get_torch_device
from models.utils.image import save_saliency


class Metrics:
    def __init__(self):
        self.loss = 0.0
        self.kl = 0.0
        self.cc = 0.0
        self.steps = 0

    def update(self, metrics: dict[str, float]):
        self.loss += metrics["total"]
        self.kl += metrics["kl"]
        self.cc += metrics["cc"]
        self.steps += 1

    def averages(self) -> dict[str, float]:
        if self.steps == 0:
            return {"loss": 0.0, "kl": 0.0, "cc": 0.0}
        return {
            "loss": self.loss / self.steps,
            "kl": self.kl / self.steps,
            "cc": self.cc / self.steps,
        }


def dataloaders() -> tuple[DataLoader, DataLoader]:
    train_dataset = SALICONDataset(
        split="train",
        img_transform=image_transform,
        heatmap_transform=heatmap_transform,
    )
    val_dataset = SALICONDataset(
        split="val",
        img_transform=image_transform,
        heatmap_transform=heatmap_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_config.BATCH_SIZE,
        num_workers=train_config.NUM_WORKERS,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=train_config.BATCH_SIZE,
        num_workers=train_config.NUM_WORKERS,
        persistent_workers=True,
    )
    return train_loader, val_loader


def net_optimizer(model: SaliencyNet) -> torch.optim.AdamW:
    encoder_params = list(model.encoder.parameters())
    decoder_params = [
        p for n, p in model.named_parameters() if not n.startswith("encoder.")
    ]
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": train_config.ENCODER_LR},
            {"params": decoder_params, "lr": train_config.DECODER_LR},
        ],
        weight_decay=train_config.WEIGHT_DECAY,
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

    if epoch < train_config.FREEZE_EPOCHS:
        model.freeze_encoder()
    else:
        model.unfreeze_encoder()

    metrics = Metrics()
    desc = f"Epoch {epoch + 1}/{train_config.NUM_EPOCHS} [train]"

    for batch in tqdm(loader, desc=desc):
        imgs = batch["image"].to(device)
        heatmaps = batch["heatmap"].to(device)

        optimizer.zero_grad()

        preds = model(imgs)
        loss, step_metrics = saliency_loss(preds, heatmaps)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.GRAD_CLIP_NORM)
        optimizer.step()

        metrics.update(step_metrics)

    return metrics


@torch.no_grad()
def validate(
    model: SaliencyNet, loader: DataLoader, device: torch.device, epoch: int
) -> Metrics:
    model.eval()
    metrics = Metrics()
    desc = f"Epoch {epoch + 1}/{train_config.NUM_EPOCHS} [val]"

    for batch in tqdm(loader, desc=desc):
        imgs = batch["image"].to(device)
        heatmaps = batch["heatmap"].to(device)

        preds = model(imgs)
        _, step_metrics = saliency_loss(preds, heatmaps)

        metrics.update(step_metrics)

    return metrics


def main():
    device = get_torch_device()

    train_loader, val_loader = dataloaders()
    model = SaliencyNet(
        train_config.BACKBONE,
        dropout=train_config.DROPOUT,
    ).to(device)

    optimizer = net_optimizer(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, train_config.NUM_EPOCHS
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(train_config.NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, device, epoch)
        scheduler.step()

        train_avg = train_metrics.averages()
        val_avg = val_metrics.averages()

        print(
            f"train loss {train_avg['loss']:.4f} (kl {train_avg['kl']:.4f}, cc {train_avg['cc']:.4f}) | "
            f"val loss {val_avg['loss']:.4f} (kl {val_avg['kl']:.4f}, cc {val_avg['cc']:.4f})"
        )

        save_epoch_preview(model, device, epoch + 1)

        if val_avg["loss"] < best_val_loss - train_config.MIN_DELTA:
            best_val_loss = val_avg["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), config.CHECKPOINT_BEST)
            print(f"Saved best model: {config.CHECKPOINT_BEST}\n")
        else:
            patience_counter += 1
            print(f"No improvement {patience_counter}/{train_config.PATIENCE}\n")

        if patience_counter >= train_config.PATIENCE:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
