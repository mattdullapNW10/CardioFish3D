"""Train the 3D conditional restoration GAN.

Usage::

    python src/gan/train.py --config configs/gan.yaml

Maps raw nuclei volumes to their Richardson-Lucy-deconvolved targets using a
ResUNet3D generator + PatchGAN3D discriminator (LSGAN + L1 + frequency loss).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import PairedDeconvDataset, find_pairs, split_pairs, tanh_to_uint8
from losses import AdversarialLoss, GeneratorLoss
from models import PatchGAN3D, ResUNet3D


# --------------------------------------------------------------------------- #
def select_device(name: str) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else project_root() / path


# --------------------------------------------------------------------------- #
def psnr(fake: np.ndarray, target: np.ndarray) -> float:
    """PSNR for signals in [-1, 1] (range = 2)."""
    mse = float(np.mean((fake - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((2.0 ** 2) / mse))


def ssim_midslice(fake: np.ndarray, target: np.ndarray) -> float | None:
    """SSIM on the central Z-slice (graceful if skimage is unavailable)."""
    try:
        from skimage.metrics import structural_similarity as sk_ssim
    except Exception:
        return None
    z = fake.shape[0] // 2
    a = (fake[z] + 1.0) / 2.0
    b = (target[z] + 1.0) / 2.0
    return float(sk_ssim(a, b, data_range=1.0))


def save_sample_triplet(raw, fake, target, path: Path) -> None:
    """Save a mid-Z-slice raw|fake|target montage as PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    z = raw.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes, [raw[z], fake[z], target[z]], ["raw", "restored", "RL target"]
    ):
        ax.imshow((img + 1.0) / 2.0, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(gen, loader, device) -> dict[str, float]:
    gen.eval()
    psnrs, ssims = [], []
    for raw, target in loader:
        raw = raw.to(device)
        fake = gen(raw).cpu().numpy()
        target = target.numpy()
        for i in range(fake.shape[0]):
            psnrs.append(psnr(fake[i, 0], target[i, 0]))
            s = ssim_midslice(fake[i, 0], target[i, 0])
            if s is not None:
                ssims.append(s)
    gen.train()
    out = {"val_psnr": float(np.mean(psnrs)) if psnrs else 0.0}
    if ssims:
        out["val_ssim"] = float(np.mean(ssims))
    return out


def main():
    ap = argparse.ArgumentParser(description="Train the 3D restoration GAN.")
    ap.add_argument("--config", default=str(project_root() / "configs/gan.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg, gcfg, dscfg = cfg["data"], cfg["generator"], cfg["discriminator"]
    lcfg, tcfg, logcfg = cfg["loss"], cfg["training"], cfg["logging"]

    torch.manual_seed(tcfg.get("seed", 42))
    np.random.seed(tcfg.get("seed", 42))
    device = select_device(tcfg.get("device", "auto"))
    print(f"[train] device = {device}")

    # ---- data ----
    pairs = find_pairs(resolve(dcfg["raw_dir"]), resolve(dcfg["deconv_dir"]))
    print(f"[train] found {len(pairs)} (raw -> deconvolved) pairs")
    train_pairs, val_pairs = split_pairs(pairs, dcfg.get("val_split", 0.2),
                                         seed=tcfg.get("seed", 42))
    print(f"[train] {len(train_pairs)} train / {len(val_pairs)} val volumes")

    patch = tuple(dcfg["patch_size"])
    common = dict(
        raw_root=resolve(dcfg["raw_dir"]),
        deconv_root=resolve(dcfg["deconv_dir"]),
        patch_size=patch,
        channel=dcfg.get("channel", 1),
        patches_per_volume=dcfg.get("patches_per_volume", 16),
    )
    train_ds = PairedDeconvDataset(**common, augment=dcfg.get("augment", True),
                                   pairs=train_pairs, seed=tcfg.get("seed", 42))
    train_loader = DataLoader(train_ds, batch_size=tcfg.get("batch_size", 2),
                              shuffle=True, num_workers=dcfg.get("num_workers", 0),
                              drop_last=True)
    val_loader = None
    if val_pairs:
        val_ds = PairedDeconvDataset(**common, augment=False, pairs=val_pairs,
                                     seed=tcfg.get("seed", 42))
        val_loader = DataLoader(val_ds, batch_size=tcfg.get("batch_size", 2),
                                shuffle=False, num_workers=dcfg.get("num_workers", 0))

    # ---- models ----
    gen = ResUNet3D(
        base_channels=gcfg.get("base_channels", 32),
        depth=gcfg.get("depth", 3),
        norm=gcfg.get("norm", "instance"),
        num_bottleneck_blocks=gcfg.get("num_bottleneck_blocks", 2),
        anisotropic_z=gcfg.get("anisotropic_z", False),
    ).to(device)
    disc = PatchGAN3D(
        base_channels=dscfg.get("base_channels", 64),
        n_layers=dscfg.get("n_layers", 3),
        norm=dscfg.get("norm", "instance"),
        spectral_norm=dscfg.get("spectral_norm", False),
    ).to(device)
    n_g = sum(p.numel() for p in gen.parameters())
    n_d = sum(p.numel() for p in disc.parameters())
    print(f"[train] generator params = {n_g/1e6:.2f}M  discriminator params = {n_d/1e6:.2f}M")

    # ---- losses / optim ----
    adv = AdversarialLoss(lcfg.get("adversarial", "lsgan"))
    g_loss = GeneratorLoss(
        adv,
        lambda_adv=lcfg.get("lambda_adv", 1.0),
        lambda_l1=lcfg.get("lambda_l1", 100.0),
        lambda_freq=lcfg.get("lambda_freq", 1.0),
        lambda_grad=lcfg.get("lambda_grad", 0.0),
    )
    betas = (tcfg.get("beta1", 0.5), tcfg.get("beta2", 0.999))
    opt_g = torch.optim.Adam(gen.parameters(), lr=tcfg.get("lr_g", 2e-4), betas=betas)
    opt_d = torch.optim.Adam(disc.parameters(), lr=tcfg.get("lr_d", 2e-4), betas=betas)

    # ---- dirs / logging ----
    ckpt_dir = resolve(logcfg.get("checkpoint_dir", "models"))
    sample_dir = resolve(logcfg.get("sample_dir", "results/gan"))
    log_dir = resolve(logcfg.get("log_dir", "logs/gan"))
    for d in (ckpt_dir, sample_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "d_loss", "g_total", "g_l1", "g_adv", "val_psnr", "val_ssim"])

    epochs = tcfg.get("epochs", 100)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running = {"d": 0.0, "g": 0.0, "l1": 0.0, "adv": 0.0}
        n_batches = 0
        for raw, target in train_loader:
            raw, target = raw.to(device), target.to(device)

            # ---- D step ----
            opt_d.zero_grad(set_to_none=True)
            fake = gen(raw).detach()
            real_logits = disc(raw, target)
            fake_logits = disc(raw, fake)
            d_loss = adv.discriminator_loss(real_logits, fake_logits)
            d_loss.backward()
            opt_d.step()

            # ---- G step ----
            opt_g.zero_grad(set_to_none=True)
            fake = gen(raw)
            fake_logits = disc(raw, fake)
            gl, logs = g_loss(fake_logits, fake, target)
            gl.backward()
            opt_g.step()

            running["d"] += float(d_loss.detach())
            running["g"] += logs["g_total"]
            running["l1"] += logs["g_l1"]
            running["adv"] += logs["g_adv"]
            n_batches += 1

        n_batches = max(1, n_batches)
        d_avg = running["d"] / n_batches
        g_avg = running["g"] / n_batches
        l1_avg = running["l1"] / n_batches
        adv_avg = running["adv"] / n_batches

        val_metrics = {}
        if val_loader is not None and epoch % logcfg.get("val_every", 5) == 0:
            val_metrics = validate(gen, val_loader, device)

        print(
            f"[epoch {epoch:03d}/{epochs}] "
            f"D={d_avg:.3f} G={g_avg:.3f} L1={l1_avg:.4f} adv={adv_avg:.3f} "
            + (f"PSNR={val_metrics.get('val_psnr', 0):.2f} " if val_metrics else "")
            + f"({time.time()-t0:.1f}s)"
        )
        csv_writer.writerow([
            epoch, f"{d_avg:.5f}", f"{g_avg:.5f}", f"{l1_avg:.5f}", f"{adv_avg:.5f}",
            f"{val_metrics.get('val_psnr', '')}", f"{val_metrics.get('val_ssim', '')}",
        ])
        csv_file.flush()

        # ---- sample triplet ----
        if epoch % logcfg.get("sample_every", 5) == 0:
            gen.eval()
            with torch.no_grad():
                raw, target = next(iter(val_loader or train_loader))
                raw, target = raw.to(device), target.to(device)
                fake = gen(raw)
            save_sample_triplet(
                raw[0, 0].cpu().numpy(), fake[0, 0].cpu().numpy(),
                target[0, 0].cpu().numpy(), sample_dir / f"epoch_{epoch:03d}.png",
            )
            gen.train()

        # ---- checkpoint ----
        if epoch % logcfg.get("save_every", 10) == 0 or epoch == epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "generator": gen.state_dict(),
                    "discriminator": disc.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "config": cfg,
                },
                ckpt_dir / f"gan_epoch_{epoch:03d}.pt",
            )
            torch.save({"generator": gen.state_dict(), "config": cfg},
                       ckpt_dir / "gan_latest.pt")

    csv_file.close()
    print(f"[train] done. checkpoints in {ckpt_dir}, logs in {log_dir}")


if __name__ == "__main__":
    main()
