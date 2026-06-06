"""Run the trained restoration GAN on a full volume (no PSF / metadata needed).

Sliding-window, overlap-tiled, Hann-blended inference so arbitrary stacks
(e.g. 193x512x512) fit in memory. Saves a single-channel restored tif and, if
the source is a hyperstack, a merged ZCYX tif mirroring deconvolve.py output.

Usage::

    python src/gan/infer.py path/to/raw.tif --ckpt models/gan_latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from microscopy_io import load_hyperstack_volume  # noqa: E402
from deconvolve import insert_channel_volume, write_merged_hyperstack  # noqa: E402

from dataset import to_unit, unit_to_tanh, tanh_to_uint8  # noqa: E402
from models import ResUNet3D  # noqa: E402
from train import select_device  # noqa: E402


def _hann_window(shape: tuple[int, int, int]) -> np.ndarray:
    """Separable 3D Hann window (avoids tile-seam artifacts), min-clamped > 0."""
    windows = []
    for n in shape:
        w = np.hanning(n + 2)[1:-1] if n > 1 else np.ones(1)
        windows.append(np.clip(w, 1e-3, None).astype(np.float32))
    wz, wy, wx = windows
    return wz[:, None, None] * wy[None, :, None] * wx[None, None, :]


def _tile_starts(total: int, patch: int, overlap: int) -> list[int]:
    if total <= patch:
        return [0]
    step = max(1, patch - overlap)
    starts = list(range(0, total - patch + 1, step))
    if starts[-1] != total - patch:
        starts.append(total - patch)
    return starts


@torch.no_grad()
def restore_volume(
    gen: torch.nn.Module,
    volume_unit: np.ndarray,
    device: torch.device,
    patch: tuple[int, int, int] = (32, 128, 128),
    overlap: tuple[int, int, int] = (8, 32, 32),
    batch_size: int = 4,
) -> np.ndarray:
    """Restore a [0,1] ZYX volume; returns a [0,1] ZYX volume of the same shape."""
    D, H, W = volume_unit.shape
    pd, ph, pw = patch
    # If the volume is smaller than the patch in any dim, pad (reflect) then crop back.
    pad = [(0, max(0, p - s)) for s, p in zip((D, H, W), patch)]
    vol = np.pad(volume_unit, pad, mode="reflect") if any(hi for _, hi in pad) else volume_unit
    Dp, Hp, Wp = vol.shape

    acc = np.zeros((Dp, Hp, Wp), dtype=np.float32)
    wsum = np.zeros((Dp, Hp, Wp), dtype=np.float32)
    win = _hann_window(patch)

    coords = [
        (z, y, x)
        for z in _tile_starts(Dp, pd, overlap[0])
        for y in _tile_starts(Hp, ph, overlap[1])
        for x in _tile_starts(Wp, pw, overlap[2])
    ]

    batch, positions = [], []

    def flush():
        if not batch:
            return
        inp = torch.from_numpy(np.stack(batch)[:, None]).to(device)  # (B,1,pd,ph,pw)
        out = gen(inp).cpu().numpy()[:, 0]  # (B,pd,ph,pw), in [-1,1]
        out = (out + 1.0) / 2.0
        for (z, y, x), o in zip(positions, out):
            acc[z:z+pd, y:y+ph, x:x+pw] += o * win
            wsum[z:z+pd, y:y+ph, x:x+pw] += win
        batch.clear()
        positions.clear()

    for (z, y, x) in coords:
        patch_vol = vol[z:z+pd, y:y+ph, x:x+pw]
        batch.append(unit_to_tanh(patch_vol).astype(np.float32))
        positions.append((z, y, x))
        if len(batch) >= batch_size:
            flush()
    flush()

    restored = np.clip(acc / np.maximum(wsum, 1e-6), 0.0, 1.0)
    return restored[:D, :H, :W]


def main():
    ap = argparse.ArgumentParser(description="Restore a volume with the trained GAN.")
    ap.add_argument("path", help="Path to raw .tif")
    ap.add_argument("--ckpt", default="models/gan_latest.pt")
    ap.add_argument("--outdir", default="results/gan")
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = select_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    gcfg = cfg.get("generator", {})
    gen = ResUNet3D(
        base_channels=gcfg.get("base_channels", 32),
        depth=gcfg.get("depth", 3),
        norm=gcfg.get("norm", "instance"),
        num_bottleneck_blocks=gcfg.get("num_bottleneck_blocks", 2),
        anisotropic_z=gcfg.get("anisotropic_z", False),
    ).to(device)
    gen.load_state_dict(ckpt["generator"])
    gen.eval()
    print(f"[infer] loaded {args.ckpt} on {device}")

    tif_path = Path(args.path)
    volume, axes, (dxy, dz) = load_hyperstack_volume(tif_path, channel_index=args.channel)
    vol_unit = to_unit(np.asarray(volume))
    patch = tuple(cfg.get("data", {}).get("patch_size", [32, 128, 128]))
    print(f"[infer] restoring volume {vol_unit.shape} ...")
    restored_unit = restore_volume(gen, vol_unit, device, patch=patch)
    restored_u8 = np.clip(restored_unit * 255.0, 0, 255).astype(np.uint8)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    single = outdir / f"{tif_path.stem}_gan_restored.tif"
    tifffile.imwrite(single, restored_u8)
    print(f"[infer] saved {single}")

    # Merged hyperstack (channel replaced) when source was 4D.
    with tifffile.TiffFile(tif_path) as tif:
        image = tif.asarray()
        ij_meta = tif.imagej_metadata
    if image.ndim == 4:
        merged = insert_channel_volume(image, axes, args.channel, restored_u8)
        merged_path = outdir / f"{tif_path.stem}_gan_restored_merged.tif"
        write_merged_hyperstack(merged_path, merged, dxy, dz, ij_meta=ij_meta)
        print(f"[infer] saved {merged_path}")


if __name__ == "__main__":
    main()
