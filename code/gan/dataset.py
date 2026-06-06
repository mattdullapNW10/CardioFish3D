"""Paired (raw -> Richardson-Lucy-deconvolved) dataset for the restoration GAN.

Each item is a spatially-aligned 3D patch pair ``(raw_B, sharp_S)`` cropped at
identical coordinates from a raw nuclei volume (channel 1) and its
``{stem}_deconvolved.tif`` target. Volumes are normalized to ``[-1, 1]``.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

# Reuse the project's IO helpers (src/ on path, matching existing scripts).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from microscopy_io import iter_raw_tifs, load_hyperstack_volume  # noqa: E402


def to_unit(vol: np.ndarray) -> np.ndarray:
    """uint8/whatever -> float32 in [0, 1] (matches the codebase /255 convention)."""
    vol = vol.astype(np.float32)
    if vol.max() > 1.0:
        vol = vol / 255.0
    return np.clip(vol, 0.0, 1.0)


def unit_to_tanh(vol: np.ndarray) -> np.ndarray:
    return vol * 2.0 - 1.0


def tanh_to_uint8(vol: np.ndarray | torch.Tensor) -> np.ndarray:
    """[-1, 1] -> uint8 [0, 255] for saving."""
    if isinstance(vol, torch.Tensor):
        vol = vol.detach().cpu().numpy()
    vol = (vol + 1.0) / 2.0
    return np.clip(vol * 255.0, 0, 255).astype(np.uint8)


def find_pairs(
    raw_root: Path, deconv_root: Path, suffix: str = "_deconvolved"
) -> list[tuple[Path, Path]]:
    """Match each raw .tif to its ``{stem}{suffix}.tif`` deconvolved target.

    Targets are matched by filename stem; raws without a target are skipped.
    """
    raw_root = Path(raw_root)
    deconv_root = Path(deconv_root)
    targets = {p.stem: p for p in iter_raw_tifs(deconv_root)}
    pairs: list[tuple[Path, Path]] = []
    for raw in iter_raw_tifs(raw_root):
        key = f"{raw.stem}{suffix}"
        tgt = targets.get(key)
        # The deconvolved single-channel file (not the *_merged hyperstack).
        if tgt is not None and "merged" not in tgt.stem:
            pairs.append((raw, tgt))
    return pairs


@lru_cache(maxsize=8)
def _load_raw_volume(path_str: str, channel: int) -> np.ndarray:
    vol, _, _ = load_hyperstack_volume(Path(path_str), channel_index=channel)
    return to_unit(np.asarray(vol))


@lru_cache(maxsize=8)
def _load_target_volume(path_str: str) -> np.ndarray:
    vol = tifffile.imread(path_str)
    if vol.ndim != 3:
        raise ValueError(f"Expected single-channel ZYX target, got {vol.shape} for {path_str}")
    return to_unit(np.asarray(vol))


class PairedDeconvDataset(Dataset):
    def __init__(
        self,
        raw_root: str | Path,
        deconv_root: str | Path,
        patch_size: tuple[int, int, int] = (32, 128, 128),
        channel: int = 1,
        patches_per_volume: int = 16,
        augment: bool = True,
        pairs: list[tuple[Path, Path]] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.raw_root = Path(raw_root)
        self.deconv_root = Path(deconv_root)
        self.patch_size = tuple(patch_size)
        self.channel = channel
        self.patches_per_volume = patches_per_volume
        self.augment = augment
        self.pairs = pairs if pairs is not None else find_pairs(self.raw_root, self.deconv_root)
        if not self.pairs:
            raise RuntimeError(
                f"No (raw -> deconvolved) pairs found under {self.raw_root} / {self.deconv_root}"
            )
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_volume

    # -- helpers ----------------------------------------------------------
    def _random_crop_coords(self, shape: tuple[int, int, int]) -> tuple[slice, ...]:
        starts = []
        for dim, p in zip(shape, self.patch_size):
            if dim <= p:
                starts.append(0)
            else:
                starts.append(int(self.rng.integers(0, dim - p + 1)))
        return tuple(slice(s, s + p) for s, p in zip(starts, self.patch_size))

    @staticmethod
    def _pad_to_patch(vol: np.ndarray, patch: tuple[int, int, int]) -> np.ndarray:
        pads = [(0, max(0, p - d)) for d, p in zip(vol.shape, patch)]
        if any(hi > 0 for _, hi in pads):
            vol = np.pad(vol, pads, mode="reflect")
        return vol

    def _augment_pair(self, raw: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Flips along any of the 3 axes.
        for axis in range(3):
            if self.rng.random() < 0.5:
                raw = np.flip(raw, axis=axis)
                tgt = np.flip(tgt, axis=axis)
        # 90-degree rotations in the YX plane (axes 1, 2).
        k = int(self.rng.integers(0, 4))
        if k:
            raw = np.rot90(raw, k=k, axes=(1, 2))
            tgt = np.rot90(tgt, k=k, axes=(1, 2))
        # Mild intensity jitter applied identically to both.
        scale = 1.0 + float(self.rng.uniform(-0.1, 0.1))
        raw = np.clip(raw * scale, 0.0, 1.0)
        tgt = np.clip(tgt * scale, 0.0, 1.0)
        return np.ascontiguousarray(raw), np.ascontiguousarray(tgt)

    # -- Dataset API ------------------------------------------------------
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raw_path, tgt_path = self.pairs[idx // self.patches_per_volume]
        raw_vol = _load_raw_volume(str(raw_path), self.channel)
        tgt_vol = _load_target_volume(str(tgt_path))
        if raw_vol.shape != tgt_vol.shape:
            raise ValueError(
                f"Shape mismatch {raw_vol.shape} vs {tgt_vol.shape} for {raw_path.name}"
            )

        raw_vol = self._pad_to_patch(raw_vol, self.patch_size)
        tgt_vol = self._pad_to_patch(tgt_vol, self.patch_size)

        crop = self._random_crop_coords(raw_vol.shape)
        raw = raw_vol[crop]
        tgt = tgt_vol[crop]

        if self.augment:
            raw, tgt = self._augment_pair(raw, tgt)

        raw = unit_to_tanh(raw)[None]  # add channel dim -> (1, D, H, W)
        tgt = unit_to_tanh(tgt)[None]
        return torch.from_numpy(raw.copy()), torch.from_numpy(tgt.copy())


def split_pairs(
    pairs: list[tuple[Path, Path]], val_split: float, seed: int = 42
) -> tuple[list, list]:
    """Deterministically split a pair list into (train, val)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(round(len(pairs) * val_split))) if len(pairs) > 1 else 0
    val_idx = set(idx[:n_val].tolist())
    train = [p for i, p in enumerate(pairs) if i not in val_idx]
    val = [p for i, p in enumerate(pairs) if i in val_idx]
    return train, val
