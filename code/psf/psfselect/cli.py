"""Command-line interface for the PSF estimation / selection workflow.

Commands
--------
  ingest    Load + validate metadata, write a resolved metadata JSON.
  generate  Render candidate PSFs (models × sweep) and write a manifest.
  compare   Score candidates from a manifest (ground-truth-free criteria).
  rank      Print/write a ranked comparison table + recommendations.
  report    Build the full report (tables, figures, ranked Markdown).
  run       End-to-end: ingest -> generate -> compare -> rank -> report.
  backends  Show which rendering backends are available.

Run ``python -m psfselect.cli <command> --help`` for per-command options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import MODELS, __version__
from .backends import available_backends
from .candidates import generate_candidates, save_manifest, Candidate
from .compare import DEFAULT_WEIGHTS, score_all
from .io_utils import LOG, ensure_dir, load_volume, read_json, read_voxel_um_from_tiff, write_json
from .metadata import load_metadata, validate_samples
from .metadata_leica import extract_metadata
from .parameters import PSFParams, expand_sweep, load_sweep_file, params_from_metadata
from .ranking import rank_candidates, recommend
from .report import build_report


# --------------------------------------------------------------------------- #
# Manifest <-> Candidate (re)hydration so stages can run independently.
# --------------------------------------------------------------------------- #
def _candidates_from_manifest(path: str | Path) -> list[Candidate]:
    payload = read_json(path)
    out = []
    for d in payload["candidates"]:
        out.append(Candidate(
            candidate_id=d["candidate_id"],
            sample_id=d.get("sample_id"),
            model=d["model"],
            backend_used=d.get("backend_used", "unknown"),
            params=d["params"],
            metrics=d["metrics"],
            voxel_um=tuple(d["voxel_um"]),
            perturbation_fwhm=d.get("perturbation_fwhm", {}),
            reblur=d.get("reblur"),
            psf_path=d.get("psf_path"),
            config_path=d.get("config_path"),
            scores=d.get("scores", {}),
        ))
    return out


# --------------------------------------------------------------------------- #
def cmd_backends(args: argparse.Namespace) -> int:
    info = available_backends()
    print("Available PSF rendering backends:")
    for name, ok in info.items():
        print(f"  {name:12s} : {'AVAILABLE' if ok else 'not available'}")
    if not info["epfl"]:
        print("\nEPFL PSF Generator: set PSF_GENERATOR_JAR=/path/to/PSFGenerator.jar to enable.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    samples = validate_samples(load_metadata(args.metadata))
    out = ensure_dir(args.outdir)
    payload = [s.to_dict() for s in samples]
    path = write_json(out / "metadata_resolved.json", payload)
    n_missing = sum(1 for s in samples if s.missing_fields)
    print(f"Ingested {len(samples)} sample(s); {n_missing} had missing fields. -> {path}")
    return 0


def _load_samples(args: argparse.Namespace):
    if getattr(args, "resolved_metadata", None):
        from .metadata import SampleMetadata

        data = read_json(args.resolved_metadata)
        return [SampleMetadata(**d) for d in data]
    return validate_samples(load_metadata(args.metadata))


def cmd_generate(args: argparse.Namespace) -> int:
    samples = _load_samples(args)
    models = args.models or list(MODELS)
    sweep = load_sweep_file(args.sweep) if args.sweep else None
    outdir = ensure_dir(args.outdir)
    psf_dir = ensure_dir(outdir / "psfs")

    all_candidates: list[Candidate] = []
    for s in samples:
        base = params_from_metadata(s, nx=args.nx, nz=args.nz)
        param_sets = expand_sweep(base, sweep)
        raw = None
        if args.with_reblur and s.image_path and Path(s.image_path).exists():
            try:
                raw = load_volume(s.image_path)
                LOG.info("Loaded raw volume for reblur test: %s %s", s.sample_id, raw.shape)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Could not load raw volume for %s: %s", s.sample_id, exc)
        cands = generate_candidates(
            base, models=models, sweep_params=param_sets,
            backend=args.backend, outdir=psf_dir, jar_path=args.jar,
            with_stability=not args.no_stability, raw_volume=raw,
        )
        all_candidates.extend(cands)

    manifest = save_manifest(all_candidates, outdir / "candidates_manifest.json")
    print(f"Generated {len(all_candidates)} candidate(s). Manifest -> {manifest}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    cands = _candidates_from_manifest(args.manifest)
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        for kv in args.weights.split(","):
            k, v = kv.split("=")
            weights[k.strip()] = float(v)
    score_all(cands, weights)
    out = ensure_dir(args.outdir)
    path = save_manifest(cands, out / "candidates_scored.json")
    print(f"Scored {len(cands)} candidate(s) (weights={weights}). -> {path}")
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    cands = _candidates_from_manifest(args.manifest)
    if not (cands and cands[0].scores):
        score_all(cands)
    df = rank_candidates(cands)
    out = ensure_dir(args.outdir)
    df.to_csv(out / "ranking.csv", index=False)
    recs = recommend(cands)
    write_json(out / "recommendations.json", [r.to_dict() for r in recs])
    cols = ["rank_in_sample", "sample_id", "model", "fwhm_lateral_um", "fwhm_z_um",
            "anisotropy", "composite"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    print()
    for r in recs:
        print(f"[{r.sample_id}] -> {r.recommended_model}  (confidence: {r.confidence})")
        if r.escalation_flags:
            print(f"    flags: {'; '.join(r.escalation_flags)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cands = _candidates_from_manifest(args.manifest)
    if not (cands and cands[0].scores):
        score_all(cands)
    artefacts = build_report(cands, args.outdir, make_orthoviews=not args.no_orthoviews)
    print("Report artefacts:")
    for k, v in artefacts.items():
        if isinstance(v, list):
            print(f"  {k}: {len(v)} figure(s)")
        else:
            print(f"  {k}: {v}")
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    """Deconvolve one image with a chosen PSF and show 3 napari windows."""
    from .visualize_napari import (
        deconvolve_multichannel,
        load_channels,
        show_three_windows,
    )

    image_path = Path(args.image)
    # Channel selection: --channel N (single) or --channels 0 1 2 / all (default all)
    if args.channels is not None:
        sel = None if args.channels == ["all"] else [int(c) for c in args.channels]
    elif args.channel is not None:
        sel = [args.channel]
    else:
        sel = None  # all channels
    raw, channel_idx = load_channels(image_path, channels=sel)

    # Voxel size: explicit flags > metadata file > TIFF tags > defaults.
    voxel = None
    if args.voxel:
        dz, dy, dx = (float(v) for v in args.voxel.split(","))
        voxel = (dz, dy, dx)
    base_params: PSFParams | None = None
    if args.metadata:
        samples = validate_samples(load_metadata(args.metadata))
        match = next((s for s in samples if s.sample_id == args.sample_id), samples[0]) \
            if args.sample_id else samples[0]
        base_params = params_from_metadata(match, nx=args.nx, nz=args.nz)
        if voxel is None:
            voxel = (match.voxel_z_um, match.voxel_y_um or match.voxel_x_um, match.voxel_x_um)
    if voxel is None:
        voxel = read_voxel_um_from_tiff(image_path) or (1.0, 0.3, 0.3)
    if base_params is None:
        base_params = PSFParams(
            na=args.na, wavelength_nm=args.wavelength_nm, ni=args.ni, ns=args.ns,
            voxel_z_um=voxel[0], voxel_xy_um=(voxel[1] + voxel[2]) / 2,
            nx=args.nx, nz=args.nz, particle_depth_um=args.depth,
            sample_id=args.sample_id or image_path.stem,
        )
    else:
        # honour CLI optical overrides on top of metadata
        base_params = base_params.copy_with(
            voxel_z_um=voxel[0], voxel_xy_um=(voxel[1] + voxel[2]) / 2,
        )

    if args.crop_xy or args.crop_z:
        from .visualize_napari import _center_crop_czyx

        raw = _center_crop_czyx(raw, args.crop_xy, args.crop_z)
        LOG.info("Cropped raw volume to %s for speed", raw.shape)

    # Per-channel emission wavelengths for PSF rendering (optional).
    wavelengths = None
    if args.wavelengths:
        wavelengths = [float(w) for w in args.wavelengths.split(",")]

    LOG.info("Image %s -> %d channel(s) %s, volume(C,Z,Y,X)=%s, voxel(dz,dy,dx)=%s, model=%s",
             image_path.name, raw.shape[0], channel_idx, raw.shape, voxel, args.model)
    dec, psf = deconvolve_multichannel(
        raw, args.model, base_params, wavelengths_nm=wavelengths,
        backend=args.backend, jar_path=args.jar, iters=args.iters,
    )

    psf_voxel = (base_params.voxel_z_um, base_params.voxel_xy_um, base_params.voxel_xy_um)
    if args.save:
        from .io_utils import save_volume

        out = ensure_dir(args.save)
        # Save as ImageJ multi-channel (CZYX) so channels stay together.
        save_volume(out / f"{image_path.stem}_raw.tif", raw, voxel_um=voxel)
        save_volume(out / f"{image_path.stem}_deconvolved_{args.model}.tif", dec, voxel_um=voxel)
        save_volume(out / f"{image_path.stem}_psf_{args.model}.tif", psf, voxel_um=psf_voxel)
        print(f"Saved raw / deconvolved / PSF volumes ({raw.shape[0]} channel(s)) to {out}")

    if args.no_show:
        return 0
    show_three_windows(
        raw, dec, psf, voxel_um=voxel, psf_voxel_um=psf_voxel,
        channel_indices=channel_idx, title_prefix=f"{args.model}", block=True,
    )
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """One-shot: image file -> metadata -> params -> PSF per model -> deconvolution."""
    from .visualize_napari import deconvolve_multichannel, load_channels, show_three_windows

    image_path = Path(args.image)
    out = ensure_dir(args.outdir)

    # 1) Extract metadata from the file, fill documented defaults, flag gaps.
    meta, info = extract_metadata(image_path)
    sample = validate_samples([meta])[0]
    write_json(out / "metadata_resolved.json", {"sample": sample.to_dict(), "info": info})
    print(f"== Metadata ==  {image_path.name}")
    print(f"  NA={sample.na}  ni={sample.ni} ({info.get('ni_source','default')})  "
          f"ns={sample.ns}  emission={sample.wavelength_nm} nm ({info.get('emission_source','default')})")
    print(f"  voxel(dz,dy,dx)um=({sample.voxel_z_um:.4f},{sample.voxel_y_um:.4f},{sample.voxel_x_um:.4f})  "
          f"objective={sample.objective}")
    if sample.missing_fields:
        print(f"  [defaults applied for: {', '.join(sample.applied_defaults)}]")

    # 2) Build PSF parameters from the metadata.
    base = params_from_metadata(sample, nx=args.nx, nz=args.nz)
    if args.depth is not None:
        base = base.copy_with(particle_depth_um=args.depth)

    # 3) Generate a PSF with each model (auto backend: EPFL JAR where reliable,
    #    psfmodels fallback, e.g. for VRIGL), score, and report.
    models = args.models or list(MODELS)
    psf_dir = ensure_dir(out / "psfs")
    print(f"\n== Generating PSFs ==  models: {', '.join(models)}")
    cands = generate_candidates(base, models=models, outdir=psf_dir, backend=args.backend,
                                jar_path=args.jar, with_stability=args.with_stability)
    save_manifest(cands, out / "candidates_manifest.json")
    score_all(cands)
    artefacts = build_report(cands, out, make_orthoviews=not args.no_orthoviews)
    for c in cands:
        m = c.metrics
        print(f"  {c.model:18s} backend={c.backend_used:9s} "
              f"FWHM lat={m['fwhm_lateral_um']*1000:6.0f} nm  ax={m['fwhm_z_um']*1000:6.0f} nm")

    # 4) Deconvolve the fluorescence channel(s) with the chosen model's PSF.
    if not args.no_deconv:
        sel = None if (args.channels and args.channels == ["all"]) else (
            [int(c) for c in args.channels] if args.channels else
            ([args.channel] if args.channel is not None else [0]))
        raw, ch_idx = load_channels(image_path, channels=sel)
        if args.crop_xy or args.crop_z:
            from .visualize_napari import _center_crop_czyx
            raw = _center_crop_czyx(raw, args.crop_xy, args.crop_z)
        voxel = (base.voxel_z_um, base.voxel_xy_um, base.voxel_xy_um)
        wls = [float(w) for w in args.wavelengths.split(",")] if args.wavelengths else None
        print(f"\n== Deconvolving ==  model={args.deconv_model}  channels={ch_idx}  "
              f"volume(C,Z,Y,X)={raw.shape}  iters={args.iters}")
        dec, psf = deconvolve_multichannel(raw, args.deconv_model, base, wavelengths_nm=wls,
                                           backend=args.backend, jar_path=args.jar, iters=args.iters)
        dec_dir = ensure_dir(out / "deconvolved")
        from .io_utils import save_volume
        save_volume(dec_dir / f"{image_path.stem}_raw.tif", raw, voxel_um=voxel)
        save_volume(dec_dir / f"{image_path.stem}_deconvolved_{args.deconv_model}.tif", dec, voxel_um=voxel)
        save_volume(dec_dir / f"{image_path.stem}_psf_{args.deconv_model}.tif", psf,
                    voxel_um=(base.voxel_z_um, base.voxel_xy_um, base.voxel_xy_um))
        print(f"  saved raw / deconvolved / PSF -> {dec_dir}")
        if args.show:
            show_three_windows(raw, dec, psf, voxel_um=voxel, channel_indices=ch_idx,
                               title_prefix=args.deconv_model, block=True)

    print(f"\nDone. Outputs in {out}/")
    print(f"  report.md  |  comparison_table.csv  |  psfs/  |  "
          f"{'deconvolved/' if not args.no_deconv else '(deconvolution skipped)'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """End-to-end pipeline."""
    out = ensure_dir(args.outdir)
    # ingest
    samples = validate_samples(load_metadata(args.metadata))
    write_json(out / "metadata_resolved.json", [s.to_dict() for s in samples])
    # generate
    models = args.models or list(MODELS)
    sweep = load_sweep_file(args.sweep) if args.sweep else None
    psf_dir = ensure_dir(out / "psfs")
    all_candidates: list[Candidate] = []
    for s in samples:
        base = params_from_metadata(s, nx=args.nx, nz=args.nz)
        param_sets = expand_sweep(base, sweep)
        raw = None
        if args.with_reblur and s.image_path and Path(s.image_path).exists():
            try:
                raw = load_volume(s.image_path)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("raw load failed for %s: %s", s.sample_id, exc)
        all_candidates.extend(generate_candidates(
            base, models=models, sweep_params=param_sets, backend=args.backend,
            outdir=psf_dir, jar_path=args.jar, with_stability=not args.no_stability,
            raw_volume=raw,
        ))
    save_manifest(all_candidates, out / "candidates_manifest.json")
    # compare + report
    score_all(all_candidates)
    save_manifest(all_candidates, out / "candidates_scored.json")
    artefacts = build_report(all_candidates, out, make_orthoviews=not args.no_orthoviews)
    print(f"\nDone. Report: {artefacts['report_md']}")
    print(f"Comparison table: {artefacts['comparison_table']}")
    recs = recommend(all_candidates)
    print("\nRecommendations:")
    for r in recs:
        print(f"  [{r.sample_id}] -> {r.recommended_model}  (confidence: {r.confidence})")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="psfselect", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"psfselect {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common_gen(sp):
        sp.add_argument("--models", nargs="*", choices=MODELS, help="subset of models (default: all)")
        sp.add_argument("--sweep", help="YAML/JSON sweep spec (axis -> list of values)")
        sp.add_argument("--backend", default="auto", choices=["auto", "epfl", "psfmodels", "gaussian"])
        sp.add_argument("--jar", help="path to PSFGenerator.jar (else $PSF_GENERATOR_JAR)")
        sp.add_argument("--nx", type=int, default=128, help="lateral PSF size (default 128)")
        sp.add_argument("--nz", type=int, default=64, help="axial PSF size (default 64)")
        sp.add_argument("--no-stability", action="store_true", help="skip perturbation renders")
        sp.add_argument("--with-reblur", action="store_true",
                        help="run reblur consistency using each sample's image_path")

    sp = sub.add_parser("backends", help="list available rendering backends")
    sp.set_defaults(func=cmd_backends)

    sp = sub.add_parser("ingest", help="load + validate metadata")
    sp.add_argument("metadata", help="metadata JSON/CSV/YAML")
    sp.add_argument("-o", "--outdir", default="psf_outputs")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("generate", help="render candidate PSFs")
    sp.add_argument("metadata", help="metadata JSON/CSV/YAML")
    sp.add_argument("--resolved-metadata", help="use a resolved metadata JSON instead")
    sp.add_argument("-o", "--outdir", default="psf_outputs")
    add_common_gen(sp)
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("compare", help="score candidates from a manifest")
    sp.add_argument("manifest", help="candidates_manifest.json")
    sp.add_argument("-o", "--outdir", default="psf_outputs")
    sp.add_argument("--weights", help="override e.g. 'plausibility=0.5,stability=0.3,reblur=0.2'")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("rank", help="rank candidates + recommendations")
    sp.add_argument("manifest", help="candidates manifest (scored or not)")
    sp.add_argument("-o", "--outdir", default="psf_outputs")
    sp.set_defaults(func=cmd_rank)

    sp = sub.add_parser("report", help="build the full report")
    sp.add_argument("manifest", help="candidates manifest (scored or not)")
    sp.add_argument("-o", "--outdir", default="psf_outputs")
    sp.add_argument("--no-orthoviews", action="store_true", help="skip per-PSF ortho figures")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("pipeline",
                        help="ONE command: image -> metadata -> params -> PSF per model -> deconvolution")
    sp.add_argument("image", help="raw 3D image (TIFF/OME-TIFF/NumPy); metadata auto-extracted")
    sp.add_argument("-o", "--outdir", default="pipeline_outputs")
    sp.add_argument("--models", nargs="*", choices=MODELS, help="PSF models to compute (default: all)")
    sp.add_argument("--deconv-model", default="gibson_lanni", choices=MODELS,
                    help="model whose PSF is used to deconvolve (default gibson_lanni)")
    sp.add_argument("--channel", type=int, default=None,
                    help="fluorescence channel to deconvolve (default 0)")
    sp.add_argument("--channels", nargs="+", help="multiple channels, e.g. 0 1, or 'all'")
    sp.add_argument("--wavelengths", help="per-channel emission nm, comma-separated (overrides metadata)")
    sp.add_argument("--depth", type=float, default=None, help="particle depth into specimen (um)")
    sp.add_argument("--iters", type=int, default=10, help="Richardson-Lucy iterations")
    sp.add_argument("--crop-xy", type=int, default=0, help="center-crop lateral size for deconv (0=full)")
    sp.add_argument("--crop-z", type=int, default=0, help="center-crop axial size for deconv (0=full)")
    sp.add_argument("--no-deconv", action="store_true", help="only compute/compare PSFs")
    sp.add_argument("--with-stability", action="store_true", help="also do perturbation/stability scoring (slower)")
    sp.add_argument("--no-orthoviews", action="store_true", help="skip per-PSF ortho figures")
    sp.add_argument("--backend", default="auto", choices=["auto", "epfl", "psfmodels", "gaussian"])
    sp.add_argument("--jar", help="path to PSFGenerator.jar (else $PSF_GENERATOR_JAR)")
    sp.add_argument("--nx", type=int, default=128, help="lateral PSF size")
    sp.add_argument("--nz", type=int, default=64, help="axial PSF size")
    sp.add_argument("--show", action="store_true", help="open raw/deconvolved/PSF in napari")
    sp.set_defaults(func=cmd_pipeline)

    sp = sub.add_parser("view", help="deconvolve an image with a PSF and open 3 napari windows")
    sp.add_argument("image", help="raw 3D image (TIFF/OME-TIFF/NumPy)")
    sp.add_argument("--model", default="gibson_lanni", choices=MODELS,
                    help="PSF model to render & deconvolve with (default gibson_lanni)")
    sp.add_argument("--channel", type=int, default=None,
                    help="single channel index (default: deconvolve & show ALL channels)")
    sp.add_argument("--channels", nargs="+",
                    help="explicit channel list, e.g. --channels 0 1 2, or 'all'")
    sp.add_argument("--wavelengths",
                    help="per-channel emission wavelengths in nm, comma-separated "
                         "(e.g. 510,580,670); applied in channel order for the PSFs")
    sp.add_argument("--metadata", help="metadata file to source optical params from")
    sp.add_argument("--sample-id", help="sample_id within --metadata to use")
    sp.add_argument("--voxel", help="override voxel size as 'dz,dy,dx' in microns")
    sp.add_argument("--na", type=float, default=0.8)
    sp.add_argument("--wavelength-nm", type=float, default=510.0)
    sp.add_argument("--ni", type=float, default=1.33)
    sp.add_argument("--ns", type=float, default=1.35)
    sp.add_argument("--depth", type=float, default=0.0, help="imaging depth (um)")
    sp.add_argument("--iters", type=int, default=10, help="Richardson-Lucy iterations")
    sp.add_argument("--crop-xy", type=int, default=0, help="center-crop lateral size (0 = full)")
    sp.add_argument("--crop-z", type=int, default=0, help="center-crop axial size (0 = full)")
    sp.add_argument("--backend", default="auto", choices=["auto", "epfl", "psfmodels", "gaussian"])
    sp.add_argument("--jar", help="path to PSFGenerator.jar (else $PSF_GENERATOR_JAR)")
    sp.add_argument("--nx", type=int, default=128, help="lateral PSF size")
    sp.add_argument("--nz", type=int, default=64, help="axial PSF size")
    sp.add_argument("--save", help="also save raw/deconvolved/PSF TIFFs to this dir")
    sp.add_argument("--no-show", action="store_true", help="don't open napari (e.g. for saving only)")
    sp.set_defaults(func=cmd_view)

    sp = sub.add_parser("run", help="end-to-end ingest->generate->compare->report")
    sp.add_argument("metadata", help="metadata JSON/CSV/YAML")
    sp.add_argument("-o", "--outdir", default="psf_outputs")
    sp.add_argument("--no-orthoviews", action="store_true")
    add_common_gen(sp)
    sp.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
