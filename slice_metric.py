"""SLICE (Skeleton-Length IoU for Crack Evaluation): reference implementation.

SLICE compares a ground-truth (GT) crack mask G with a predicted mask P by
centreline length (Ann, Park, Lee, in preparation; Sections 2.2 and 2.4):

    Lg = |skel(G)|,  Lp = |skel(P)|,  s = |skel(G & P)|
    a1 = min(s, Lg, Lp)          matched length
    a2 = Lg - a1                 missed length
    a3 = Lp - a1                 spurious length
    SLICE = a1 / (a1 + a2 + a3) = a1 / (Lg + Lp - a1)

skel() is Zhang-Suen thinning without pruning
(skimage.morphology.skeletonize, method="zhang"), and a length is the
number of skeleton pixels.

Empty masks: SLICE is 0 if Lg or Lp is 0, including an image in which both
masks are empty. In the pooled value an image with an empty GT adds its
predicted length to a3, and two empty masks add nothing.

Masks must already be binary (step 1 of Section 2.4): resize the prediction
to the GT size, binarize at one fixed threshold for all images, and do not
fill holes, remove components or crop. A mask can be given as an array
(bool, 0/1 or 0/255) or as an image file path; in a file, crack pixels are
those above 127 (or above 0 for a 0/1 image).

Quick use:
    from slice_metric import slice_metric
    print(slice_metric("gt.png", "pred.png").SLICE)

Command line (GT first, then prediction; one image pair, or two folders with the same file names):
    python slice_metric.py gt.png pred.png
    python slice_metric.py gt_folder pred_folder
    python slice_metric.py gt_folder pred_folder --out results   # also save a CSV table and one picture per image
"""
import os
import sys
import warnings
from typing import Iterable, NamedTuple, Tuple, Union

import numpy as np
from skimage.morphology import skeletonize

__version__ = "1.1.0"
__all__ = ["SliceResult", "slice_metric", "slice_pooled", "slice_pool", "slice_report",
           "skeleton_length", "load_mask", "save_overlay"]


class SliceResult(NamedTuple):
    """Lengths in pixels and the SLICE score of one image or one pooled set."""
    a1: int      # matched centreline length
    a2: int      # missed centreline length
    a3: int      # spurious centreline length
    Lg: int      # GT centreline length
    Lp: int      # predicted centreline length
    SLICE: float


def load_mask(path) -> np.ndarray:
    """Read a mask image file as a boolean array.

    Crack pixels are pixels brighter than 127 in any colour channel (white-on-black or
    coloured masks), or any nonzero pixel in a 0/1 image. In an image with a transparent
    background, the opaque pixels are the crack.
    Use lossless files (PNG, BMP, TIFF): a file with more than two grey levels or an empty
    mask gives a warning.
    """
    from PIL import Image  # Pillow is installed with scikit-image
    im = Image.open(path)
    rgba = np.array(im.convert("RGBA"))
    a = rgba[..., :3].max(axis=2)
    if rgba[..., 3].min() == 0 and rgba[..., 3].max() > 127:
        a = rgba[..., 3]                                   # transparent background: opaque pixels are the crack
    levels = np.unique(a)
    m = a > 127 if a.max() > 1 else a > 0
    name = os.path.basename(str(path))
    if len(levels) > 2:
        warnings.warn(f"{name}: {len(levels)} grey levels, not a binary mask; pixels above "
                      f"{127 if a.max() > 1 else 0} are read as crack", stacklevel=2)
    if not m.any():
        warnings.warn(f"{name}: no crack pixels found (empty mask)", stacklevel=2)
    return m


def _binary(mask, name: str) -> np.ndarray:
    """Return a 2D boolean array from an array or an image path; reject masks that are not binary."""
    if isinstance(mask, (str, os.PathLike)):
        return load_mask(mask)
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask, got shape {m.shape}")
    if m.dtype == bool:
        return m
    values = np.unique(m)
    if np.isin(values, (0, 1)).all():
        return m.astype(bool)
    if np.isin(values, (0, 255)).all():
        return m == 255
    raise ValueError(f"{name} must be binary (bool, 0/1 or 0/255), got values {values[:6].tolist()}...; "
                     f"threshold it first, e.g. P = prob > 0.5 (or > 127 for a 0-255 image)")


def skeleton_length(mask) -> int:
    """Centreline length of a binary mask: pixel count of its Zhang-Suen skeleton."""
    m = np.ascontiguousarray(_binary(mask, "mask"))
    return int(skeletonize(m, method="zhang").sum())


def _score(a1: int, Lg: int, Lp: int) -> float:
    return a1 / (Lg + Lp - a1) if (Lg > 0 and Lp > 0) else 0.0


def _pair(G, P):
    g, p = _binary(G, "G"), _binary(P, "P")
    if g.shape != p.shape:
        raise ValueError(f"G and P must have the same size, got {g.shape} and {p.shape}")
    return g, p


def slice_metric(G, P) -> SliceResult:
    """SLICE of one GT mask G and one predicted mask P of the same size.

    G and P are binary arrays or image file paths.

    Returns SliceResult(a1, a2, a3, Lg, Lp, SLICE); lengths are in pixels.
    SLICE is symmetric in G and P and lies in [0, 1].
    """
    g, p = _pair(G, P)
    Lg, Lp = skeleton_length(g), skeleton_length(p)
    s = skeleton_length(g & p)
    a1 = min(s, Lg, Lp)
    return SliceResult(a1, Lg - a1, Lp - a1, Lg, Lp, _score(a1, Lg, Lp))


def slice_pool(results: Iterable[SliceResult]) -> SliceResult:
    """Pool per-image results: sum a1, a2, a3 (and Lg, Lp) before the ratio.

    SLICE_pool = sum(a1) / (sum(Lg) + sum(Lp) - sum(a1)); 0 if nothing is summed.
    """
    a1 = a2 = a3 = 0
    for r in results:
        a1, a2, a3 = a1 + r.a1, a2 + r.a2, a3 + r.a3
    Lg, Lp = a1 + a2, a1 + a3
    n = a1 + a2 + a3
    return SliceResult(a1, a2, a3, Lg, Lp, a1 / n if n else 0.0)


def slice_pooled(pairs: Iterable[Tuple[Union[np.ndarray, str], Union[np.ndarray, str]]]) -> SliceResult:
    """Pooled SLICE of a dataset or survey section given as [(G, P), ...] (arrays or file paths)."""
    return slice_pool(slice_metric(G, P) for G, P in pairs)


def _pixels(g, p):
    """Pixel counts of one image: overlap, union, GT area, predicted area."""
    return np.array([(g & p).sum(), (g | p).sum(), g.sum(), p.sum()], dtype=np.int64)


def slice_report(pairs) -> dict:
    """Everything Table 4 of the paper asks to report, for a set of images [(G, P), ...].

    Returns a dict with the pooled SLICE, the per-image mean SLICE (an image with an empty mask
    counts as 0), the missed and spurious shares a2/Lg and a3/Lg, the pooled IoU and pixel
    precision (from summed pixel counts), the pooled mean mask widths w_G and w_P
    (mask area / centreline length), the number of images and the per-image results.
    """
    rows, px = [], np.zeros(4, np.int64)
    for G, P in pairs:
        g, p = _pair(G, P)
        rows.append(slice_metric(g, p))
        px += _pixels(g, p)
    return _summary(rows, px)


def _summary(rows, px) -> dict:
    pr = slice_pool(rows)
    inter, union, area_g, area_p = (int(v) for v in px)
    div = lambda a, b: a / b if b else 0.0
    return {"SLICE": pr.SLICE, "mean_SLICE": div(sum(r.SLICE for r in rows), len(rows)),
            "a2/Lg": div(pr.a2, pr.Lg), "a3/Lg": div(pr.a3, pr.Lg),
            "IoU": div(inter, union), "precision": div(inter, area_p),
            "w_G": div(area_g, pr.Lg), "w_P": div(area_p, pr.Lp),
            "n": len(rows), "pooled": pr, "per_image": rows}


def save_overlay(G, P, path) -> None:
    """Save a picture of one result on a white background: GT area light orange, prediction area light blue,
    overlap light grey, and three centrelines: skel(G & P) green (matched, a1), skel(G) outside P orange
    (missed), skel(P) outside G magenta (spurious). The colours show where the errors are; a2 and a3 are
    the totals Lg - a1 and Lp - a1."""
    from PIL import Image
    g, p = _pair(G, P)
    g, p = np.ascontiguousarray(g), np.ascontiguousarray(p)
    img = np.full(g.shape + (3,), 255, np.uint8)
    img[g] = (250, 212, 185); img[p] = (190, 215, 245); img[g & p] = (205, 205, 205)
    sg, sp = skeletonize(g, method="zhang"), skeletonize(p, method="zhang")
    img[sg & ~p] = (213, 94, 0)                                   # missed
    img[sp & ~g] = (227, 26, 109)                                 # spurious
    img[skeletonize(g & p, method="zhang")] = (0, 158, 115)       # matched
    Image.fromarray(img).quantize(colors=8, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)


def _main(argv) -> int:
    warnings.formatwarning = lambda msg, *args, **kw: f"warning: {msg}\n"
    try:
        return _run(argv)
    except (ValueError, OSError) as e:
        print(f"error: {e}")
        return 1


def _run(argv) -> int:
    out_dir = None
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 >= len(argv):
            print("--out needs a folder name"); return 2
        out_dir = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
        os.makedirs(out_dir, exist_ok=True)
    if len(argv) != 2:
        print("usage: python slice_metric.py GT.png PRED.png [--out FOLDER]\n"
              "       python slice_metric.py GT_FOLDER PRED_FOLDER [--out FOLDER]")
        return 2
    gt, pred = argv
    ext = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")
    if os.path.isdir(gt) != os.path.isdir(pred):
        print("error: give two files or two folders (GT first, then prediction)"); return 2
    folder = os.path.isdir(gt)
    if folder:
        names = sorted(n for n in os.listdir(gt) if n.lower().endswith(ext))
        preds = set(n for n in os.listdir(pred) if n.lower().endswith(ext))
        if not names:
            print("error: no mask images (.png, .bmp, .tif, .jpg) in the GT folder"); return 1
        if not any(n in preds for n in names):
            print(f"error: none of the {len(names)} GT file names was found in {pred}; "
                  "GT and prediction files need the same names, extension included"); return 1
        items = [(n, os.path.join(gt, n), os.path.join(pred, n) if n in preds else None) for n in names]
        print(f"{'image':30s} {'SLICE':>6s} {'a1':>6s} {'a2':>6s} {'a3':>6s} {'IoU':>6s}")
    else:
        items = [(os.path.basename(pred), gt, pred)]
    rows, px, lines = [], np.zeros(4, np.int64), []
    for n, gpath, ppath in items:
        g = load_mask(gpath)
        p = load_mask(ppath) if ppath else np.zeros_like(g)
        try:
            g, p = _pair(g, p)
        except ValueError as e:
            raise ValueError(f"{n}: {e}") from None
        r = slice_metric(g, p); c = _pixels(g, p)
        rows.append(r); px += c
        i, u, ag, ap = (int(v) for v in c)
        lines.append(f"{n},{r.SLICE:.4f},{r.a1},{r.a2},{r.a3},{r.Lg},{r.Lp},"
                     f"{i / u if u else 0:.4f},{i / ap if ap else 0:.4f},{ag},{ap}")
        if out_dir:
            save_overlay(g, p, os.path.join(out_dir, os.path.splitext(n)[0] + "_slice.png"))
        if folder:
            note = "" if ppath else "  (no prediction file: scored as empty)"
            print(f"{n:30s} {r.SLICE:6.3f} {r.a1:6d} {r.a2:6d} {r.a3:6d} {i / u if u else 0:6.3f}{note}")
    rep = _summary(rows, px)
    if folder:
        extra = sorted(preds - set(names))
        if extra:
            print(f"ignored {len(extra)} prediction file(s) without a GT file: {', '.join(extra[:5])}{' ...' if len(extra) > 5 else ''}")
        missing = sum(1 for _, _, pp in items if pp is None)
        if missing:
            print(f"{missing} of {len(items)} GT images had no prediction file and were scored as empty")
        print(f"\n{rep['n']} images   pooled SLICE {rep['SLICE']:.3f}   per-image mean {rep['mean_SLICE']:.3f}")
        print(f"missed a2/Lg {rep['a2/Lg']:.3f}   spurious a3/Lg {rep['a3/Lg']:.3f}")
    else:
        r = rows[0]
        print(f"SLICE {r.SLICE:.3f}   matched a1 {r.a1} px   missed a2 {r.a2} px   spurious a3 {r.a3} px   (Lg {r.Lg}, Lp {r.Lp})")
    print(f"IoU {rep['IoU']:.3f}   precision {rep['precision']:.3f}   mask width w_G {rep['w_G']:.2f} px   w_P {rep['w_P']:.2f} px")
    if out_dir:
        with open(os.path.join(out_dir, "slice_results.csv"), "w") as f:
            f.write("image,SLICE,a1_matched,a2_missed,a3_spurious,Lg,Lp,IoU,precision,area_G,area_P\n")
            for line in lines:
                f.write(line + "\n")
            if folder:
                r = rep["pooled"]
                f.write(f"POOLED,{r.SLICE:.4f},{r.a1},{r.a2},{r.a3},{r.Lg},{r.Lp},"
                        f"{rep['IoU']:.4f},{rep['precision']:.4f},{int(px[2])},{int(px[3])}\n")
        print(f"saved {os.path.join(out_dir, 'slice_results.csv')} and {len(rows)} picture(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
