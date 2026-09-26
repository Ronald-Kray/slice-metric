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
(bool or 0/1) or as an image file path; in a file, crack pixels are those
above 127 (or above 0 for a 0/1 image).

Quick use:
    from slice_metric import slice_metric
    print(slice_metric("gt.png", "pred.png").SLICE)

Command line (one image pair, or two folders with matching file names):
    python slice_metric.py gt.png pred.png
    python slice_metric.py gt_folder pred_folder
    python slice_metric.py gt_folder pred_folder --out results   # also save a CSV table and one picture per image
"""
import os
import sys
from typing import Iterable, NamedTuple, Tuple, Union

import numpy as np
from skimage.morphology import skeletonize

__version__ = "1.0.0"
__all__ = ["SliceResult", "slice_metric", "slice_pooled", "slice_pool", "skeleton_length", "load_mask", "save_overlay"]


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
    coloured masks), or any nonzero pixel in a 0/1 image. Use lossless files (PNG, BMP, TIFF).
    """
    from PIL import Image  # Pillow is installed with scikit-image
    a = np.array(Image.open(path).convert("RGB")).max(axis=2)
    return a > 127 if a.max() > 1 else a > 0


def _binary(mask, name: str) -> np.ndarray:
    """Return a 2D boolean array from an array or an image path; reject masks that are not binary."""
    if isinstance(mask, (str, os.PathLike)):
        return load_mask(mask)
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask, got shape {m.shape}")
    if m.dtype == bool:
        return m
    if not np.isin(np.unique(m), (0, 1)).all():
        raise ValueError(f"{name} must be binary (bool or 0/1); binarize it at a fixed threshold first")
    return m.astype(bool)


def skeleton_length(mask) -> int:
    """Centreline length of a binary mask: pixel count of its Zhang-Suen skeleton."""
    m = _binary(mask, "mask")
    return int(skeletonize(m, method="zhang").sum())


def _score(a1: int, Lg: int, Lp: int) -> float:
    return a1 / (Lg + Lp - a1) if (Lg > 0 and Lp > 0) else 0.0


def slice_metric(G, P) -> SliceResult:
    """SLICE of one GT mask G and one predicted mask P of the same size.

    G and P are binary arrays or image file paths.

    Returns SliceResult(a1, a2, a3, Lg, Lp, SLICE); lengths are in pixels.
    SLICE is symmetric in G and P and lies in [0, 1].
    """
    g, p = _binary(G, "G"), _binary(P, "P")
    if g.shape != p.shape:
        raise ValueError(f"G and P must have the same size, got {g.shape} and {p.shape}")
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


def save_overlay(G, P, path) -> None:
    """Save a picture of one result on a white background: GT area light orange, prediction area light blue,
    overlap light grey, and the centrelines that SLICE counts: matched a1 (green), missed GT centreline (orange),
    spurious predicted centreline (magenta)."""
    from PIL import Image
    g, p = _binary(G, "G"), _binary(P, "P")
    img = np.full(g.shape + (3,), 255, np.uint8)
    img[g] = (250, 212, 185); img[p] = (190, 215, 245); img[g & p] = (205, 205, 205)
    sg, sp = skeletonize(g, method="zhang"), skeletonize(p, method="zhang")
    img[sg & ~p] = (213, 94, 0)                                   # missed
    img[sp & ~g] = (227, 26, 109)                                 # spurious
    img[skeletonize(g & p, method="zhang")] = (0, 158, 115)       # matched
    Image.fromarray(img).quantize(colors=8, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)


def _main(argv) -> int:
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
    rows = []

    def keep(name, g, p, r):
        rows.append((name, r))
        if out_dir:
            save_overlay(g, p, os.path.join(out_dir, os.path.splitext(name)[0] + "_slice.png"))

    def write_csv(pooled=None):
        if not out_dir:
            return
        with open(os.path.join(out_dir, "slice_results.csv"), "w") as f:
            f.write("image,SLICE,a1_matched,a2_missed,a3_spurious,Lg,Lp\n")
            for n, r in rows + ([("POOLED", pooled)] if pooled else []):
                f.write(f"{n},{r.SLICE:.4f},{r.a1},{r.a2},{r.a3},{r.Lg},{r.Lp}\n")
        print(f"saved {os.path.join(out_dir, 'slice_results.csv')} and {len(rows)} picture(s) in {out_dir}")
    if os.path.isdir(gt) and os.path.isdir(pred):
        ext = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")
        names = sorted(n for n in os.listdir(gt) if n.lower().endswith(ext))
        if not names:
            print("no mask images (.png, .bmp, .tif, .jpg) in the GT folder"); return 1
        results = []
        print(f"{'image':30s} {'SLICE':>6s} {'a1':>6s} {'a2':>6s} {'a3':>6s}")
        for n in names:
            g = load_mask(os.path.join(gt, n))
            pp = os.path.join(pred, n)
            if os.path.isfile(pp):
                p = load_mask(pp)
                note = ""
            else:  # no prediction file: scored as an empty prediction, so its GT length counts as missed
                p = np.zeros_like(g)
                note = "  (no prediction file: scored as empty)"
            r = slice_metric(g, p)
            results.append(r); keep(n, g, p, r)
            print(f"{n:30s} {r.SLICE:6.3f} {r.a1:6d} {r.a2:6d} {r.a3:6d}{note}")
        pr = slice_pool(results)
        mean = sum(r.SLICE for r in results) / len(results)
        print(f"\n{len(results)} images   pooled SLICE {pr.SLICE:.3f}   per-image mean {mean:.3f}")
        print(f"missed a2/Lg {pr.a2 / pr.Lg if pr.Lg else 0:.3f}   spurious a3/Lg {pr.a3 / pr.Lg if pr.Lg else 0:.3f}")
        write_csv(pr)
        return 0
    g, p = load_mask(gt), load_mask(pred)
    r = slice_metric(g, p)
    print(f"SLICE {r.SLICE:.3f}   matched a1 {r.a1} px   missed a2 {r.a2} px   spurious a3 {r.a3} px   (Lg {r.Lg}, Lp {r.Lp})")
    keep(os.path.basename(pred), g, p, r)
    write_csv()
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
