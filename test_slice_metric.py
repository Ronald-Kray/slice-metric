"""Test cases of Table 2 (Section 2.4) and the worked example of Section 2.2 (Fig. 1b).

Run: python -m pytest -q
"""
import os

import numpy as np
import pytest

from slice_metric import SliceResult, skeleton_length, slice_metric, slice_pool, slice_pooled

H, W = 40, 120


def bar(r0, r1, c0, c1):
    """Mask of a 40 x 120 image with rows r0..r1 and columns c0..c1 set (0-based, inclusive)."""
    m = np.zeros((H, W), bool)
    m[r0:r1 + 1, c0:c1 + 1] = True
    return m


G = bar(18, 22, 10, 109)  # 5-px bar, Lg = 97


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def g_plus_square():
    P = G.copy()
    P[26:40, 50:64] = True  # square rows 26-39, columns 50-63
    return P


# Table 2: P, Lp, s = |skel(G & P)|, a1, a2, a3, SLICE (2 d.p.), IoU (2 d.p.)
TABLE = {
    "G shifted down 3 px": (bar(21, 25, 10, 109), 97, 100, 97, 0, 0, 1.00, 0.25),
    "left half of G (c 10-59)": (bar(18, 22, 10, 59), 47, 47, 47, 50, 0, 0.48, 0.50),
    "bar r 15-25, c 10-109": (bar(15, 25, 10, 109), 92, 97, 92, 5, 0, 0.95, 0.45),
    "G + square r 26-39, c 50-63": (g_plus_square(), 100, 97, 97, 0, 3, 0.97, 0.72),
    "whole image": (np.ones((H, W), bool), 83, 97, 83, 14, 0, 0.86, 0.10),
}


def test_gt_length():
    assert skeleton_length(G) == 97


@pytest.mark.parametrize("name", list(TABLE))
def test_table_rows(name):
    P, Lp, s, a1, a2, a3, score, iou_ = TABLE[name]
    r = slice_metric(G, P)
    assert skeleton_length(G & P) == s
    assert (r.Lg, r.Lp, r.a1, r.a2, r.a3) == (97, Lp, a1, a2, a3)
    assert r.SLICE == pytest.approx(a1 / (97 + Lp - a1))
    assert round(r.SLICE, 2) == score
    assert round(float(iou(G, P)), 2) == iou_


def test_pooled_rows_2_and_5():
    pairs = [(G, TABLE["left half of G (c 10-59)"][0]), (G, TABLE["whole image"][0])]
    r = slice_pooled(pairs)
    assert (r.a1, r.a2, r.a3) == (130, 64, 0)
    assert r.Lp == 130 and r.Lg == 194
    assert r.SLICE == pytest.approx(130 / 194)
    assert round(r.SLICE, 2) == 0.67
    # pooling precomputed results gives the same value
    assert slice_pool(slice_metric(g, p) for g, p in pairs) == r


@pytest.mark.parametrize("name", list(TABLE))
def test_symmetric(name):
    P = TABLE[name][0]
    r, q = slice_metric(G, P), slice_metric(P, G)
    assert r.SLICE == q.SLICE and r.a1 == q.a1 and (r.a2, r.a3) == (q.a3, q.a2)


def test_empty_prediction():
    r = slice_metric(G, np.zeros((H, W), bool))
    assert r == SliceResult(0, 97, 0, 97, 0, 0.0)


def test_empty_gt():
    r = slice_metric(np.zeros((H, W), bool), G)
    assert r == SliceResult(0, 0, 97, 0, 97, 0.0)


def test_both_empty():
    E = np.zeros((H, W), bool)
    assert slice_metric(E, E) == SliceResult(0, 0, 0, 0, 0, 0.0)
    assert slice_pooled([(E, E)]).SLICE == 0.0
    assert slice_pooled([]).SLICE == 0.0


def test_pooled_empty_masks():
    """An empty GT adds its predicted length to a3; two empty masks add nothing."""
    E = np.zeros((H, W), bool)
    P = TABLE["left half of G (c 10-59)"][0]
    base = slice_pooled([(G, P)])
    assert slice_pooled([(G, P), (E, E)]) == base
    r = slice_pooled([(G, P), (E, G)])
    assert (r.a1, r.a2, r.a3) == (base.a1, base.a2, base.a3 + 97)
    assert r.SLICE == pytest.approx(47 / (47 + 50 + 97))


def test_accepts_0_1_integers():
    assert slice_metric(G.astype(np.uint8), G.astype(int)) == slice_metric(G, G)


def test_rejects_non_binary_and_size_mismatch():
    with pytest.raises(ValueError):
        slice_metric(G, G.astype(np.uint8) * 255)
    with pytest.raises(ValueError):
        slice_metric(G, np.zeros((H, W + 1), bool))
    with pytest.raises(ValueError):
        slice_metric(G[None], G[None])


# Worked example of Section 2.2 (Fig. 1b): CrackForest image 044, rows 70-189, columns 0-359,
# reference GT against the U-Net output. The masks are not part of this package; set SLICE_EXAMPLE_DIR
# to a folder with reference_gt/CF_044.png and predictions/CF_044/unet.png to run this test.
EXAMPLE_DIR = os.environ.get("SLICE_EXAMPLE_DIR", "")


@pytest.mark.skipif(not os.path.isfile(os.path.join(EXAMPLE_DIR, "reference_gt", "CF_044.png")),
                    reason="example masks not available (set SLICE_EXAMPLE_DIR)")
def test_fig1b_example():
    from PIL import Image

    def load(path):  # 8-bit mask file: crack above 127
        return np.array(Image.open(path).convert("L")) > 127

    g = load(os.path.join(EXAMPLE_DIR, "reference_gt", "CF_044.png"))
    p = load(os.path.join(EXAMPLE_DIR, "predictions", "CF_044", "unet.png"))
    if p.shape != g.shape:  # nearest-neighbour resize to the GT size
        p = np.array(Image.fromarray(p.astype(np.uint8) * 255).resize((g.shape[1], g.shape[0]), Image.NEAREST)) > 127
    g, p = g[70:190, 0:360], p[70:190, 0:360]
    r = slice_metric(g, p)
    assert (r.Lg, r.Lp, skeleton_length(g & p)) == (596, 368, 321)
    assert (r.a1, r.a2, r.a3) == (321, 275, 47)
    assert r.SLICE == pytest.approx(321 / 643)
    assert round(r.SLICE, 2) == 0.50


def test_image_files_and_command_line(tmp_path, capsys):
    """Masks given as image files (0/255 and 0/1) give the same result as arrays; the command line runs."""
    from PIL import Image
    import slice_metric as sm
    G = np.zeros((40, 120), bool); G[18:23, 10:110] = True
    P = np.zeros((40, 120), bool); P[17:26, 10:80] = True
    Image.fromarray((G * 255).astype(np.uint8)).save(tmp_path / "gt.png")
    Image.fromarray(P.astype(np.uint8)).save(tmp_path / "pred.png")          # 0/1 image
    assert sm.slice_metric(str(tmp_path / "gt.png"), tmp_path / "pred.png") == sm.slice_metric(G, P)
    assert sm._main([str(tmp_path / "gt.png"), str(tmp_path / "pred.png")]) == 0
    (tmp_path / "g").mkdir(); (tmp_path / "p").mkdir()
    Image.fromarray((G * 255).astype(np.uint8)).save(tmp_path / "g" / "a.png")
    Image.fromarray((P * 255).astype(np.uint8)).save(tmp_path / "p" / "a.png")
    Image.fromarray((G * 255).astype(np.uint8)).save(tmp_path / "g" / "b.png")   # no prediction for b.png
    (tmp_path / "g" / "notes.txt").write_text("not a mask")
    assert sm._main([str(tmp_path / "g"), str(tmp_path / "p")]) == 0
    out = capsys.readouterr().out
    assert "pooled SLICE" in out and "scored as empty" in out and "notes.txt" not in out
    rgb = np.zeros((40, 120, 3), np.uint8); rgb[G] = (255, 0, 0)                  # red-on-black mask
    Image.fromarray(rgb).save(tmp_path / "red.png")
    assert sm.slice_metric(str(tmp_path / "red.png"), G).SLICE == 1.0
