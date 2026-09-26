"""Test cases of Table 2 (Section 2.4), the worked example of Section 2.2 (Fig. 1b) and the example folder.

Run: python -m pytest -q
"""
import os

import numpy as np
import pytest

from slice_metric import SliceResult, load_mask, skeleton_length, slice_metric, slice_pool, slice_pooled, slice_report

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


def test_accepts_0_1_and_0_255_integers():
    assert slice_metric(G.astype(np.uint8), G.astype(int)) == slice_metric(G, G)
    P = TABLE["left half of G (c 10-59)"][0]
    assert slice_metric(G.astype(np.uint8) * 255, P.astype(np.uint8) * 255) == slice_metric(G, P)


def test_non_contiguous_arrays():
    """Cropped or transposed views give the same result as copies."""
    P = TABLE["G + square r 26-39, c 50-63"][0]
    assert slice_metric(G[:, 5:], P[:, 5:]) == slice_metric(G[:, 5:].copy(), P[:, 5:].copy())
    assert slice_metric(G.T, P.T) == slice_metric(G.T.copy(), P.T.copy())


def test_rejects_non_binary_and_size_mismatch():
    with pytest.raises(ValueError):
        slice_metric(G, G.astype(np.uint8) * 2)
    with pytest.raises(ValueError):
        slice_metric(G, np.where(G, 0.7, 0.2))
    with pytest.raises(ValueError):
        slice_metric(G, np.zeros((H, W + 1), bool))
    with pytest.raises(ValueError):
        slice_metric(G[None], G[None])


# Worked example of Section 2.2 (Fig. 1b): CrackForest image 044, rows 70-189, columns 0-359,
# reference GT against the U-Net output (example/gt/CF_044.png and example/pred/CF_044.png).
HERE = os.path.dirname(os.path.abspath(__file__))


def test_fig1b_example():
    g = load_mask(os.path.join(HERE, "example", "gt", "CF_044.png"))[70:190, 0:360]
    p = load_mask(os.path.join(HERE, "example", "pred", "CF_044.png"))[70:190, 0:360]
    r = slice_metric(g, p)
    assert (r.Lg, r.Lp, skeleton_length(g & p)) == (596, 368, 321)
    assert (r.a1, r.a2, r.a3) == (321, 275, 47)
    assert r.SLICE == pytest.approx(321 / 643)
    assert round(r.SLICE, 2) == 0.50


def test_example_folder_matches_saved_results(tmp_path):
    """Running the command on example/ reproduces example/output/slice_results.csv and its pictures."""
    from PIL import Image
    import slice_metric as sm
    assert sm._main([os.path.join(HERE, "example", "gt"), os.path.join(HERE, "example", "pred"), "--out", str(tmp_path)]) == 0
    saved = open(os.path.join(HERE, "example", "output", "slice_results.csv")).read()
    assert open(tmp_path / "slice_results.csv").read() == saved
    made = sorted(tmp_path.glob("*_slice.png"))
    assert len(made) == 10
    for f in made:
        ref = Image.open(os.path.join(HERE, "example", "output", f.name)).convert("RGB")
        assert np.array_equal(np.array(Image.open(f).convert("RGB")), np.array(ref))


def test_report_of_example_folder():
    """slice_report gives the pooled values printed by the command line (README section 3)."""
    names = sorted(n for n in os.listdir(os.path.join(HERE, "example", "gt")))
    rep = slice_report([(os.path.join(HERE, "example", "gt", n), os.path.join(HERE, "example", "pred", n)) for n in names])
    assert rep["n"] == 10 and (rep["pooled"].a1, rep["pooled"].a2, rep["pooled"].a3) == (4861, 1600, 214)
    assert [round(rep[k], 3) for k in ("SLICE", "mean_SLICE", "a2/Lg", "a3/Lg", "IoU", "precision")] == \
        [0.728, 0.757, 0.248, 0.033, 0.428, 0.512]
    assert (round(rep["w_G"], 2), round(rep["w_P"], 2)) == (2.97, 5.34)


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
    assert sm._main([str(tmp_path / "gt.png"), str(tmp_path / "pred.png"), "--out", str(tmp_path / "res")]) == 0
    assert (tmp_path / "res" / "slice_results.csv").is_file() and (tmp_path / "res" / "pred_slice.png").is_file()


def test_mask_files_and_mistakes(tmp_path, capsys):
    """Alpha-only masks, warnings for grey-level or empty files, and clear messages for common mistakes."""
    from PIL import Image
    import slice_metric as sm
    G = np.zeros((40, 120), bool); G[18:23, 10:110] = True
    rgba = np.zeros((40, 120, 4), np.uint8); rgba[..., :3] = 255; rgba[G, 3] = 255     # white, crack only in alpha
    Image.fromarray(rgba).save(tmp_path / "alpha.png")
    assert np.array_equal(load_mask(tmp_path / "alpha.png"), G)
    rgba[..., :3] = 0                                                                   # black crack on a transparent white background
    rgba[~G, :3] = 255
    Image.fromarray(rgba).save(tmp_path / "alpha2.png")
    assert np.array_equal(load_mask(tmp_path / "alpha2.png"), G)
    soft = (G * 180 + np.linspace(0, 60, 120)).astype(np.uint8)                          # a probability map, not thresholded
    Image.fromarray(soft).save(tmp_path / "soft.png")
    with pytest.warns(UserWarning, match="grey levels"):
        assert np.array_equal(load_mask(tmp_path / "soft.png"), G)
    Image.fromarray(np.zeros((40, 120), np.uint8)).save(tmp_path / "empty.png")
    with pytest.warns(UserWarning, match="empty"):
        load_mask(tmp_path / "empty.png")
    (tmp_path / "g").mkdir(); (tmp_path / "p").mkdir()
    Image.fromarray((G * 255).astype(np.uint8)).save(tmp_path / "g" / "a.png")
    Image.fromarray((G * 255).astype(np.uint8)).save(tmp_path / "p" / "a_pred.png")    # names do not match
    assert sm._main([str(tmp_path / "g"), str(tmp_path / "p")]) == 1
    assert "same names" in capsys.readouterr().out
    Image.fromarray(np.full((20, 60), 255, np.uint8)).save(tmp_path / "p" / "a.png")   # size mismatch
    assert sm._main([str(tmp_path / "g"), str(tmp_path / "p")]) == 1
    assert "a.png: G and P must have the same size" in capsys.readouterr().out
    assert sm._main([str(tmp_path / "g" / "a.png"), str(tmp_path / "p")]) == 2          # a file and a folder
