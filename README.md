# SLICE: Skeleton-Length IoU for Crack Evaluation

SLICE scores a road crack segmentation by **centreline length**: how much of the crack's centreline is matched,
missed or spurious, in pixels. Pavement surveys measure linear cracks by length (for example ASTM D6433), and SLICE
follows that measure. It depends little on how wide the masks are drawn, it drops with missed and spurious length,
and the user sets no tolerance radius. **SLICE measures length, not area, so it should be reported with IoU**
(intersection over union: overlap area divided by the combined area of the two masks).

![SLICE on a real road image: input, computation, output, mask width and detection errors](docs/overview.webp)

*CrackForest image 044: the two input masks, their centrelines, the matched centreline, the score and the printed
output, then mask width (SLICE stays, IoU falls), missed length and spurious length (SLICE falls).*

Results in the paper (30 road images, three public datasets):

- **Width.** Predictions drawn 1 to 17 px wide on the GT centreline: SLICE 0.97 to 1.00 (CV 1.0%).
- **Errors.** With 50% of the prediction pixels removed or added: SLICE 0.44 and 0.30. SLICE responds 21.7 times more
  to these errors than to width (clDice 2.0, IoU 0.47); two metrics with a 4-px tolerance radius reach 13.1 and 13.2,
  not significantly different.
- **Annotators.** Five people drawing the ground truth: the U-Net score varies 3.1% with SLICE,
  5.4% with the 4-px metrics, 10.4% with clDice and 19.2% with IoU.
- **Raters.** On constructed pairs of a complete thick mask and an incomplete thin one, SLICE, recall and both
  4-px metrics agree with 99.3% of the choices of five raters.
- **Tolerance.** SLICE has an implicit tolerance of about (w_G + w_P)/2 that grows with mask width,
  so scores are compared within one ground truth convention.

The terms in this list (CV, clDice, 4-px metrics, raters) are explained in the paper; they are not needed to run SLICE.
Words used below:

| Word | Meaning |
|---|---|
| GT (ground truth) | the crack mask drawn by a person |
| prediction | the crack mask from a model, thresholded to crack and background |
| centreline (skeleton) | the mask thinned to lines 1 pixel wide; its length is its number of pixels |
| matched, missed, spurious | centreline length found in both masks, only in the GT, only in the prediction |
| pooled | summed over all images before the ratio is taken |

Everything is in one file, `slice_metric.py`. The `example/` folder has 10 real road images with their
ground truth, a U-Net prediction, and the output the code writes for them.

## 1. Input

![Input: two mask images of the same size](docs/input.webp)

Two masks of the **same size**, **GT first, prediction second**:

- the **ground truth (GT)**;
- the **prediction**, **already thresholded** (for example `P = prob > 0.5`), not a probability map.

As files: PNG, BMP or TIFF, white (or any colour) crack on black, with 0/255 or 0/1 values. On a transparent
background, the opaque pixels are read as crack. A file with more than two grey levels (a probability map, a JPEG) or with no
crack pixels gives a warning. JPEG blurs the edges and is best avoided. As arrays: `bool`, 0/1 or 0/255.
The road image itself is not an input.

For a folder of images, the GT and prediction files need **the same names, extension included**
(`gt/CF_044.png` and `pred/CF_044.png`); subfolders are not searched. The example folder:

```
example/
  images/   CF_004.jpg ... CF_113.jpg    10 road images (for reference only)
  gt/       CF_004.png ... CF_113.png    ground truth masks
  pred/     CF_004.png ... CF_113.png    U-Net predictions, same file names
  output/   slice_results.csv, CF_004_slice.png ...   files written by the code (section 3)
```

## 2. Run

```
git clone https://github.com/Ronald-Kray/slice-metric.git
cd slice-metric
pip install -r requirements.txt          # numpy, scikit-image, Pillow; Python 3.9 or later
```

**One image pair** (GT, then prediction):

```
python slice_metric.py example/gt/CF_044.png example/pred/CF_044.png
```

**A whole folder**, saving a table and one picture per image:

```
python slice_metric.py example/gt example/pred --out results
```

**In Python:**

```python
from slice_metric import slice_metric, slice_report

r = slice_metric("example/gt/CF_044.png", "example/pred/CF_044.png")   # file paths or arrays (bool, 0/1, 0/255)
print(r.SLICE, r.a1, r.a2, r.a3)                                         # 0.4903... 506 459 67

names = ["CF_004", "CF_020", "CF_044"]
rep = slice_report([(f"example/gt/{n}.png", f"example/pred/{n}.png") for n in names])
print(rep["SLICE"], rep["mean_SLICE"], rep["IoU"])                       # 0.6743... 0.7194... 0.4345...
```

With a model, threshold the output first: `slice_metric(gt_mask, prob > 0.5)`.
To use SLICE in another project, copying `slice_metric.py` next to your script is enough.

## 3. Output

![Output: the command and the printed result](docs/output.webp)

One image pair:

```
SLICE 0.490   matched a1 506 px   missed a2 459 px   spurious a3 67 px   (Lg 965, Lp 573)
IoU 0.337   precision 0.455   mask width w_G 2.16 px   w_P 4.54 px
```

A folder, one line per image and the values for the whole set:

```
image                           SLICE     a1     a2     a3    IoU
CF_004.png                      0.862    523     36     48  0.493
CF_020.png                      0.806    466    112      0  0.475
CF_040.png                      0.532    442    327     62  0.343
CF_044.png                      0.490    506    459     67  0.337
CF_060.png                      0.709    501    206      0  0.440
CF_061.png                      0.909    571     51      6  0.413
CF_062.png                      0.944    503     25      5  0.423
CF_088.png                      0.925    516     32     10  0.552
CF_106.png                      0.654    420    206     16  0.375
CF_113.png                      0.739    413    146      0  0.434

10 images   pooled SLICE 0.728   per-image mean 0.757
missed a2/Lg 0.248   spurious a3/Lg 0.033
IoU 0.428   precision 0.512   mask width w_G 2.97 px   w_P 5.34 px
saved results/slice_results.csv and 10 picture(s) in results
```

| Output | Meaning |
|---|---|
| `SLICE` | 1 = all centreline length matched, 0 = nothing matched |
| `a1` | matched length (px): centreline of the overlap of GT and prediction |
| `a2` | missed length (px): `Lg − a1` |
| `a3` | spurious length (px): `Lp − a1` |
| `Lg`, `Lp` | centreline length of the GT and of the prediction |
| pooled SLICE | Σa1 / (Σa1 + Σa2 + Σa3) over all images, the value to report for a dataset |
| per-image mean | mean of the per-image SLICE; an image with an empty mask counts as 0 |
| `IoU`, `precision` | pixel-area IoU and precision, from pixel counts summed over the images |
| `w_G`, `w_P` | mean mask width: mask area divided by centreline length, summed over the images |

A GT image without a prediction file is scored as an empty prediction, so all its length counts as missed.
Prediction files without a GT file are listed and ignored. SLICE is 0 if either mask is empty.

With `--out`, the code also writes `slice_results.csv` (per image: SLICE, a1, a2, a3, Lg, Lp, IoU, precision and the
two mask areas, then a `POOLED` row) and one picture per image. In the pictures, light orange is the GT area, light blue the predicted area and light grey their
overlap. The lines are the centrelines: **green** the matched centreline (a1), **orange** the GT centreline outside
the prediction (missed), **magenta** the predicted centreline outside the GT (spurious). The colours show *where*
the errors are; a2 and a3 are the totals `Lg − a1` and `Lp − a1`, so the numbers of coloured pixels can differ from
a2 and a3. On CF_044 the picture has 143 magenta pixels while a3 = 67; on CF_020 magenta is visible although a3 = 0
(see the `min` in section 4). Short magenta pieces along a found crack mark places where the predicted centreline runs just
outside the GT mask. The pictures in `example/output/`:

| Road image | Picture written by `--out` | SLICE |
|---|---|---|
| <img src="example/images/CF_004.jpg" width="400"> | <img src="example/output/CF_004_slice.png" width="400"> | 0.862 |
| <img src="example/images/CF_020.jpg" width="400"> | <img src="example/output/CF_020_slice.png" width="400"> | 0.806 |
| <img src="example/images/CF_040.jpg" width="400"> | <img src="example/output/CF_040_slice.png" width="400"> | 0.532 |
| <img src="example/images/CF_044.jpg" width="400"> | <img src="example/output/CF_044_slice.png" width="400"> | 0.490 |
| <img src="example/images/CF_060.jpg" width="400"> | <img src="example/output/CF_060_slice.png" width="400"> | 0.709 |
| <img src="example/images/CF_061.jpg" width="400"> | <img src="example/output/CF_061_slice.png" width="400"> | 0.909 |
| <img src="example/images/CF_062.jpg" width="400"> | <img src="example/output/CF_062_slice.png" width="400"> | 0.944 |
| <img src="example/images/CF_088.jpg" width="400"> | <img src="example/output/CF_088_slice.png" width="400"> | 0.925 |
| <img src="example/images/CF_106.jpg" width="400"> | <img src="example/output/CF_106_slice.png" width="400"> | 0.654 |
| <img src="example/images/CF_113.jpg" width="400"> | <img src="example/output/CF_113_slice.png" width="400"> | 0.739 |

## 4. Computing SLICE: a1, a2, a3

![Computing SLICE step by step on CF_044](docs/how_it_works.webp)

The method of the paper (Section 2.4), in the order of the animation, on the whole of CF_044:

1. **Masks.** GT `G` and prediction `P` of the same size, `P` thresholded at one fixed value.
2. **Centrelines.** Thin `G` and `P` to 1-pixel centrelines (Zhang-Suen) and count their pixels: `Lg = 965`, `Lp = 573`.
3. **Overlap first, then its centreline.** Thin the overlap `G ∩ P` and count its pixels `s`;
   the matched length is `a1 = min(s, Lg, Lp) = 506` (green).
   The overlap of the two masks decides which centreline counts as matched, so the user sets no tolerance radius.
4. **Missed, spurious and the score.** `a2 = Lg − a1 = 459`, `a3 = Lp − a1 = 67`,
   `SLICE = a1 / (a1 + a2 + a3) = 506 / 1032 = 0.49`.
5. **Dataset.** Sum a1, a2 and a3 over the images before taking the ratio (pooled SLICE).

The centreline of the overlap can be a little longer than that of `G` or `P`. The `min` keeps a2 and a3 from becoming
negative; it sets a3 to 0 in 3 of the 10 examples (CF_020, CF_060, CF_113), so a3 = 0 does not always mean that no
predicted centreline lies outside the GT.

Fig. 1b of the paper shows the same image cropped to rows 70–189 and columns 0–359, where SLICE = 321 / 643 = 0.50.

## 5. Mask width and detection errors

These figures repeat the constructions of Experiment 1 of the paper on one image, CF_044. The prediction is the
GT centreline redrawn with a plus-shaped 3 × 3 element; the GT is not changed. The paper uses 30 images, so its
values differ slightly.

**Mask width (Experiment 1a).** Prediction widths of 1, 3, 5, 7, 11 and 17 px keep SLICE at 0.98 to 1.00,
while IoU peaks at 0.63 (3 px) and falls to 0.12 (17 px).

![Mask width: SLICE stays, IoU falls](docs/width.webp)

**Missed and spurious length (Experiment 1b).** From the 5-px prediction, 7-px squares remove 10 to 50% of the
pixels (missed length), or straight fragments 1 px wide and 3 to 7 px long (nominal) add 10 to 50% (spurious length).
At 50%, SLICE falls to 0.41 (missed) and 0.28 (spurious); IoU changes less.

![Detection errors: missed and spurious length lower SLICE](docs/errors.webp)

## Preparing the masks

- The prediction has the same size as the GT. Resize it first and note the interpolation
  (the paper used nearest-neighbour interpolation).
- Every prediction is thresholded at **one fixed value** (0.5 on probabilities unless stated), not per image.
  The paper read 8-bit masks as crack above 127, as `slice_metric.py` does.
- No hole filling, removal of small parts or cropping. These steps change the lengths.

## Reporting SLICE

SLICE measures length, not area, so it should be reported with IoU. Following Table 4 of the paper, report for
each GT set separately (`slice_report` and the folder command give all the numbers):

- the **pooled** SLICE as the headline value, with the per-image mean next to it;
- `a2/Lg` and `a3/Lg`, the missed and spurious shares of the GT length;
- IoU and precision;
- the mean mask widths `w_G` and `w_P`;
- the threshold, the resize interpolation, the scikit-image version, the image size and the pixel size.

## Limitations

- **Thick or blob-shaped predictions** can raise SLICE, because extra area adds little centreline length.
  A SLICE gain that comes with a lower IoU and a higher `w_P/w_G` is not counted as an improvement.
- **Implicit tolerance.** Centreline offsets smaller than about `(w_G + w_P)/2` count as matched.
  IoU or HD95 shows the position error.
- **Gaps.** Each gap in a predicted crack counts as missed length. Use a tolerance-based metric if short gaps
  should count as one crack.
- **GT about 1 px wide.** The implicit tolerance is small, so small offsets count as errors; also report a
  tolerance-based centreline IoU (clIoU with a 4-px radius in the paper).
- **One GT set at a time.** The implicit tolerance follows the GT width, so scores against GT sets drawn
  to different conventions are not directly comparable (shifts up to 0.14 in the paper).
- **Pixel length.** Length is a count of skeleton pixels, so a diagonal crack counts up to about 29% fewer pixels
  than its true length. For a length in metres, a chain-code length (1 per side step, √2 per diagonal step) is more
  accurate; in the paper it changed SLICE by 0.01 to 0.02 on average.
- **Crack-free images.** Two empty masks give a per-image SLICE of 0; use the pooled value and `a3`.
- **Very wide predictions.** Tested on linear (longitudinal and transverse) cracks with predictions up to 17 px wide.
  With 21-px predictions containing holes, SLICE fell to 0.38–0.49 in the paper, because the holes add skeleton loops.
  Area-type damage, crack width and severity were not tested.

## Tests

The tests reproduce the test cases (Table 2) and the worked example (Fig. 1b) of the paper,
and the table and pictures in `example/output/`. Other thinning implementations can give other lengths,
so run them once, especially with another scikit-image version:

```
pip install pytest
python -m pytest -q
```

The paper values were computed with scikit-image 0.26 (Python 3.11, NumPy 2.4). The tests also pass with
scikit-image 0.19 to 0.24; with a scikit-image version older than 0.23, install NumPy below 2 (`pip install "numpy<2"`).

## Definition

```
Lg = |skel(G)|,  Lp = |skel(P)|,  s = |skel(G ∩ P)|
a1 = min(s, Lg, Lp),  a2 = Lg − a1,  a3 = Lp − a1
SLICE = a1 / (a1 + a2 + a3) = a1 / (Lg + Lp − a1)
pooled SLICE = Σa1 / (ΣLg + ΣLp − Σa1)
```

`skel` is Zhang-Suen thinning without pruning (`skimage.morphology.skeletonize(mask, method="zhang")`),
and a length is a count of skeleton pixels.

## Data and citation

The road images and ground truth masks in `example/` are 10 images of the CrackForest dataset
(https://github.com/cuilimeng/CrackForest-dataset), which is made available for non-commercial research purposes
only. The masks are the dataset's ground truth saved as 1-bit PNG. Please cite:

- Y. Shi, L. Cui, Z. Qi, F. Meng, Z. Chen, Automatic road crack detection using random structured forests,
  IEEE Transactions on Intelligent Transportation Systems 17(12) (2016) 3434–3445.
- L. Cui, Z. Qi, Z. Chen, F. Meng, Y. Shi, Pavement distress detection using random decision forests,
  International Conference on Data Science (2015) 95–102.

The predictions are the outputs of the U-Net used in the paper, trained on public crack data from earlier studies
that include these images. They serve as realistic inputs for the metric, not as a test of model accuracy.

Citation for SLICE:
H. Ann, H. Park, J.-J. Lee, SLICE (Skeleton-Length IoU for Crack Evaluation): a centreline-length metric
with low mask-width dependence for road crack segmentation, in preparation.

The other data of the paper (annotator masks, rater responses, evaluation scripts) are shared on request.

## License

MIT for the code (see `LICENSE`). The example images and masks keep the CrackForest terms
(non-commercial research use only).
