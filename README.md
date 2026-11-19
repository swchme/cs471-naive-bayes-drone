# Naive Bayes Drone Object Classifier (CS471 Final Project)

A drone camera's detector draws a box around something on the ground. This project
decides what that something is: **person**, **car**, or **other**. When it isn't
confident enough, it says **"unable to identify"** instead of guessing.

The core algorithm is a **Gaussian Naive Bayes classifier written from scratch**
in plain Python, with no scikit-learn or numpy. Each box is described by six
features: the red, green and blue share of its color, box width, box height, and
aspect ratio (width ÷ height). It is compared against a simple baseline rule:
**"tall and narrow means person."**

> **Checkpoint status ("Ugly but Functional"):** the core algorithm, a working
> command-line demo, real-data evaluation and automated correctness tests are
> complete. See [PROGRESS.md](PROGRESS.md) for completed work, remaining work and
> contributions.

---

## Quick start (no installs needed)

Requires **Python 3.10+** (tested on 3.12.10). From this folder:

```bash
python final-project.py                      # demo on 30 built-in drone examples
python -m unittest discover -s tests -v      # correctness tests
```

The demo prints, for each held-out example:
- the posterior probability of each class;
- Naive Bayes's choice, or "unable to identify";
- the tall-and-narrow rule's choice.

It then prints accuracy, a confusion matrix, and a comparison with the rule. A
saved copy of the output is in [demo_output.txt](demo_output.txt).

## Full setup (real VisDrone data and charts)

The real-data scripts use the included `visdrone_features.csv`, so **you don't
need to download any images** to run them. Pillow is only needed to rebuild that
CSV from images, and matplotlib only to draw charts.

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
python -m pip install -r requirements.txt
```

| Command | What it does | Needs |
|---|---|---|
| `python final-project.py` | Demo on 30 hand-made examples: posteriors, threshold, accuracy, confusion matrix, rule comparison | nothing |
| `python visdrone_nb.py` | Trains and tests on **17,316 real drone crops**, split by image. Prints accuracy and confusion matrices for the natural and balanced class mixes, RGB vs HSV color features, a threshold sweep, a learning curve and 5-fold cross-validation (~35 s). Saves 3 charts. | nothing (matplotlib for charts) |
| `python visdrone_nb.py --only results` | Just the main train/test report (choose from `results sweep curve cv`) | nothing |
| `python export_nb_model.py` | Trains on all crops and saves priors, means and variances to `nb_model.json` | nothing |
| `python visdrone_features.py --data-dir PATH` | Rebuilds `visdrone_features.csv` from VisDrone images | Pillow + data |
| `python -m unittest discover -s tests -v` | 13 correctness tests (2 skip themselves without Pillow) | nothing |

### Rebuilding the CSV from images (optional)
1. Download **VisDrone2019-DET-val** (81.6 MB). Either use the
   [Ultralytics mirror](https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip)
   or the official [VisDrone dataset page](https://github.com/VisDrone/VisDrone-Dataset).
2. Unzip it. The folder contains `annotations/` and `images/`.
3. Run `python visdrone_features.py --data-dir path/to/VisDrone2019-DET-val`.

   `data/` is git-ignored, so `data/VisDrone2019-DET-val` (the default) is a
   convenient place to unzip it.

---

## How the algorithm works

For a box with features **x** = (x₁ … x₆), Bayes' rule gives

&nbsp;&nbsp;&nbsp;&nbsp;**P(class | x) ∝ P(class) · P(x | class)**

**The "naive" assumption:** features are treated as independent given the class,
so the likelihood is a product of one bell curve (Gaussian) per feature:

&nbsp;&nbsp;&nbsp;&nbsp;**P(x | class) = ∏ⱼ N(xⱼ ; μ<sub>class,j</sub>, σ²<sub>class,j</sub>)**

### Training (`GaussianNaiveBayes.fit`)
For each class it stores three things:
- the **prior**: the share of training examples in that class;
- the **mean** of each feature;
- the **variance** of each feature, plus a tiny floor so it can never be 0.

### Prediction (`GaussianNaiveBayes.posteriors` / `predict`)
1. Add the log prior and the log Gaussian likelihood of every feature. Logs avoid
   rounding tiny probabilities to 0.
2. Convert back to probabilities and normalize with the **log-sum-exp trick**, so
   the three posteriors add up to 1.
3. Pick the class with the highest posterior. **If that posterior is below 0.70,
   output "unable to identify."**

### Evaluation
- **Held-out test sets** that the model never trains on.
- On real data, the split is **by image**, so crops from one photo are never on both
  sides.
- **Accuracy** is reported two ways: counting "unable to identify" as wrong, and on
  only the examples it answered.
- A **confusion matrix** counts every example, including "unable to identify".

## Correctness verification

`tests/test_naive_bayes.py` contains 13 tests. Each docstring states the expected
result and why. The key ones:

| # | Test | Expected result | Why it shows correctness |
|---|---|---|---|
| 1 | `test_training_learns_priors_means_variances` | Priors 1/3 each, means 2/6/10, variances 1/1/1 on a tiny dataset | Matches values computed by hand: training is correct |
| 2 | `test_posterior_matches_hand_calculation` | P(person \| x=2) = 1/(1+e⁻⁸+e⁻³²) = 0.99966…, to 9 decimals | Bayes' rule, the Gaussian likelihood and normalization match a hand calculation |
| 3 | `test_priors_shift_the_answer` | When likelihoods tie, posterior ratio = prior ratio = 2 exactly | Priors are really multiplied in |
| 4 | `test_posteriors_always_sum_to_one` | Sum = 1 for any input, including ±1,000,000 | Normalization and numerical stability (log-sum-exp) |
| 5 | `test_ambiguous_input_is_unable_to_identify` | Halfway between two classes gives 0.5 / 0.5, so **"unable to identify"** | The confidence threshold works |
| 6 | `test_clear_input_gets_the_right_class` | A point at the "other" mean gives **"other"** above 0.70 | Confident, correct answers come through |
| 7 | `test_documented_results` | Demo: Naive Bayes 7/9, rule 6/9; red car → "other", sign post → "unable" | The demo output is reproducible |
| 8 | `test_split_by_image_never_leaks` (+ the real CSV) | No image appears in both train and test | The real-data evaluation is honest |
| 9 | `test_solid_red_image`, `test_gray_image_goes_to_gray_bin` | Pure red gives red share 1.0; gray gives 1/3 each and the HSV gray bin | Feature extraction is correct |

Expected output: `Ran 13 tests ... OK` (or `OK (skipped=2)` without Pillow).

## Results so far

**Hand-made demo** (30 examples; 21 train / 9 test):

| | Accuracy |
|---|---|
| Naive Bayes, "unable" counted wrong | 7/9 = 77.8% |
| Naive Bayes, on the 8 it answered | 7/8 = 87.5% |
| Tall-and-narrow rule | 6/9 = 66.7% |

**Real VisDrone crops** (17,316 crops from 548 images; 5-fold cross-validation by
image; full output in [visdrone_results.txt](visdrone_results.txt)):

| Natural class mix | Accuracy |
|---|---|
| Naive Bayes (RGB), forced to always answer | 66.5% ± 2.9 |
| Naive Bayes (RGB), on the 54.5% it answers at 0.70 | 76.5% ± 3.3 |
| Tall-and-narrow rule | 66.7% ± 4.2 |
| Always guess "car" | 60.9% ± 2.8 |

| Balanced class mix (equal crops per class) | Accuracy |
|---|---|
| Naive Bayes (RGB), forced | **61.1% ± 1.9** |
| Tall-and-narrow rule | 55.9% ± 1.6 |
| Chance | 33.3% |

**Charts:**
- [chart_threshold_sweep.png](chart_threshold_sweep.png): the threshold trades
  coverage for accuracy;
- [chart_learning_curve.png](chart_learning_curve.png): accuracy barely improves
  beyond about 10% of the training data;
- [chart_cross_validation.png](chart_cross_validation.png): per-fold results.

## Repository layout

| File | Purpose |
|---|---|
| `final-project.py` | **Core algorithm** (`GaussianNaiveBayes`), the 30-example dataset, the demo, and a plain-language explanation at the bottom |
| `visdrone_features.py` | Crops VisDrone boxes and computes RGB and HSV color features, width, height and aspect ratio. Writes the CSV |
| `visdrone_features.csv` | The 17,316 extracted crops (derived features only, no images) |
| `visdrone_nb.py` | Real-data evaluation, threshold sweep, learning curve and cross-validation, plus charts |
| `export_nb_model.py`, `nb_model.json` | Trained model saved as JSON for the drone integration |
| `tests/test_naive_bayes.py` | Correctness tests |
| `demo_output.txt`, `visdrone_results.txt`, `chart_*.png` | Saved outputs |
| `PROGRESS.md` | Checkpoint progress summary |

## Known limitations
- **Features overlap.** Aspect ratio comes from width and height, and the RGB
  shares add up to 1. That breaks the independence assumption and makes the model
  over-confident.
- **Small boxes are skipped.** Boxes under 20 px on a side are left out, which is
  81% of VisDrone people. Most people seen from a drone are tiny.
- **Box sizes are in VisDrone pixels.** A different camera resolution shifts the
  size features.

## Data credit
VisDrone2019-DET: P. Zhu et al., "Detection and Tracking Meet Drones Challenge,"
*IEEE TPAMI*, 2021. <https://github.com/VisDrone/VisDrone-Dataset>.

Only derived numeric features are included here, not the images.
