"""
Steps 3 and 4 - Train and test the baseline Naive Bayes on real VisDrone crops.

Step 3: split the crops into training and test sets BY IMAGE. All the crops
        from one photo go to the same side, so the test set is made of
        photos the model has never seen.
Step 4: train the exact GaussianNaiveBayes class from final-project.py,
        then report accuracy, a confusion matrix that counts every test
        crop (including "unable to identify"), and the
        "tall and narrow means person" rule for comparison.

Everything is reported twice:
  * natural mix  - the classes as they really occur (lots of cars), and
  * balanced     - the same number of crops per class, so that a model
                   which just says "car" a lot can't look good by luck.

Three extra analyses (added later):
  1. threshold sweep   - accuracy-when-answering vs. share refused,
                         for thresholds 0.50 to 0.95
  2. learning curve    - train on 5%..100% of the training images,
                         10 random picks each, mean and spread
  3. cross-validation  - 5 folds, split by image
Each one is saved as a PNG chart in this folder if matplotlib is installed.

Run (from this folder; standard library only, matplotlib optional for charts):
    python visdrone_nb.py                  (everything)
    python visdrone_nb.py --only sweep cv  (just some parts: choose from
                                            results, sweep, curve, cv)
"""

import argparse
import csv
import importlib.util
import math
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "visdrone_features.csv")

# ---------------------------------------------------------------------------
# Load the baseline code from final-project.py. Its name has a dash in it, so
# a normal "import" won't work; this loads it straight from the file. Its
# main() does NOT run, because that only happens when the file is run
# directly.
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "baseline", os.path.join(HERE, "final-project.py"))
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)

GaussianNaiveBayes = baseline.GaussianNaiveBayes
CLASSES = baseline.CLASSES                   # ["person", "car", "other"]
UNKNOWN = baseline.UNKNOWN                   # "unable to identify"
THRESHOLD = baseline.CONFIDENCE_THRESHOLD    # 0.70
RULE_CUTOFF = baseline.RULE_ASPECT_CUTOFF    # 0.75

RANDOM_SEED = 471
TEST_FRACTION = 0.20      # 20% of the IMAGES go to the test set
SAMPLE_PER_CLASS = 3      # how many individual test crops to print per class

# Which CSV columns are used as features for each feature set.
FEATURE_SETS = {
    "rgb": ["red", "green", "blue", "width", "height", "aspect"],
    # Same box-shape features, but color comes from the 7-bin HSV histogram
    # instead of the 3 RGB shares.
    "hsv": ["hue_red", "hue_yellow", "hue_green", "hue_cyan", "hue_blue",
            "hue_magenta", "hue_gray", "width", "height", "aspect"],
}


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------

def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def split_by_image(rows, test_fraction, seed):
    """Step 3: put whole images (not single crops) into train or test."""
    images = sorted({r["image"] for r in rows})
    random.Random(seed).shuffle(images)
    n_test = round(len(images) * test_fraction)
    test_images = set(images[:n_test])
    train = [r for r in rows if r["image"] not in test_images]
    test = [r for r in rows if r["image"] in test_images]

    # Safety check: no photo may appear on both sides.
    overlap = {r["image"] for r in train} & {r["image"] for r in test}
    assert not overlap, f"images in both train and test: {overlap}"
    return train, test, len(images) - n_test, n_test


def balance(rows, seed):
    """Keep the same number of crops per class (the size of the rarest one),
    picked at random."""
    rng = random.Random(seed)
    by_class = {c: [r for r in rows if r["label"] == c] for c in CLASSES}
    n = min(len(v) for v in by_class.values())
    out = []
    for c in CLASSES:
        out.extend(rng.sample(by_class[c], n))
    return out


def to_examples(rows, columns):
    """Convert CSV rows into the (features, label, description) tuples that
    the baseline classifier expects."""
    return [([float(r[c]) for c in columns], r["label"],
             f"{r['visdrone_category']} in {r['image']}") for r in rows]


# ---------------------------------------------------------------------------
# The comparison rule (same cutoff as the baseline)
# ---------------------------------------------------------------------------

def tall_and_narrow_rule(row):
    """Person if width/height < 0.75, otherwise car (exactly like the
    baseline rule)."""
    return "person" if float(row["aspect"]) < RULE_CUTOFF else "car"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def confusion_matrix(true_labels, predicted, columns):
    """Rows = actual class, columns = predicted. Every crop lands in exactly
    one cell, so the row totals add up to the whole test set."""
    print("  (rows = actual class, columns = predicted)")
    w = 10
    # The column header says "unable" (short), but the label it counts is
    # the full "unable to identify" text.
    header = ["unable" if c == UNKNOWN else c for c in columns]
    print(" " * 10 + "".join(c.rjust(w) for c in header) + "total".rjust(w))
    grand = 0
    for actual in CLASSES:
        counts = [sum(1 for t, p in zip(true_labels, predicted)
                      if t == actual and p == col) for col in columns]
        grand += sum(counts)
        print(f"{actual:<10}" + "".join(str(n).rjust(w) for n in counts)
              + str(sum(counts)).rjust(w))
    assert grand == len(true_labels), "confusion matrix lost some examples"
    print(f"{'':<10}" + " " * (w * len(columns)) + str(grand).rjust(w))


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "  n/a"


def evaluate(name, train_rows, test_rows, columns, show_samples):
    """Train on train_rows, test on test_rows, print everything, and return
    a short summary for the final comparison table."""
    model = GaussianNaiveBayes()
    model.fit(to_examples(train_rows, columns))
    test = to_examples(test_rows, columns)

    true = [label for _, label, _ in test]
    preds, posts = [], []
    for features, _, _ in test:
        label, post = model.predict(features, THRESHOLD)
        preds.append(label)
        posts.append(post)
    rule = [tall_and_narrow_rule(r) for r in test_rows]

    n = len(test)
    correct = sum(t == p for t, p in zip(true, preds))
    answered = sum(p != UNKNOWN for p in preds)
    correct_answered = sum(t == p for t, p in zip(true, preds) if p != UNKNOWN)
    # What if NB had to answer every time (no "unable to identify")?
    forced = [max(p, key=p.get) for p in posts]
    forced_correct = sum(t == p for t, p in zip(true, forced))
    rule_correct = sum(t == p for t, p in zip(true, rule))
    majority = Counter(r["label"] for r in train_rows).most_common(1)[0][0]
    majority_correct = sum(t == majority for t in true)

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(f"training crops: {len(train_rows)} {dict(Counter(r['label'] for r in train_rows))}")
    print(f"test crops:     {n} {dict(Counter(true))}")
    print("learned priors: " + ", ".join(
        f"P({c}) = {model.priors[c]:.2f}" for c in CLASSES))

    if show_samples:
        print(f"\nA few individual test crops ({SAMPLE_PER_CLASS} per class, "
              "picked at random):")
        rng = random.Random(RANDOM_SEED)
        for cls in CLASSES:
            idx = [i for i, t in enumerate(true) if t == cls]
            for i in rng.sample(idx, min(SAMPLE_PER_CLASS, len(idx))):
                r = test_rows[i]
                p = posts[i]
                print(f"  {r['visdrone_category']:>10} {r['width']:>4}x{r['height']:<4}"
                      f" actual={true[i]:<6} "
                      + " ".join(f"P({c})={p[c]:.3f}" for c in CLASSES)
                      + f"  -> {preds[i]}"
                      + ("" if preds[i] == UNKNOWN else
                         ("  [ok]" if preds[i] == true[i] else "  [WRONG]")))

    print(f"\nNaive Bayes accuracy ('{UNKNOWN}' counts as wrong): "
          f"{correct}/{n} = {pct(correct, n)}")
    print(f"  answered (top posterior >= {THRESHOLD}): {answered}/{n} = "
          f"{pct(answered, n)}")
    print(f"  accuracy when it answered: {correct_answered}/{answered} = "
          f"{pct(correct_answered, answered)}")
    print(f"  accuracy if forced to answer every crop (no threshold): "
          f"{forced_correct}/{n} = {pct(forced_correct, n)}")
    print(f"Tall-and-narrow rule accuracy:{rule_correct}/{n} = "
          f"{pct(rule_correct, n)}")
    print(f"Reference - always guess the most common class ('{majority}'): "
          f"{majority_correct}/{n} = {pct(majority_correct, n)}")

    print("\nNaive Bayes confusion matrix:")
    confusion_matrix(true, preds, CLASSES + [UNKNOWN])
    print("\nTall-and-narrow rule confusion matrix:")
    confusion_matrix(true, rule, CLASSES)

    return {"name": name, "nb": correct / n, "answered": answered / n,
            "nb_when_answered": correct_answered / answered if answered else 0,
            "forced": forced_correct / n,
            "rule": rule_correct / n, "majority": majority_correct / n}


# ---------------------------------------------------------------------------
# Shared helpers for the three extra analyses
# ---------------------------------------------------------------------------

def fit_and_posteriors(train_rows, test_rows, columns):
    """Train the baseline model and return (true labels, posteriors) for
    every test crop. The analyses below reuse these posteriors for many
    thresholds instead of re-training each time."""
    model = GaussianNaiveBayes()
    model.fit(to_examples(train_rows, columns))
    posts = [model.posteriors([float(r[c]) for c in columns]) for r in test_rows]
    return [r["label"] for r in test_rows], posts


def score(true, posts, threshold):
    """Accuracy numbers at one threshold. A crop is 'answered' when its top
    posterior is >= threshold, exactly like baseline.predict()."""
    n = len(true)
    best = [max(p, key=p.get) for p in posts]
    answered = [p[b] >= threshold for p, b in zip(posts, best)]
    n_ans = sum(answered)
    right_ans = sum(a and b == t for a, b, t in zip(answered, best, true))
    return {
        "nb": right_ans / n,                         # 'unable' counted wrong
        "answered": n_ans / n,
        "refused": 1 - n_ans / n,
        "nb_when_answered": right_ans / n_ans if n_ans else float("nan"),
        "forced": sum(b == t for b, t in zip(best, true)) / n,
    }


def rule_accuracy(test_rows):
    return sum(tall_and_narrow_rule(r) == r["label"] for r in test_rows) / len(test_rows)


def mean_sd(values):
    """Mean and sample standard deviation (the "spread")."""
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def heading(text):
    print("\n" + "#" * 72)
    print("# " + text)
    print("#" * 72)


# ---------------------------------------------------------------------------
# Analysis 1: threshold sweep
# ---------------------------------------------------------------------------
SWEEP_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95


def threshold_sweep(train, test):
    """Train once per feature set, then try every threshold on the SAME
    posteriors. Raising the threshold can only make the model refuse more."""
    heading("Analysis 1: threshold sweep (natural class mix, fixed 80/20 split)")
    results = {}
    for fs, columns in FEATURE_SETS.items():
        true, posts = fit_and_posteriors(train, test, columns)
        results[fs] = [score(true, posts, t) for t in SWEEP_THRESHOLDS]
        print(f"\n[{fs.upper()}]  threshold   accuracy when it answers   refuses")
        for t, s in zip(SWEEP_THRESHOLDS, results[fs]):
            print(f"         {t:9.2f}   {100*s['nb_when_answered']:22.1f}%"
                  f"   {100*s['refused']:6.1f}%")
    print(f"\n(For reference: the tall-and-narrow rule never refuses and scores "
          f"{100*rule_accuracy(test):.1f}% on this test set.)")
    return results


# ---------------------------------------------------------------------------
# Analysis 2: learning curve
# ---------------------------------------------------------------------------
CURVE_FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
CURVE_REPEATS = 10


def learning_curve(train, test):
    """Train on a random part of the TRAINING images (never touching the test
    images) and see how accuracy grows with more data. Each size is repeated
    with 10 different random picks to show how much luck matters."""
    heading("Analysis 2: learning curve (natural class mix, same test set)")
    train_images = sorted({r["image"] for r in train})
    by_image = {}
    for r in train:
        by_image.setdefault(r["image"], []).append(r)

    results = {fs: [] for fs in FEATURE_SETS}
    for frac in CURVE_FRACTIONS:
        n_img = max(1, round(frac * len(train_images)))
        subsets = []
        for rep in range(CURVE_REPEATS):
            rng = random.Random(RANDOM_SEED * 1000 + rep)
            # Re-draw if a tiny sample happens to miss a class entirely
            # (the model can't learn a class it has never seen).
            for _ in range(100):
                picked = rng.sample(train_images, n_img)
                rows = [r for img in picked for r in by_image[img]]
                if {r["label"] for r in rows} >= set(CLASSES):
                    break
            else:
                raise RuntimeError(f"could not draw all 3 classes at {frac:.0%}")
            subsets.append(rows)

        for fs, columns in FEATURE_SETS.items():
            forced, answered_acc = [], []
            for rows in subsets:
                true, posts = fit_and_posteriors(rows, test, columns)
                s = score(true, posts, THRESHOLD)
                forced.append(s["forced"])
                answered_acc.append(s["nb_when_answered"])
            fm, fsd = mean_sd(forced)
            am, asd = mean_sd(answered_acc)
            n_crops = sum(len(r) for r in subsets) / len(subsets)
            results[fs].append({"frac": frac, "images": n_img,
                                "crops": n_crops, "mean": fm, "sd": fsd,
                                "min": min(forced), "max": max(forced)})
            print(f"[{fs.upper()}] {frac:4.0%} of training images "
                  f"({n_img:3d} images, ~{n_crops:5.0f} crops): "
                  f"forced accuracy {100*fm:5.1f}% +/- {100*fsd:4.1f} "
                  f"(range {100*min(forced):.1f}-{100*max(forced):.1f}); "
                  f"accuracy when answering at {THRESHOLD}: {100*am:5.1f}% +/- {100*asd:4.1f}")
    print("(+/- is the standard deviation over the 10 random picks. At 100% every "
          "pick is the same\n full training set, so the spread there is 0.)")
    return results


# ---------------------------------------------------------------------------
# Analysis 3: 5-fold cross-validation, split by image
# ---------------------------------------------------------------------------
N_FOLDS = 5


def cross_validation(rows):
    """Cut the IMAGES into 5 groups. Each group takes one turn as the test
    set while the other 4 train the model, so every image is tested exactly
    once and the result doesn't hinge on one lucky (or unlucky) split."""
    heading(f"Analysis 3: {N_FOLDS}-fold cross-validation, split by image")
    images = sorted({r["image"] for r in rows})
    random.Random(RANDOM_SEED).shuffle(images)
    folds = [set(images[i::N_FOLDS]) for i in range(N_FOLDS)]

    methods = [f"NB {fs.upper()}" for fs in FEATURE_SETS] + ["rule", "base"]
    results = {mix: {m: [] for m in methods} for mix in ("natural", "balanced")}
    extra = {mix: {fs: [] for fs in FEATURE_SETS} for mix in ("natural", "balanced")}

    for k, test_images in enumerate(folds, start=1):
        train = [r for r in rows if r["image"] not in test_images]
        test = [r for r in rows if r["image"] in test_images]
        assert not ({r["image"] for r in train} & {r["image"] for r in test})
        for mix in ("natural", "balanced"):
            tr, te = ((train, test) if mix == "natural" else
                      (balance(train, RANDOM_SEED + k), balance(test, RANDOM_SEED + k)))
            for fs, columns in FEATURE_SETS.items():
                true, posts = fit_and_posteriors(tr, te, columns)
                s = score(true, posts, THRESHOLD)
                results[mix][f"NB {fs.upper()}"].append(s["forced"])
                extra[mix][fs].append(s)
            results[mix]["rule"].append(rule_accuracy(te))
            majority = Counter(r["label"] for r in tr).most_common(1)[0][0]
            results[mix]["base"].append(
                sum(r["label"] == majority for r in te) / len(te))
        print(f"fold {k}: {len(folds[k-1]):3d} test images, {len(test):4d} test crops"
              + "".join(f" | {m} {100*results['natural'][m][-1]:5.1f}%"
                        for m in methods) + "  (natural mix, NB forced)")

    for mix in ("natural", "balanced"):
        print(f"\n{mix.capitalize()} class mix - mean +/- spread over {N_FOLDS} folds:")
        for fs in FEATURE_SETS:
            parts = []
            for key, label in (("nb", "NB ('unable' wrong)"),
                               ("answered", "answered"),
                               ("nb_when_answered", "NB when answered"),
                               ("forced", "NB forced")):
                m, sd = mean_sd([s[key] for s in extra[mix][fs]])
                parts.append(f"{label} {100*m:.1f}% +/- {100*sd:.1f}")
            print(f"  NB {fs.upper()}: " + " | ".join(parts))
        for m in ("rule", "base"):
            mm, sd = mean_sd(results[mix][m])
            name = ("tall-and-narrow rule" if m == "rule" else
                    "always guess the most common class")
            print(f"  {name}: {100*mm:.1f}% +/- {100*sd:.1f}")
    return results


# ---------------------------------------------------------------------------
# Charts (only if matplotlib is installed in the venv)
# ---------------------------------------------------------------------------
# Colors: blue = RGB features, orange = HSV features, gray = reference lines.
COLOR = {"rgb": "#2a78d6", "hsv": "#eb6834"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False,
    })


def _pct_axis(ax, lo=0, hi=100):
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")


def plot_sweep(plt, sweep, rule_acc, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    for fs, res in sweep.items():
        kw = dict(color=COLOR[fs], linewidth=2, marker="o", markersize=5,
                  label=f"NB {fs.upper()}")
        left.plot(SWEEP_THRESHOLDS, [100 * s["nb_when_answered"] for s in res], **kw)
        right.plot(SWEEP_THRESHOLDS, [100 * s["refused"] for s in res], **kw)
        left.annotate(f"NB {fs.upper()}", (SWEEP_THRESHOLDS[-1],
                      100 * res[-1]["nb_when_answered"]), xytext=(6, 0),
                      textcoords="offset points", va="center", color=INK_2)
    left.axhline(100 * rule_acc, color=INK_2, linewidth=1.2, linestyle="--")
    # Label sits under the dashed line on the right, where no curve runs.
    left.annotate(f"rule, never refuses ({100 * rule_acc:.1f}%)",
                  (0.95, 100 * rule_acc), xytext=(0, -13), ha="right",
                  textcoords="offset points", color=INK_2, fontsize=9)
    left.set_title("Accuracy on the crops it answers")
    right.set_title("Share of crops it refuses")
    for ax in (left, right):
        ax.set_xlabel("confidence threshold")
        ax.axvline(THRESHOLD, color=GRID, linewidth=6, zorder=0)
        _pct_axis(ax)
    left.set_xlim(0.48, 1.03)
    right.legend(loc="upper left")
    fig.suptitle("Threshold sweep: a stricter threshold is more accurate but "
                 "refuses more (shaded = 0.70 used so far)", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curve(plt, curve, rule_acc, base_acc, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for fs, res in curve.items():
        x = [100 * r["frac"] for r in res]
        m = [100 * r["mean"] for r in res]
        sd = [100 * r["sd"] for r in res]
        ax.fill_between(x, [a - b for a, b in zip(m, sd)],
                        [a + b for a, b in zip(m, sd)],
                        color=COLOR[fs], alpha=0.15, linewidth=0)
        ax.plot(x, m, color=COLOR[fs], linewidth=2, marker="o", markersize=5,
                label=f"NB {fs.upper()} (mean ± 1 SD of 10 picks)")
        ax.annotate(f"NB {fs.upper()}", (x[-1], m[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK_2)
    for y, text in ((rule_acc, "tall-and-narrow rule"),
                    (base_acc, "always guess 'car'")):
        # Stop the line at 100% so it doesn't run through the end labels.
        ax.hlines(100 * y, 4, 100, color=INK_2, linewidth=1.2, linestyle="--")
        ax.annotate(text, (5, 100 * y), xytext=(0, 4),
                    textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([100 * f for f in CURVE_FRACTIONS])
    ax.set_xticklabels([f"{f:.0%}" for f in CURVE_FRACTIONS])
    ax.minorticks_off()
    ax.set_xlim(4, 140)
    ax.set_xlabel("share of training images used (log scale)")
    ax.set_ylabel("accuracy, forced to answer")
    _pct_axis(ax, 40, 80)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Learning curve: more training images barely help")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cv(plt, cv, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, mix in zip(axes, ("natural", "balanced")):
        methods = list(cv[mix])
        for i, m in enumerate(methods):
            vals = [100 * v for v in cv[mix][m]]
            color = COLOR.get(m.split()[-1].lower(), INK_2)
            ax.scatter([i] * len(vals), vals, s=36, color=color, alpha=0.8,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
            mean = sum(vals) / len(vals)
            ax.hlines(mean, i - 0.25, i + 0.25, color=INK, linewidth=2, zorder=4)
            ax.annotate(f"{mean:.1f}%", (i + 0.28, mean), va="center",
                        fontsize=9, color=INK)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([{"rule": "tall-narrow\nrule",
                             "base": "always most\ncommon class"}.get(m, m + "\n(forced)")
                            for m in methods])
        ax.set_xlim(-0.5, len(methods) - 0.2)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{mix.capitalize()} class mix")
        _pct_axis(ax, 20, 90)
    axes[0].set_ylabel("test accuracy")
    fig.suptitle(f"{N_FOLDS}-fold cross-validation by image: each dot is one "
                 "fold, bar = mean", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_charts(sweep, curve, cv, test):
    try:
        import matplotlib
        matplotlib.use("Agg")  # draw straight to files, no window
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib is not installed, so no PNG charts were made. "
              "Install it with:  python -m pip install matplotlib)")
        return
    _style(plt)
    rule_acc = rule_accuracy(test)
    base_acc = sum(r["label"] == "car" for r in test) / len(test)
    paths = {name: os.path.join(HERE, name) for name in (
        "chart_threshold_sweep.png", "chart_learning_curve.png",
        "chart_cross_validation.png")}
    if sweep:
        plot_sweep(plt, sweep, rule_acc, paths["chart_threshold_sweep.png"])
    if curve:
        plot_curve(plt, curve, rule_acc, base_acc, paths["chart_learning_curve.png"])
    if cv:
        plot_cv(plt, cv, paths["chart_cross_validation.png"])
    print("\nSaved charts:")
    for p in paths.values():
        if os.path.exists(p):
            print("  " + os.path.basename(p))


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--only", choices=["results", "sweep", "curve", "cv"], nargs="*",
        help="run only these parts (default: everything)")
    parser.add_argument("--no-charts", action="store_true",
                        help="skip saving PNG charts")
    args = parser.parse_args()
    parts = set(args.only or ["results", "sweep", "curve", "cv"])

    rows = load_rows(CSV_PATH)
    train, test, n_train_img, n_test_img = split_by_image(
        rows, TEST_FRACTION, RANDOM_SEED)
    print(f"Loaded {len(rows)} crops from {os.path.basename(CSV_PATH)}")
    print(f"Split by image: {n_train_img} training images ({len(train)} crops), "
          f"{n_test_img} test images ({len(test)} crops). "
          "No image appears in both.")

    if "results" in parts:
        main_results(train, test)
    sweep = threshold_sweep(train, test) if "sweep" in parts else None
    curve = learning_curve(train, test) if "curve" in parts else None
    cv = cross_validation(rows) if "cv" in parts else None
    if not args.no_charts and (sweep or curve or cv):
        make_charts(sweep, curve, cv, test)


def main_results(train, test):
    """The original Step 4 report: one split, natural and balanced mixes."""
    summaries = []
    for fs_name, columns in FEATURE_SETS.items():
        summaries.append(evaluate(
            f"[{fs_name.upper()}] natural class mix", train, test, columns, True))
        summaries.append(evaluate(
            f"[{fs_name.upper()}] balanced (equal crops per class)",
            balance(train, RANDOM_SEED), balance(test, RANDOM_SEED),
            columns, True))

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"{'setting':<42}{'NB':>7}{'answered':>10}{'NB|ans':>8}"
          f"{'forced':>8}{'rule':>7}{'base':>7}")
    for s in summaries:
        print(f"{s['name']:<42}{100*s['nb']:6.1f}%{100*s['answered']:9.1f}%"
              f"{100*s['nb_when_answered']:7.1f}%{100*s['forced']:7.1f}%"
              f"{100*s['rule']:6.1f}%{100*s['majority']:6.1f}%")
    print("NB     = accuracy, 'unable to identify' counted as wrong\n"
          "NB|ans = accuracy on only the crops NB answered\n"
          "forced = NB accuracy if it must always answer (no threshold)\n"
          "base   = always guess the most common training class "
          "(for balanced data this is just chance, 33.3%)")


if __name__ == "__main__":
    main()
