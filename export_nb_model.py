"""
Export the trained Naive Bayes model to a small JSON file.

The Sentinel-Drone repo can't import this project's code, so this script
trains the SAME GaussianNaiveBayes class from final-project.py on all the
VisDrone crops (RGB features) and saves the numbers it learned:

    priors     P(class)
    means      average of each feature, per class
    variances  spread of each feature, per class (the small variance floor
               from final-project.py is already added in)

It also saves a few "golden" examples: feature vectors together with the
exact posteriors this code produces. The drone repo's unit tests check that
its classifier reproduces these numbers, which proves the two
implementations agree.

Run (from this folder; standard library only):
    python export_nb_model.py
"""

import csv
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "visdrone_features.csv")
OUT_PATH = os.path.join(HERE, "nb_model.json")

_spec = importlib.util.spec_from_file_location(
    "baseline", os.path.join(HERE, "final-project.py"))
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)

FEATURES = ["red", "green", "blue", "width", "height", "aspect"]
assert FEATURES == baseline.FEATURE_NAMES, "feature order must match the baseline"
MIN_SIDE = 20  # same cutoff visdrone_features.py used when building the CSV


def main():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    examples = [([float(r[c]) for c in FEATURES], r["label"], r["image"])
                for r in rows]

    model = baseline.GaussianNaiveBayes()
    model.fit(examples)
    threshold = baseline.CONFIDENCE_THRESHOLD

    # Golden examples: the first crop of each class, plus the first crop the
    # model refuses ("unable to identify"), so both paths are covered.
    golden, seen = [], set()
    for features, label, image in examples:
        predicted, post = model.predict(features, threshold)
        key = "unable" if predicted == baseline.UNKNOWN else label
        if key in seen:
            continue
        seen.add(key)
        golden.append({"features": features, "true_label": label,
                       "expected_label": predicted,
                       "expected_posteriors": post, "source_image": image})
        if len(seen) == 4:
            break

    out = {
        "format": "sentinel-gaussian-naive-bayes/1",
        "description": ("Gaussian Naive Bayes over 6 box features, trained "
                        "from scratch for CS471 on VisDrone2019-DET val crops."),
        "classes": baseline.CLASSES,
        "unknown_label": baseline.UNKNOWN,
        "feature_names": FEATURES,
        "feature_notes": {
            "red/green/blue": ("share of the crop's summed R, G and B values; "
                               "the three add up to 1. Computed in RGB order "
                               "(OpenCV frames are BGR and must be swapped)."),
            "width/height": "bounding-box size in pixels",
            "aspect": "width / height",
        },
        "confidence_threshold": threshold,
        "min_box_side_px": MIN_SIDE,
        "priors": model.priors,
        "means": model.means,
        "variances": model.vars,
        "training": {
            "dataset": "VisDrone2019-DET-val (548 images)",
            "crops": len(examples),
            "class_counts": {c: sum(1 for _, l, _ in examples if l == c)
                             for c in baseline.CLASSES},
            "class_mapping": {"person": ["pedestrian", "people"],
                              "car": ["car", "van"],
                              "other": "every other VisDrone category"},
            "image_sizes_px": "about 1360x765 to 2000x1500",
            "cv_accuracy_note": ("5-fold CV by image, natural mix: 66.5% when "
                                 "forced to answer; 76.5% on the 54.5% of crops "
                                 "it answers at threshold 0.70."),
            "box_size_caveat": (
                "width/height were learned in VisDrone pixels. A 640x480 "
                "drone camera sees the same object smaller in pixels, so "
                "size-based evidence is shifted on the drone. Not corrected."),
        },
        "golden_examples": golden,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"Trained on {len(examples)} crops; wrote {os.path.basename(OUT_PATH)} "
          f"({os.path.getsize(OUT_PATH)} bytes)")
    for g in golden:
        print(f"  golden: true={g['true_label']:<6} -> {g['expected_label']:<18} "
              + " ".join(f"{c}={p:.3f}" for c, p in g["expected_posteriors"].items()))


if __name__ == "__main__":
    main()
