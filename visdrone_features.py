"""
Step 2 - Turn VisDrone annotations into a CSV of features.

For every labeled box in every VisDrone image, this script:
  1. reads the box from the image's annotation .txt file,
  2. maps VisDrone's 11 categories down to our 3 classes
     (person / car / other),
  3. crops the box out of the image,
  4. computes the same features as the hand-made baseline in
     final-project.py (red/green/blue color share, box width, box height,
     aspect ratio = width / height), plus an optional HSV hue histogram
     (7 bins) so the two kinds of color feature can be compared,
  5. writes one row per box to visdrone_features.csv in this folder.

Run (from this folder; needs Pillow):
    python visdrone_features.py --data-dir PATH/TO/VisDrone2019-DET-val

PATH/TO/VisDrone2019-DET-val is the unzipped folder that contains
annotations/ and images/. You can set the VISDRONE_DIR environment variable
instead of passing --data-dir. See README.md for the download link.
"""

import argparse
import csv
import os
import sys
from collections import Counter

from PIL import Image, ImageChops, ImageStat

# Where the downloaded VisDrone data lives (--data-dir or VISDRONE_DIR) and
# where the CSV goes (this project folder).
DEFAULT_DATA_DIR = os.environ.get(
    "VISDRONE_DIR", os.path.join("data", "VisDrone2019-DET-val"))
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "visdrone_features.csv")

# Boxes smaller than this on EITHER side are skipped: a 6x14-pixel person
# has too few pixels for its color to mean much.
MIN_SIDE = 20

# VisDrone's category numbers (from the dataset's documentation).
VISDRONE_NAMES = {
    0: "ignored region", 1: "pedestrian", 2: "people", 3: "bicycle",
    4: "car", 5: "van", 6: "truck", 7: "tricycle", 8: "awning-tricycle",
    9: "bus", 10: "motor", 11: "others",
}

# Our 3 classes. Anything not listed as person or car becomes "other".
TO_OUR_CLASS = {1: "person", 2: "person", 4: "car", 5: "car"}

CSV_COLUMNS = [
    "image", "visdrone_category", "label",
    "left", "top", "width", "height", "aspect",
    "red", "green", "blue",
    "hue_red", "hue_yellow", "hue_green", "hue_cyan", "hue_blue",
    "hue_magenta", "hue_gray",
    "truncation", "occlusion",
]

# --- HSV histogram settings -------------------------------------------------
# HSV describes a color as Hue (which color: red, green, ...), Saturation (how
# vivid vs. grayish) and Value (how bright). Pillow stores each as 0-255.
# A pixel only gets a hue bin if it is vivid enough and bright enough;
# otherwise its "hue" is meaningless noise and it goes in the gray bin.
MIN_SATURATION = 51   # 20% of 255
MIN_VALUE = 38        # 15% of 255
HUE_BINS = ["hue_red", "hue_yellow", "hue_green", "hue_cyan", "hue_blue",
            "hue_magenta"]


def _hue_bin(h):
    """Which of the 6 hue bins (60 degrees each) a Pillow hue 0-255 falls in.
    Red is centered on 0 degrees, so it wraps around (330-360 and 0-30)."""
    degrees = h * 360 / 256
    return int(((degrees + 30) % 360) // 60)


HUE_BIN_OF = [_hue_bin(h) for h in range(256)]  # precomputed lookup table


def rgb_shares(crop):
    """Share of the crop's total color that is red, green and blue.

    Same idea as the baseline: add up every pixel's red value, green value
    and blue value, then divide each total by the grand total so the three
    numbers add up to 1.
    """
    r_sum, g_sum, b_sum = ImageStat.Stat(crop).sum[:3]
    total = r_sum + g_sum + b_sum
    if total == 0:  # a completely black crop: call it neutral
        return 1 / 3, 1 / 3, 1 / 3
    return r_sum / total, g_sum / total, b_sum / total


def hsv_histogram(crop):
    """Fraction of the crop's pixels in each hue bin, plus a gray bin.

    The 7 fractions add up to 1. Unlike the RGB shares (which mostly
    measure overall tint), this counts pixels: "30% of this box is green".
    """
    h, s, v = crop.convert("HSV").split()
    # Mask is white (255) where the pixel is vivid AND bright enough.
    vivid = s.point(lambda x: 255 if x >= MIN_SATURATION else 0)
    bright = v.point(lambda x: 255 if x >= MIN_VALUE else 0)
    mask = ImageChops.multiply(vivid, bright)

    hue_counts = h.histogram(mask=mask)   # 256 counts, colorful pixels only
    bins = [0] * len(HUE_BINS)
    for hue_value, count in enumerate(hue_counts):
        bins[HUE_BIN_OF[hue_value]] += count

    total_pixels = crop.width * crop.height
    gray = total_pixels - sum(bins)
    return [b / total_pixels for b in bins] + [gray / total_pixels]


def read_annotations(path):
    """Yield (left, top, width, height, category, truncation, occlusion)."""
    with open(path) as f:
        for line in f:
            parts = line.strip().rstrip(",").split(",")
            if len(parts) < 8:
                continue
            left, top, width, height, _score, cat, trunc, occ = map(int, parts[:8])
            yield left, top, width, height, cat, trunc, occ


def main():
    parser = argparse.ArgumentParser(
        description="Crop VisDrone boxes and write visdrone_features.csv")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="unzipped VisDrone2019-DET-val folder "
                             "(contains annotations/ and images/)")
    args = parser.parse_args()
    ann_dir = os.path.join(args.data_dir, "annotations")
    img_dir = os.path.join(args.data_dir, "images")
    if not (os.path.isdir(ann_dir) and os.path.isdir(img_dir)):
        sys.exit(f"VisDrone data not found in {args.data_dir!r}: expected "
                 "annotations/ and images/ inside it (see README.md).")
    ann_files = sorted(f for f in os.listdir(ann_dir) if f.endswith(".txt"))

    kept = Counter()          # boxes written to the CSV, per class
    skipped_small = Counter()  # boxes skipped for being < MIN_SIDE, per class
    skipped_ignored = 0        # "ignored region" boxes (not real objects)
    rows = []

    for n, ann_name in enumerate(ann_files, start=1):
        image_name = ann_name[:-4] + ".jpg"
        with Image.open(os.path.join(img_dir, image_name)) as img:
            img = img.convert("RGB")
            img_w, img_h = img.size

            for left, top, width, height, cat, trunc, occ in read_annotations(
                    os.path.join(ann_dir, ann_name)):
                if cat == 0:
                    # Category 0 marks crowded areas the labelers chose not
                    # to label. It isn't an object, so we leave it out.
                    skipped_ignored += 1
                    continue
                label = TO_OUR_CLASS.get(cat, "other")
                if width < MIN_SIDE or height < MIN_SIDE:
                    skipped_small[label] += 1
                    continue

                # Clip the box to the picture so we never crop past the edge.
                x1, y1 = max(0, left), max(0, top)
                x2, y2 = min(img_w, left + width), min(img_h, top + height)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    skipped_small[label] += 1
                    continue
                crop = img.crop((x1, y1, x2, y2))

                red, green, blue = rgb_shares(crop)
                hsv = hsv_histogram(crop)
                rows.append({
                    **{name: round(val, 4) for name, val in
                       zip(HUE_BINS + ["hue_gray"], hsv)},
                    "image": image_name,
                    "visdrone_category": VISDRONE_NAMES.get(cat, str(cat)),
                    "label": label,
                    "left": left, "top": top,
                    "width": width, "height": height,
                    "aspect": round(width / height, 4),
                    "red": round(red, 4), "green": round(green, 4),
                    "blue": round(blue, 4),
                    "truncation": trunc, "occlusion": occ,
                })
                kept[label] += 1

        if n % 100 == 0:
            print(f"  processed {n}/{len(ann_files)} images...", file=sys.stderr)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    total_small = sum(skipped_small.values())
    print(f"Images read: {len(ann_files)}")
    print(f"Boxes kept (written to CSV): {sum(kept.values())}")
    for cls in ("person", "car", "other"):
        print(f"  {cls:<7} {kept[cls]:>6}")
    print(f"Boxes skipped for being smaller than {MIN_SIDE}px on a side: "
          f"{total_small}")
    for cls in ("person", "car", "other"):
        print(f"  {cls:<7} {skipped_small[cls]:>6}")
    print(f"'Ignored region' boxes left out (not objects): {skipped_ignored}")
    print(f"Wrote {os.path.basename(OUTPUT_CSV)}")


if __name__ == "__main__":
    main()
