"""
CS471 Final Project - Baseline Naive Bayes classifier for drone detections
==========================================================================

Goal
----
A drone camera has already drawn a bounding box around "something" on the
ground. Our job is to decide what that something is:

    person, car, or other

...or to say "unable to identify" when the classifier is not confident enough.

Everything here is written from scratch using only the Python standard
library (no scikit-learn, no numpy). Run it with:

    python final-project.py

A plain-language explanation of how it all works is at the bottom of this
file (search for "HOW IT WORKS").
"""

import math
import random

# ---------------------------------------------------------------------------
# Settings you can change
# ---------------------------------------------------------------------------

# If the best class has a posterior probability below this, we refuse to
# guess and print "unable to identify" instead.
CONFIDENCE_THRESHOLD = 0.70

# How many examples of EACH class to hold out for testing (the rest are used
# for training). 3 per class -> 9 test examples, 21 training examples.
TEST_PER_CLASS = 3

# Fixed random seed so the train/test split is the same every time you run it
# (important so your results are repeatable when you present them).
RANDOM_SEED = 471

# The "tall and narrow" rule: if width / height is below this, call it a person.
RULE_ASPECT_CUTOFF = 0.75

CLASSES = ["person", "car", "other"]
UNKNOWN = "unable to identify"

# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
# Each example is one detected object (one bounding box) from a drone image.
#
#   red, green, blue : a 3-bin color histogram. Each value is the share of
#                      the box's total color that is red / green / blue,
#                      so the three numbers add up to about 1.0.
#   width, height    : size of the bounding box in pixels.
#   (aspect ratio = width / height is calculated from these below, so it can
#    never disagree with the width and height.)
#
# NOTE: these numbers are made up by hand to look like realistic drone
# detections. They are NOT measured from real images.
#
#   (red,  green, blue,  width, height, label,    description)
RAW_DATA = [
    # --- People: small boxes, taller than wide, slightly reddish (skin,
    #     clothing) ---
    (0.42, 0.30, 0.28, 18, 40, "person", "person walking"),
    (0.40, 0.31, 0.29, 16, 38, "person", "person standing"),
    (0.45, 0.28, 0.27, 20, 44, "person", "person in red jacket"),
    (0.38, 0.32, 0.30, 15, 35, "person", "person, far away"),
    (0.41, 0.29, 0.30, 22, 48, "person", "person, closer to camera"),
    (0.39, 0.30, 0.31, 17, 36, "person", "person in blue shirt"),
    (0.43, 0.30, 0.27, 19, 42, "person", "person carrying bag"),
    (0.40, 0.33, 0.27, 24, 30, "person", "person crouching"),
    (0.37, 0.31, 0.32, 14, 33, "person", "person in dark clothes"),
    (0.44, 0.29, 0.27, 21, 46, "person", "person jogging"),

    # --- Cars: big boxes, usually wider than tall, gray/neutral colors.
    #     Two cars are parked "vertically" in the image, so their boxes are
    #     tall and narrow - these will fool the simple rule. ---
    (0.33, 0.33, 0.34, 90, 45, "car", "gray sedan"),
    (0.32, 0.33, 0.35, 95, 48, "car", "silver SUV"),
    (0.34, 0.32, 0.34, 85, 42, "car", "white car"),
    (0.50, 0.25, 0.25, 88, 44, "car", "red car"),
    (0.31, 0.32, 0.37, 92, 46, "car", "blue car"),
    (0.33, 0.33, 0.34, 44, 88, "car", "car parked vertically"),
    (0.32, 0.34, 0.34, 46, 92, "car", "van parked vertically"),
    (0.34, 0.33, 0.33, 100, 50, "car", "pickup truck"),
    (0.33, 0.32, 0.35, 80, 40, "car", "small hatchback"),
    (0.35, 0.32, 0.33, 87, 47, "car", "dark sedan"),

    # --- Other: trees, bushes, animals, signs, shadows. Very mixed sizes;
    #     often green. ---
    (0.28, 0.48, 0.24, 60, 58, "other", "tree from above"),
    (0.27, 0.50, 0.23, 40, 38, "other", "bush"),
    (0.30, 0.45, 0.25, 75, 70, "other", "large tree"),
    (0.40, 0.33, 0.27, 30, 18, "other", "dog"),
    (0.33, 0.33, 0.34, 10, 50, "other", "sign post (tall and narrow!)"),
    (0.31, 0.32, 0.37, 55, 30, "other", "shadow"),
    (0.29, 0.47, 0.24, 35, 33, "other", "hedge"),
    (0.42, 0.34, 0.24, 50, 45, "other", "pile of dirt"),
    (0.34, 0.34, 0.32, 12, 55, "other", "light pole"),
    (0.30, 0.44, 0.26, 25, 26, "other", "small shrub"),
]

FEATURE_NAMES = ["red", "green", "blue", "width", "height", "aspect"]


def build_dataset(raw):
    """Turn each raw row into (feature list, label, description).

    The feature list is [red, green, blue, width, height, aspect_ratio].
    """
    dataset = []
    for red, green, blue, width, height, label, desc in raw:
        aspect = width / height  # < 1 means taller than wide
        features = [red, green, blue, width, height, aspect]
        dataset.append((features, label, desc))
    return dataset


def split_train_test(dataset, test_per_class, seed):
    """Hold out `test_per_class` random examples from each class for testing.

    Splitting per class ("stratified") makes sure every class shows up in
    both the training set and the test set, which matters a lot when we
    only have 10 examples of each.
    """
    rng = random.Random(seed)
    train, test = [], []
    for cls in CLASSES:
        examples = [ex for ex in dataset if ex[1] == cls]
        rng.shuffle(examples)
        test.extend(examples[:test_per_class])
        train.extend(examples[test_per_class:])
    return train, test


# ---------------------------------------------------------------------------
# The Naive Bayes classifier (Gaussian version), written from scratch
# ---------------------------------------------------------------------------

class GaussianNaiveBayes:
    """Naive Bayes where each feature, inside each class, is modeled as a
    bell curve (a Gaussian / normal distribution).

    Training  = for every class, remember
                  * how common the class is (the "prior"), and
                  * the average and spread (mean, variance) of every feature.
    Predicting = for a new example, ask each class "how likely is it that
                 YOU produced these feature values?", multiply by the prior,
                 and normalize so the answers add up to 1 (the "posterior").
    """

    def __init__(self, variance_floor=1e-3):
        # A tiny amount added to every variance. Without it, a feature that
        # happens to be identical for every training example of a class
        # would have variance 0, and dividing by 0 would crash the math.
        self.variance_floor = variance_floor
        self.priors = {}  # class -> P(class)
        self.means = {}   # class -> [mean of each feature]
        self.vars = {}    # class -> [variance of each feature]

    def fit(self, examples):
        """Learn priors, means and variances from labeled training examples."""
        total = len(examples)
        for cls in CLASSES:
            rows = [features for features, label, _ in examples if label == cls]
            n = len(rows)
            self.priors[cls] = n / total

            num_features = len(rows[0])
            means, variances = [], []
            for j in range(num_features):
                column = [row[j] for row in rows]
                mean = sum(column) / n
                var = sum((x - mean) ** 2 for x in column) / n
                means.append(mean)
                # Scale the floor by the size of the feature so it is "tiny"
                # for both color shares (~0.3) and pixel sizes (~50).
                variances.append(var + self.variance_floor * (abs(mean) + 1e-9))
            self.means[cls] = means
            self.vars[cls] = variances

    @staticmethod
    def _log_gaussian(x, mean, var):
        """Log of the bell-curve height at x.

        We work with logarithms because multiplying many small probabilities
        together can round down to 0.0 on a computer; adding their logs
        does not have that problem.
        """
        return -0.5 * math.log(2 * math.pi * var) - (x - mean) ** 2 / (2 * var)

    def posteriors(self, features):
        """Return {class: P(class | features)}, which adds up to 1."""
        # Step 1: log P(class) + sum over features of log P(feature | class)
        #         This is the "naive" part: we treat features as independent,
        #         so their probabilities simply multiply (logs add).
        log_scores = {}
        for cls in CLASSES:
            score = math.log(self.priors[cls])
            for x, mean, var in zip(features, self.means[cls], self.vars[cls]):
                score += self._log_gaussian(x, mean, var)
            log_scores[cls] = score

        # Step 2: turn the log scores back into probabilities that sum to 1.
        #         Subtracting the largest score first keeps exp() from
        #         overflowing/underflowing (the "log-sum-exp" trick).
        biggest = max(log_scores.values())
        unnormalized = {c: math.exp(s - biggest) for c, s in log_scores.items()}
        total = sum(unnormalized.values())
        return {c: v / total for c, v in unnormalized.items()}

    def predict(self, features, threshold):
        """Return (label, posteriors). The label is UNKNOWN if not confident."""
        post = self.posteriors(features)
        best = max(post, key=post.get)
        if post[best] < threshold:
            return UNKNOWN, post
        return best, post


# ---------------------------------------------------------------------------
# The simple comparison rule: "tall and narrow means person"
# ---------------------------------------------------------------------------

def tall_and_narrow_rule(features):
    """If the box is clearly taller than it is wide, say person.

    The rule only knows about people, so for everything else it has to pick
    something; it guesses "car", the most common non-person thing a drone
    looking at a road sees. (Its person/not-person accuracy is also reported,
    which is the fairest comparison since that is all the rule really does.)
    """
    aspect = features[FEATURE_NAMES.index("aspect")]
    return "person" if aspect < RULE_ASPECT_CUTOFF else "car"


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_confusion_matrix(true_labels, predicted_labels, columns, title):
    """Rows = what the object really was; columns = what we predicted."""
    print(f"\n{title}")
    print("  (rows = actual class, columns = predicted class)")
    col_width = max(len(c) for c in columns) + 2
    header = " " * 10 + "".join(c.rjust(col_width) for c in columns)
    print(header)
    for actual in CLASSES:
        counts = [
            sum(1 for t, p in zip(true_labels, predicted_labels)
                if t == actual and p == col)
            for col in columns
        ]
        print(actual.ljust(10) + "".join(str(n).rjust(col_width) for n in counts))


def percent(numerator, denominator):
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/a"


def person_binary(label):
    """Collapse a label to 'person' vs 'not person' (UNKNOWN stays UNKNOWN)."""
    if label == UNKNOWN:
        return UNKNOWN
    return "person" if label == "person" else "not person"


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    dataset = build_dataset(RAW_DATA)
    train, test = split_train_test(dataset, TEST_PER_CLASS, RANDOM_SEED)

    print("=" * 72)
    print("Naive Bayes drone object classifier (baseline)")
    print("=" * 72)
    print(f"Total examples: {len(dataset)}  |  training: {len(train)}  |  "
          f"held-out test: {len(test)}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f} "
          f"(below this -> '{UNKNOWN}')")

    # ---- Train ----
    model = GaussianNaiveBayes()
    model.fit(train)

    print("\nWhat the model learned (class prior, and mean of each feature):")
    print("  class    prior  " + "".join(f.rjust(8) for f in FEATURE_NAMES))
    for cls in CLASSES:
        means = "".join(f"{m:8.2f}" for m in model.means[cls])
        print(f"  {cls:<8} {model.priors[cls]:.2f}  {means}")

    # ---- Classify every held-out test example ----
    print("\n" + "-" * 72)
    print("Held-out test examples")
    print("-" * 72)

    true_labels, nb_labels, rule_labels = [], [], []
    for i, (features, actual, desc) in enumerate(test, start=1):
        label, post = model.predict(features, CONFIDENCE_THRESHOLD)
        rule = tall_and_narrow_rule(features)
        true_labels.append(actual)
        nb_labels.append(label)
        rule_labels.append(rule)

        red, green, blue, width, height, aspect = features
        print(f"\nTest #{i}: {desc}  (actual: {actual})")
        print(f"  features: color r/g/b = {red:.2f}/{green:.2f}/{blue:.2f}, "
              f"box = {width}x{height} px, aspect = {aspect:.2f}")
        print("  posteriors: " + "   ".join(
            f"P({c}) = {post[c]:.3f}" for c in CLASSES))
        best = max(post, key=post.get)
        if label == UNKNOWN:
            verdict = (f"{UNKNOWN.upper()} (best guess was {best} at "
                       f"{post[best]:.3f}, below {CONFIDENCE_THRESHOLD:.2f})")
        else:
            verdict = label + ("  [correct]" if label == actual else "  [WRONG]")
        print(f"  Naive Bayes says: {verdict}")
        print(f"  Tall-and-narrow rule says: {rule}"
              + ("  [correct]" if rule == actual else "  [WRONG]"))

    # ---- Accuracy ----
    n = len(test)
    nb_correct = sum(1 for t, p in zip(true_labels, nb_labels) if t == p)
    nb_answered = sum(1 for p in nb_labels if p != UNKNOWN)
    nb_correct_when_answered = sum(
        1 for t, p in zip(true_labels, nb_labels) if p != UNKNOWN and t == p)
    rule_correct = sum(1 for t, p in zip(true_labels, rule_labels) if t == p)

    print("\n" + "=" * 72)
    print("Results on the held-out test set")
    print("=" * 72)
    print("Naive Bayes:")
    print(f"  accuracy (an 'unable to identify' counts as wrong): "
          f"{nb_correct}/{n} = {percent(nb_correct, n)}")
    print(f"  answered (confident) on: {nb_answered}/{n} = "
          f"{percent(nb_answered, n)}")
    print(f"  accuracy when it did answer: {nb_correct_when_answered}/"
          f"{nb_answered} = {percent(nb_correct_when_answered, nb_answered)}")
    print("Tall-and-narrow rule (everything not tall/narrow is called 'car'):")
    print(f"  accuracy: {rule_correct}/{n} = {percent(rule_correct, n)}")

    # Fairer head-to-head: just "is it a person or not?"
    true_bin = [person_binary(t) for t in true_labels]
    nb_bin = [person_binary(p) for p in nb_labels]
    rule_bin = [person_binary(p) for p in rule_labels]
    nb_bin_correct = sum(1 for t, p in zip(true_bin, nb_bin) if t == p)
    rule_bin_correct = sum(1 for t, p in zip(true_bin, rule_bin) if t == p)
    print("Person vs. not-person only (the question the rule actually answers):")
    print(f"  Naive Bayes: {nb_bin_correct}/{n} = {percent(nb_bin_correct, n)}"
          f"   |   rule: {rule_bin_correct}/{n} = {percent(rule_bin_correct, n)}")

    print_confusion_matrix(true_labels, nb_labels, CLASSES + ["unable"],
                           "Naive Bayes confusion matrix:")
    print_confusion_matrix(true_labels, rule_labels, CLASSES,
                           "Tall-and-narrow rule confusion matrix:")

    # ---- Bonus: run the rule on ALL 30 examples to show where it breaks ----
    fooled = [(desc, label) for features, label, desc in dataset
              if (tall_and_narrow_rule(features) == "person") != (label == "person")]
    print("\nWhere the tall-and-narrow rule goes wrong on person vs. not-person"
          " (all 30 examples):")
    for desc, label in fooled:
        print(f"  - {desc} (really: {label})")
    print()


if __name__ == "__main__":
    main()


# ===========================================================================
# HOW IT WORKS (plain-language explanation for the presentation)
# ===========================================================================
#
# 1. THE PROBLEM
#    A drone's detector has already put a box around an object. We want to
#    label the box as person, car, or other, and we want the system to admit
#    when it isn't sure ("unable to identify") instead of guessing.
#
# 2. THE FEATURES (what the classifier "sees")
#    For each box we use six numbers:
#      - red / green / blue share of the color: people tend to be a bit
#        reddish (skin, clothes), cars gray/neutral, trees and bushes green.
#      - width and height in pixels: from the air, people are small, cars
#        are big, "other" things are all over the place.
#      - aspect ratio (width / height): people standing up are tall and
#        narrow; most cars in the image are wide.
#
# 3. BAYES' RULE
#    We want P(class | features): "given what I see, how likely is each
#    class?" Bayes' rule flips that around into things we CAN estimate:
#
#        P(class | features)  is proportional to  P(features | class) * P(class)
#
#      - P(class) is the PRIOR: how common the class is in the training data
#        (here each class is 7 of 21, so 1/3 each).
#      - P(features | class) is the LIKELIHOOD: if this really were a car,
#        how likely would these exact color and size values be?
#
# 4. THE "NAIVE" PART
#    Estimating the likelihood of all six features together would need far
#    more data than 30 examples. So Naive Bayes assumes the features are
#    independent once you know the class, which lets us just multiply:
#
#        P(features | class) = P(red | class) * P(green | class) * ... *
#                              P(aspect | class)
#
#    This assumption is clearly not true here (aspect ratio is computed from
#    width and height, and the three color shares add up to 1), so the model
#    tends to be OVER-confident: it counts the same evidence more than once.
#    Naive Bayes often still picks the right class even when its
#    probabilities are too extreme.
#
# 5. GAUSSIAN (BELL-CURVE) LIKELIHOODS
#    Our features are continuous numbers, not yes/no. So for each class and
#    each feature, training just computes the average (mean) and the spread
#    (variance) of the training examples. When a new example arrives we ask
#    "how high is the bell curve at this value?" A value near the class
#    average gets a high likelihood; a value far away gets a tiny one.
#
# 6. LOGS AND NORMALIZING
#    Multiplying six small numbers can round down to zero on a computer, so
#    the code adds logarithms instead of multiplying probabilities. At the
#    end it converts back and divides by the total so the three class
#    probabilities add up to exactly 1. Those are the POSTERIORS printed for
#    each test example.
#
# 7. "UNABLE TO IDENTIFY"
#    We pick the class with the highest posterior. But if even the best class
#    is below CONFIDENCE_THRESHOLD (0.70), we say "unable to identify". For a
#    drone this is the safer behavior: a wrong confident answer could be
#    worse than handing the box to a human. Raising the threshold means
#    fewer answers but more of them right; lowering it does the opposite.
#    Because of point 4, the posteriors are often near 0 or 1, so the
#    threshold only kicks in for really ambiguous objects.
#
# 8. EVALUATION
#    - We hold out 3 examples of each class (9 total) that the model never
#      sees while training, and test only on those. Testing on training data
#      would make the model look better than it really is.
#    - Accuracy is reported two ways: counting "unable to identify" as wrong,
#      and accuracy only on the examples it was confident about.
#    - The confusion matrix shows WHICH mistakes happen (e.g. are cars being
#      confused with "other"?), not just how many.
#
# 9. THE COMPARISON RULE
#    "Tall and narrow means person" looks only at the aspect ratio. It fails
#    on things that are tall and narrow but aren't people (sign posts, light
#    poles, cars parked pointing up the image) and on people who aren't
#    tall and narrow (someone crouching). It also never says "other" at
#    all. Naive Bayes combines size and color with shape. In our test set
#    it correctly called the tree and the dog "other", and on the sign post
#    (where the rule said "person") it said "unable to identify" instead of
#    making a wrong guess.
#
#    WHAT WENT WRONG: the red car. Every red car in our data ended up in the
#    test set, so during training "car" meant "gray". The red color looked
#    so unlike a car that Naive Bayes said "other" with probability 1.000.
#    The threshold could not catch it because the model was confidently
#    wrong. This is the over-confidence from point 4, plus a training set too
#    small to cover every kind of car.
#
# 10. LIMITATIONS / NEXT STEPS
#    - The data is small and hand-made; real results need real drone images.
#    - With only 9 test examples, each one is worth 11% accuracy, so the
#      numbers are noisy. Cross-validation (rotating which examples are held
#      out) would give a more reliable estimate.
#    - Pixel size depends on drone altitude; dividing by altitude (or using
#      ground-size in meters) would make the size features more reliable.
#    - Dropping redundant features (e.g. keep aspect, drop height) would
#      reduce the over-confidence described in point 4.
