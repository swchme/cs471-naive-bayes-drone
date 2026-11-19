"""
Correctness checks for the from-scratch Gaussian Naive Bayes classifier.

Run from the project folder (standard library only):
    python -m unittest discover -s tests -v

Each test's docstring says WHAT is expected and WHY, so the output can be read
as evidence that the algorithm behaves correctly.
"""

import csv
import importlib.util
import math
import os
import random
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)


def _load(name, filename):
    """Import a project file by path (final-project.py has a dash in its name)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(PROJECT, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = _load("final_project", "final-project.py")
vnb = _load("visdrone_nb", "visdrone_nb.py")

try:
    import PIL  # noqa: F401  (only needed for the feature-extraction test)
    vfeat = _load("visdrone_features", "visdrone_features.py")
except ImportError:
    vfeat = None


def one_feature_data():
    """Tiny 1-feature training set where every class mean and variance is easy
    to work out by hand:
        person: 1, 3   -> mean 2,  variance 1
        car:    5, 7   -> mean 6,  variance 1
        other:  9, 11  -> mean 10, variance 1
    Each class has 2 of 6 examples, so every prior is 1/3."""
    rows = [(1, "person"), (3, "person"), (5, "car"), (7, "car"),
            (9, "other"), (11, "other")]
    return [([float(x)], label, f"x={x}") for x, label in rows]


def hand_model():
    model = nb.GaussianNaiveBayes(variance_floor=0.0)  # no smoothing: exact math
    model.fit(one_feature_data())
    return model


class TestTrainingMath(unittest.TestCase):

    def test_training_learns_priors_means_variances(self):
        """Expected: priors 1/3 each, means 2 / 6 / 10, variances 1 / 1 / 1
        (the hand-calculated values in one_feature_data's docstring)."""
        model = hand_model()
        for cls, mean in (("person", 2.0), ("car", 6.0), ("other", 10.0)):
            self.assertAlmostEqual(model.priors[cls], 1 / 3)
            self.assertAlmostEqual(model.means[cls][0], mean)
            self.assertAlmostEqual(model.vars[cls][0], 1.0)

    def test_posterior_matches_hand_calculation(self):
        """Bayes' rule by hand at x = 2 (exactly the person mean).
        With equal priors and variance 1, the likelihood of each class is
        proportional to exp(-(x - mean)^2 / 2):
            person: exp(0) = 1,  car: exp(-8),  other: exp(-32)
        so P(person | x=2) = 1 / (1 + e^-8 + e^-32) = 0.99966...
        Expected: the code's posteriors equal these numbers to 9 decimals."""
        post = hand_model().posteriors([2.0])
        denom = 1 + math.exp(-8) + math.exp(-32)
        self.assertAlmostEqual(post["person"], 1 / denom, places=9)
        self.assertAlmostEqual(post["car"], math.exp(-8) / denom, places=9)
        self.assertAlmostEqual(post["other"], math.exp(-32) / denom, places=9)

    def test_priors_shift_the_answer(self):
        """At x = 4 (halfway between person and car) the likelihoods tie, so
        only the PRIOR can break the tie. Adding another copy of the person
        examples (1 and 3) keeps person's mean 2 and variance 1, but makes
        person twice as common as car: priors 4/8 vs 2/8.
        Expected: P(person) / P(car) = 2 exactly at x = 4. This checks that
        priors are actually multiplied in."""
        data = one_feature_data() + [([1.0], "person", "copy"),
                                     ([3.0], "person", "copy")]
        model = nb.GaussianNaiveBayes(variance_floor=0.0)
        model.fit(data)
        self.assertAlmostEqual(model.means["person"][0], 2.0)
        self.assertAlmostEqual(model.vars["person"][0], 1.0)
        post = model.posteriors([4.0])
        self.assertAlmostEqual(post["person"] / post["car"], 2.0, places=9)


class TestPosteriorsAndThreshold(unittest.TestCase):

    def test_posteriors_always_sum_to_one(self):
        """Expected: for any input, P(person)+P(car)+P(other) = 1, because
        posteriors are normalized. Also checks extreme inputs don't crash
        (that is what the log-sum-exp trick is for)."""
        model = hand_model()
        for x in (-1e6, -3.0, 0.0, 4.0, 6.5, 50.0, 1e6):
            post = model.posteriors([x])
            self.assertAlmostEqual(sum(post.values()), 1.0, places=12)
            self.assertTrue(all(0.0 <= p <= 1.0 for p in post.values()))

    def test_ambiguous_input_is_unable_to_identify(self):
        """At x = 4, person and car are equally likely (about 0.5 each), so
        the top posterior is below the 0.70 threshold.
        Expected: 'unable to identify'."""
        label, post = hand_model().predict([4.0], threshold=0.70)
        self.assertEqual(label, nb.UNKNOWN)
        self.assertAlmostEqual(post["person"], 0.5, places=6)
        self.assertAlmostEqual(post["car"], 0.5, places=6)

    def test_clear_input_gets_the_right_class(self):
        """x = 10 is exactly the 'other' mean, far from the rest.
        Expected: 'other', with a posterior above the threshold."""
        label, post = hand_model().predict([10.0], threshold=0.70)
        self.assertEqual(label, "other")
        self.assertGreater(post["other"], 0.70)


class TestBaselineDemo(unittest.TestCase):
    """Checks on the 30 hand-made drone examples used by final-project.py."""

    def setUp(self):
        dataset = nb.build_dataset(nb.RAW_DATA)
        self.train, self.test = nb.split_train_test(
            dataset, nb.TEST_PER_CLASS, nb.RANDOM_SEED)
        self.model = nb.GaussianNaiveBayes()
        self.model.fit(self.train)

    def test_split_is_stratified_and_disjoint(self):
        """Expected: 21 training and 9 test examples, 3 test examples per
        class, and no example on both sides."""
        self.assertEqual((len(self.train), len(self.test)), (21, 9))
        for cls in nb.CLASSES:
            self.assertEqual(sum(1 for _, l, _ in self.test if l == cls), 3)
        train_ids = {d for _, _, d in self.train}
        self.assertFalse(train_ids & {d for _, _, d in self.test})

    def test_learned_means_equal_training_averages(self):
        """Expected: each learned mean is exactly the average of that class's
        training rows (computed independently here)."""
        for cls in nb.CLASSES:
            rows = [f for f, l, _ in self.train if l == cls]
            for j in range(len(nb.FEATURE_NAMES)):
                self.assertAlmostEqual(self.model.means[cls][j],
                                       sum(r[j] for r in rows) / len(rows))

    def test_documented_results(self):
        """Expected (the numbers in README.md): Naive Bayes gets 7/9 right
        (one refusal, one mistake: the red car), and the
        'tall and narrow means person' rule gets 6/9."""
        nb_right = rule_right = 0
        labels = {}
        for features, actual, desc in self.test:
            label, _ = self.model.predict(features, nb.CONFIDENCE_THRESHOLD)
            labels[desc] = label
            nb_right += label == actual
            rule_right += nb.tall_and_narrow_rule(features) == actual
        self.assertEqual(nb_right, 7)
        self.assertEqual(rule_right, 6)
        self.assertEqual(labels["red car"], "other")
        self.assertEqual(labels["sign post (tall and narrow!)"], nb.UNKNOWN)


class TestRealDataPipeline(unittest.TestCase):

    def test_split_by_image_never_leaks(self):
        """Crops from the same photo must never be in both train and test.
        Expected: no shared image names, and every crop is used once."""
        rng = random.Random(0)
        rows = [{"image": f"img{rng.randrange(40)}.jpg", "label": "car"}
                for _ in range(500)]
        train, test, _, _ = vnb.split_by_image(rows, 0.2, 471)
        self.assertFalse({r["image"] for r in train} & {r["image"] for r in test})
        self.assertEqual(len(train) + len(test), len(rows))

    @unittest.skipUnless(os.path.exists(os.path.join(PROJECT, "visdrone_features.csv")),
                         "visdrone_features.csv not present")
    def test_real_csv_split_never_leaks(self):
        """Same check on the real 17,316-crop VisDrone CSV."""
        with open(os.path.join(PROJECT, "visdrone_features.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        train, test, _, _ = vnb.split_by_image(rows, vnb.TEST_FRACTION, vnb.RANDOM_SEED)
        self.assertFalse({r["image"] for r in train} & {r["image"] for r in test})
        self.assertEqual(len(rows), 17316)


@unittest.skipIf(vfeat is None, "Pillow not installed (pip install -r requirements.txt)")
class TestFeatureExtraction(unittest.TestCase):

    def test_solid_red_image(self):
        """Expected: a pure red crop has red share 1.0, green and blue 0.0,
        and all its saturated pixels fall in the red hue bin."""
        from PIL import Image
        crop = Image.new("RGB", (30, 40), (255, 0, 0))
        self.assertEqual(vfeat.rgb_shares(crop), (1.0, 0.0, 0.0))
        hsv = vfeat.hsv_histogram(crop)
        self.assertAlmostEqual(hsv[0], 1.0)     # hue_red
        self.assertAlmostEqual(sum(hsv), 1.0)

    def test_gray_image_goes_to_gray_bin(self):
        """Expected: a gray crop has equal RGB shares (1/3 each) and every
        pixel in the gray bin, since gray has no meaningful hue."""
        from PIL import Image
        crop = Image.new("RGB", (25, 25), (120, 120, 120))
        for share in vfeat.rgb_shares(crop):
            self.assertAlmostEqual(share, 1 / 3)
        self.assertAlmostEqual(vfeat.hsv_histogram(crop)[-1], 1.0)  # hue_gray


if __name__ == "__main__":
    unittest.main()
