# Checkpoint Progress Summary: "Ugly but Functional"

**Project:** Naive Bayes classifier for drone-camera detections (person / car /
other / unable to identify)
**Team:** Sweta Chandra Mohan (GitHub `swchme`). Sole team member.
**Checkpoint tag:** `checkpoint1`

## Where to find each requirement
Click any link to open the file.

1. **Implemented the core algorithm** in
   [`final-project.py`](final-project.py#L143), class `GaussianNaiveBayes`:
   - training, in [`fit`](final-project.py#L164);
   - Bayes' rule and normalization, in [`posteriors`](final-project.py#L195);
   - the "unable to identify" threshold, in [`predict`](final-project.py#L215).
2. **Demonstrated basic functionality** by running `python final-project.py`.
   - Saved output: [`demo_output.txt`](demo_output.txt).
   - Real-data run: [`visdrone_nb.py`](visdrone_nb.py), with output in
     [`visdrone_results.txt`](visdrone_results.txt) and three charts:
     [threshold sweep](chart_threshold_sweep.png),
     [learning curve](chart_learning_curve.png),
     [cross-validation](chart_cross_validation.png).
3. **Verified correctness** with 13 tests in
   [`tests/test_naive_bayes.py`](tests/test_naive_bayes.py).
   - Run them with `python -m unittest discover -s tests -v`.
   - Each test's expected result and why it matters:
     [README, "Correctness verification"](README.md#correctness-verification).
4. **Documented how to run the project** in [`README.md`](README.md):
   - [Quick start](README.md#quick-start-no-installs-needed);
   - [Full setup and dependencies](README.md#full-setup-real-visdrone-data-and-charts);
   - [`requirements.txt`](requirements.txt).
5. **Documented team progress** in this file:
   [Contributions](#contributions), [Remaining work](#remaining-work),
   [Challenges](#challenges).

## Completed functionality
- **Core algorithm:** a Gaussian Naive Bayes classifier written from scratch in
  standard-library Python, with no scikit-learn.
  - Training estimates each class's priors, per-feature means and variances.
  - Prediction combines log-priors with log Gaussian likelihoods, normalizes with
    log-sum-exp, and returns "unable to identify" when the top posterior is below
    0.70.
- **Demonstration:** `final-project.py` runs on 30 hand-made drone examples. It
  prints posteriors, a decision for each example, accuracy and a confusion matrix,
  and compares against a "tall and narrow means person" rule.
  - Naive Bayes: 7/9. Rule: 6/9.
- **Real data:** 17,316 crops from the VisDrone2019 validation set, with the train
  and test split done by image.
  - 5-fold cross-validation: 66.5% when forced to answer, 76.5% on the crops it
    answers.
  - With balanced classes, Naive Bayes beats the rule: 61.1% vs 55.9%.
  - Also done: a threshold sweep, a learning curve, and a comparison of RGB vs HSV
    color features.
- **Verification:** 13 unit tests, including hand-calculated posteriors, prior
  effects, normalization, the threshold behavior, and checks that no image leaks
  between train and test.

## Remaining work
- Choose the confidence threshold by cross-validation on training data only.
- Reduce over-confidence by removing redundant features, and try size features
  that don't depend on camera resolution.
- Integrate the exported model into the drone's tracking pipeline as a
  display-only label. This is in progress on a separate branch of the team drone
  repository.
- Final report and presentation.

## Challenges
- Most people seen from a drone are tiny: 81% are under 20 px and were excluded.
- Cars make up 61% of the data, so I report results on balanced data too.
- The features overlap, which breaks the independence assumption.
- Box sizes don't transfer between camera resolutions.

## Contributions
**Sweta Chandra Mohan (sole member):** problem selection and project design;
requirements and design decisions; data and evaluation choices (split by image,
balanced vs natural class mix, threshold, RGB vs HSV); and reviewing and directing
all code, tests and analysis. Code was developed with AI assistance (Claude Code),
following my specifications.
