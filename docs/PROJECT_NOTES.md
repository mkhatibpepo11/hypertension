# Project Notes

## Objective
Predict the dataset label `hypertension_status` (`normal` = 0, `hypertensive` = 1) from available non-leakage features. This is a classification exercise, not a clinical risk estimate.

## Original notebook
The source notebook explored distributions, missing values, duplicates, class balance, relationships with the target, IQR capping, one-hot encoding, SMOTE, Random Forest, threshold adjustment, feature importance, and test evaluation.

### Original saved results (not reproduced)
- Training: 10,000 rows; validation: 5,000; test: 2,000, as recorded in notebook outputs.
- Class 1 prevalence in training: 8,105 / 10,000.
- Reported final test accuracy: ~0.83 at threshold 0.30.
- These are historical outputs only; they should not be presented as independently verified results of the new pipeline.

## Issues found in original notebook
1. IQR bounds were fitted separately to train, validation, and test.
2. Dummy encoding was performed separately on each split, potentially causing mismatched category meaning.
3. A SMOTE cell references undefined `X`, `y`, and `X_encoded` variables in the visible notebook execution path.
4. SMOTE on one-hot-encoded categories can generate invalid fractional values.
5. The saved `.pkl` included the estimator, not a complete input-processing pipeline.
6. The data is synthetic; clinical generalization is unproven.

## Rebuilt baseline decisions
- Keep provided train/validation/test splits.
- Fit preprocessing only on training data using `ColumnTransformer`.
- Use Random Forest with `class_weight='balanced_subsample'` as a straightforward baseline.
- Do not apply SMOTE in the baseline.
- Preserve threshold 0.30 for comparison with original work; compare validation metrics at 0.50 and 0.30.
- Report test results without tuning on test data.

## Before publishing publicly
- [ ] Execute the training script successfully on a clean environment.
- [ ] Review dataset license and metadata.
- [ ] Check duplicate identities and leakage across splits.
- [ ] Check missingness and data types after loading.
- [ ] Inspect the generated `metrics.json` and report actual measured results.
- [ ] Try the Streamlit app with realistic example inputs.
- [ ] Add screenshots from your own app, not stock images.
- [ ] Compare a majority-class baseline and a logistic regression baseline.
- [ ] Document any model and threshold selection criteria.
