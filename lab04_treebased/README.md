# Lab 04 - Tree-Based Classification

Decision Tree, Random Forest and Gradient Boosting on `MBA.csv`, all at scikit-learn
defaults with `random_state=42`. No tuning: the point is to watch what each model does
with an imbalanced target, not to win a leaderboard.

## Running it

```
..\.venv\Scripts\python.exe build_notebook.py     # regenerate the notebook
..\.venv\Scripts\python.exe run_experiment.py     # same experiment as a script
```

The notebook is generated, not hand-edited - edit `build_notebook.py` and rebuild.

## Data

`MBA.csv` - synthetic data generated from the Wharton Class of 2025's statistics.
6,194 applicants, 10 columns. `admission` records `Admit` (900) or `Waitlist` (100) and
is null for the other 5,194; the column documentation gives null as Deny, so the label is
`admission == "Admit"` and the classes land at 14.5% / 85.5%. `application_id` is dropped
as a serial number a tree would happily split on.

## Results

| Model | Train acc | Test acc | Gap | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Decision Tree | 0.999 | 0.802 | 0.197 | 0.340 | 0.383 | 0.360 | 0.628 |
| Random Forest | 0.999 | 0.852 | 0.147 | 0.480 | 0.200 | 0.282 | 0.847 |
| Gradient Boosting | 0.871 | 0.855 | 0.017 | 0.500 | 0.106 | 0.174 | 0.857 |

Saying no to every applicant scores **0.855**, which is what Gradient Boosting scores.

## What the models do

Of the 180 applicants in the test set who were actually admitted:

| Model | found | missed | offers made |
|---|---|---|---|
| Decision Tree | 69 | 111 | 203 |
| Random Forest | 36 | 144 | 75 |
| Gradient Boosting | 19 | 161 | 38 |

Across the three, overfitting collapses (19.7 → 14.7 → 1.7 points) and ranking improves
(AUC 0.628 → 0.847 → 0.857) while recall falls (0.383 → 0.200 → 0.106). The model that
generalises best is the one that has most thoroughly learned to say no. Both ensembles
rank applicants well and decide badly, which is a threshold problem, not a ranking one.

## Files

```
A4_TreeBased_Classification.ipynb    the write-up, executed
build_notebook.py                     generates it
run_experiment.py                     the same experiment as a plain script
MBA.csv                               the data
figures/                              the four figures the notebook saves
maximize_f1.py, extreme_fe.py         separate excursions: SMOTE, XGBoost, stacking,
                                      heavy feature engineering. Not part of the
                                      baseline comparison above.
```
