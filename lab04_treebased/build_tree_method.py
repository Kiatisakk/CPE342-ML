r"""Build the tree-model notebook that follows the KNN reference notebook's method.

    ..\.venv\Scripts\python.exe build_tree_method.py
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
co = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# MBA Admissions - Decision Tree, Random Forest and Gradient Boosting

| Name | Student ID |
|---|---|
| Kiatisak Markmeeshap | 67070501005 |

Same method as `mba-admissions-data-classification-knnmethod.ipynb` by Seyedvala Khorasani
- same three-class target, same one-hot encoding, same scaling, same 80/20 split with
`random_state=4` - with the three tree models in place of KNN.

One deliberate change, and section 3 shows why it is necessary.
""")

md(r"""
## Import necessary packages
""")

co(r"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
""")

md(r"""
## Read in data
""")

co(r"""
df = pd.read_csv("MBA.csv")
print(df.head())
print(df.describe())
""")

md(r"""
## Converting categorical data to numerical format

`admission` is text with blanks. Following the reference: `Admit` becomes 1, `Waitlist`
becomes 2, and the blanks become 0 - the applicants who were turned down.
""")

co(r"""
df["admission"] = df["admission"].map({"Admit": 1, "Waitlist": 2}).fillna(0)
df["admission"].value_counts()
""")

co(r"""
categorical_columns = ["gender", "major", "race", "work_industry"]
df_encoded = pd.get_dummies(df, columns=categorical_columns)
df_encoded.shape
""")

md(r"""
## 3. The feature list, and one change to it

The reference builds its feature matrix like this:

```python
X = df_encoded[['gpa', 'gmat', 'work_exp', 'admission',
                'gender_Female', 'gender_Male', ...]].values
y = df[['admission']].values
```

`admission` appears in both. The model is handed the answer as an input column, which is
why that notebook reports 98.87% test accuracy. Trees make the problem impossible to
miss - watch what they score with the column left in.
""")

co(r"""
FEATURES = [
    "gpa", "gmat", "work_exp",
    "gender_Female", "gender_Male", "major_Business", "major_Humanities",
    "work_industry_Health Care", "work_industry_Investment Banking",
    "work_industry_Investment Management", "work_industry_Media/Entertainment",
    "work_industry_Nonprofit/Gov", "work_industry_Other", "work_industry_PE/VC",
    "work_industry_Real Estate", "work_industry_Retail", "work_industry_Technology",
]

y = df["admission"].values

leaky = preprocessing.StandardScaler().fit_transform(
    df_encoded[["admission"] + FEATURES].values.astype(float))
Xl_train, Xl_test, yl_train, yl_test = train_test_split(leaky, y, test_size=0.2, random_state=4)

print("with 'admission' left in the feature list:")
for name, model in [("Decision Tree", DecisionTreeClassifier(random_state=42)),
                    ("Random Forest", RandomForestClassifier(random_state=42))]:
    model.fit(Xl_train, yl_train)
    print(f"   {name:<15} test accuracy {metrics.accuracy_score(yl_test, model.predict(Xl_test)):.4f}")
""")

md(r"""
A perfect score on data the model has never seen. Nothing is being learned: a tree splits
on `admission` once and every leaf is pure. So `admission` comes out of the feature list
below, and every number after this point is a real one.
""")

md(r"""
## Feature matrix and target
""")

co(r"""
X = df_encoded[FEATURES].values
X
""")

co(r"""
y = df["admission"].values
y
""")

md(r"""
## Normalizing the data

Trees do not need this - they split on order, not on distance - but it is in the
reference method and it changes nothing for them, so it stays.
""")

co(r"""
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]
""")

md(r"""
## Test train split and fitting the models
""")

co(r"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)
""")

co(r"""
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

for model in models.values():
    model.fit(X_train, y_train)

predictions = {name: model.predict(X_test) for name, model in models.items()}
""")

md(r"""
## Evaluating the models
""")

co(r"""
for name, model in models.items():
    train_acc = metrics.accuracy_score(y_train, model.predict(X_train))
    test_acc = metrics.accuracy_score(y_test, predictions[name])
    print(f"{name:<18} Train set Accuracy: {train_acc:.4f}   Test set Accuracy: {test_acc:.4f}")

majority = (y_test == 0).mean()
print(f"\npredicting 'not admitted' for everyone: {majority:.4f}")
""")

md(r"""
Every model lands below the line you get by turning everyone down. The Decision Tree
memorises the training set - 99.35% against 77.56% on the test set - while Gradient
Boosting barely fits it at all, 85.77% against 82.41%, and comes closest to the majority
rule without beating it.
""")

md(r"""
## Visualizing true labels vs predicted labels
""")

co(r"""
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

for ax, (name, pred) in zip(axes, predictions.items()):
    ax.plot(y_test[:200], "go-", label="True Labels (y_test)", markersize=8, alpha=0.8)
    ax.plot(pred[:200], "bo-", label=f"Predicted ({name})", markersize=6, alpha=0.3)
    ax.set(ylabel="Labels", title=name, yticks=[0, 1, 2])
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Sample Index")
plt.tight_layout()
plt.savefig("figures/true_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md(r"""
The green line spends most of its time at 0, and so does the blue one. Where green jumps
to 1 - an applicant who was admitted - blue usually stays down. That is the picture
behind the accuracy scores: the models agree with the majority class and little else.
""")

co(r"""
for name, pred in predictions.items():
    correct = np.sum(y_test == pred)
    incorrect = np.sum(y_test != pred)
    admits_found = np.sum((y_test == 1) & (pred == 1))
    print(f"{name}")
    print(f"   correct {correct}   incorrect {incorrect}   accuracy {correct / len(y_test) * 100:.2f}%")
    print(f"   admitted applicants in the test set: {(y_test == 1).sum()}, of which found: {admits_found}")
""")

md(r"""
## What this says

Copying the reference method exactly reproduces its 98.87% - and with trees, 100% - but
that number is the feature list leaking the answer, not a model working. With the leak
closed, none of the three trees beats the rule "turn everyone down", which scores 0.8402
on this split.

The models are not useless; the task is imbalanced and three-class, and accuracy is the
wrong instrument for it. `A4_TreeBased_Classification.ipynb` takes the same data as a
binary problem and looks at precision, recall and ROC-AUC instead.
""")

nb["cells"] = C
nb.metadata.update({
    "kernelspec": {"display_name": "CPE342 (.venv)", "language": "python", "name": "cpe342"},
    "language_info": {"name": "python", "version": "3.14.6"},
})

path = "mba-admissions-data-classification-treemethods.ipynb"
nbf.write(nb, path)
print(f"wrote {path} with {len(C)} cells")
