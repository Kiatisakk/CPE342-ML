# CPE 342 Machine Learning

Coursework repository for CPE 342 (KMUTT, Faculty of Engineering, Department of
Computer Engineering). One directory per lab/assignment. This file is the shared
vocabulary used across every lab — a glossary, not a spec.

## Language

### Fitting a model

**OLS (Ordinary Least Squares)**:
The method that chooses model coefficients by minimising the sum of squared
residuals. It is a *criterion for choosing coefficients*, not a model in itself.
_Avoid_: "linear regression" as a synonym — linear regression is the model, OLS is
how its coefficients are chosen.

**Coefficient**:
A learned number in the model equation. In simple regression there are exactly two:
the **intercept** (beta_0) and the **slope** (beta_1).
_Avoid_: parameter, weight, constant (all correct elsewhere; pick one here).

**Intercept (beta_0)**:
The predicted response when every predictor is zero. Meaningful only if zero is
inside the observed range of the predictor.
_Avoid_: constant term, bias.

**Slope (beta_1)**:
The change in the response for a one-unit increase in the predictor. Carries units:
*response units per predictor unit*.
_Avoid_: gradient (reserved for the gradient of a loss function), coefficient (too
general — that word covers the intercept too).

**Normal equations**:
The system obtained by setting the derivatives of the sum of squared residuals to
zero. Its matrix solution is the closed form of OLS.
_Avoid_: "the OLS formula" — there are several equivalent ones.

**Closed form**:
A solution computed directly by formula in one step. Contrasted with an **iterative**
solution such as gradient descent, which approaches the answer over many steps.
_Avoid_: analytical solution, exact solution.

### Judging a model

**Fitted value (y-hat)**:
The response the model predicts for an observation that was *in the training data*.
_Avoid_: prediction — reserve that word for inputs the model has not seen.

**Prediction**:
The response the model gives for an input that was not in the training data.

**Residual**:
observed minus fitted (y - y-hat), for a row that was in the training data. A
property of the fit, not of the model's future accuracy.
_Avoid_: error — error is the unobservable population quantity the residual estimates.

**SSE (Sum of Squared Errors)**:
The sum of squared residuals. The quantity OLS minimises.
_Avoid_: RSS, SSR (SSR is used elsewhere for the *regression* sum of squares — an
entirely different number).

**SST (Total Sum of Squares)**:
The sum of squared deviations of the observed responses from their mean. The error a
model that always predicts the mean would incur — the baseline R-squared compares against.

**R-squared**:
1 - SSE/SST. The proportion of the variance in the response explained by the model,
relative to always predicting the mean. Describes *fit to the data at hand*; it is not
evidence of predictive accuracy or of causation.
_Avoid_: accuracy, goodness (too vague), correlation (r is a different quantity, even
though r-squared equals R-squared in simple regression).

**Extrapolation**:
Using the model at a predictor value outside the range observed in the training data.
The fitted line carries no evidence about that region, so such predictions must always
be labelled as extrapolation.
_Avoid_: out-of-range prediction, forecasting.

### Talking about units

**Model-native units**:
The units the data was recorded in, which are also the units the coefficients speak in.
Every reported number is stated in model-native units first, with the plain real-world
count in parentheses on first mention — e.g. "16.93 thousand units (about 16,933 units)".
A bare number with no unit is never acceptable in a report.
