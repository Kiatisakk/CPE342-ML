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
A learned number in the model equation. Which coefficients exist depends on the
model: a regression line has exactly two, the **intercept** (beta_0) and the
**slope** (beta_1), while a model of the form c_0 + c_1 e^(c_2 x) has three, and
only the first two of those enter the equation linearly.
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

### Fitting iteratively

**Loss function**:
The single number a fitting procedure tries to make small, computed from the residuals
of the whole training set. SSE and MSE are two choices of loss, not two different ideas.
_Avoid_: cost function, objective function, error function (all correct elsewhere; pick
one here).

**MSE (Mean Squared Error)**:
SSE divided by the number of observations. Dividing keeps the loss and its gradient the
same size whatever the sample size, so a learning rate chosen once keeps working.
_Avoid_: variance, average error.

**Gradient**:
The vector of partial derivatives of the *loss* with respect to every coefficient. It
points the way the loss increases fastest, which is why descent moves against it.
_Avoid_: derivative (singular — the gradient is the whole vector), slope (reserved for
beta_1), the gradient of the fitted curve (a different object entirely).

**Learning rate (eta)**:
The factor scaling each step along the negative gradient. Too small and the run never
arrives; too large and the loss climbs instead of falling.
_Avoid_: step size — the step is eta times the gradient, so the two are not the same
number.

**Iteration**:
One update of every coefficient, using a gradient computed from the whole training set.
_Avoid_: epoch (one pass over the data — the same thing only in batch gradient descent),
round, step.

**Update rule**:
The assignment applied once per iteration: new coefficients = old coefficients minus the
learning rate times the gradient.
_Avoid_: training step, learning step.

**Convergence**:
The state in which further iterations no longer move the coefficients meaningfully,
evidenced by a gradient whose norm is near zero. A loss that stops changing is weaker
evidence — that also happens when a run stalls or oscillates.
_Avoid_: "it converged" as a synonym for "it fits well" — convergence says the search
stopped moving and nothing about the quality of the fit.

**Initialisation**:
The coefficient values the first iteration starts from. Irrelevant when a closed form
exists; for an iterative fit it can decide which minimum is reached, or whether one is
reached at all.
_Avoid_: seed (that is the random number generator's seed), starting guess.

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
