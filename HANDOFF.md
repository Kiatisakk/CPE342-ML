# Handoff — CPE 342 Assignment 2 (gradient descent)

Repo: `C:\Users\kiati\Documents\CPE342-ML`, branch `master`.
Written 2026-08-23.

## Where things stand

A2 is implemented, executed, exported and committed — nine commits, `c1d83b6` through
`5f0043d` (`git log --oneline c1d83b6~1..`). The deliverable is
`lab02_gd/A2_Gradient_Descent.ipynb` (22 cells, all executed with outputs) plus
`lab02_gd/A2_Gradient_Descent.pdf` (8 pages) and `lab02_gd/figures/fit_and_loss.png`.

Nothing is half-finished. The only thing outstanding is one question the user has not
answered yet — see **Open question** below.

## Context you need that is not in the repo

- **The assignment PDF hides its dataset behind a hyperlink.** `lab02_gd/material/A2.pdf`
  says "downloaded from this link"; the URL is inside a compressed PDF stream (Google
  Drive id `13k4WwdwKGBtaCsqSStr3rWxe2pfpTvbk`). It is already downloaded and committed
  as `lab02_gd/data/gd_data.csv`, so nothing needs fetching.
- **`lab02_gd/material/Telco-Churn.csv` is not A2's data.** It shipped in the same folder
  but is a churn-classification dataset, presumably for a later assignment.
- **Correction to the plan file.** The marks are 1 / 3 / 1 / 4 / 1 across the five tasks —
  task 4 (implement) is the heaviest at 4 points. The plan file at
  `C:\Users\kiati\.claude\plans\partitioned-swinging-cupcake.md` records 2/3/2/2/1, which
  is wrong; the rest of that file is accurate. The notebook itself never prints marks, so
  nothing shipped is affected.
- **The design decisions were settled in a `/grilling` session** — data placement, MSE vs
  SSE, three implementations, fixed init at (0,0,0), 5000 iterations with no early stop.
  The reasoning is written up in the plan file; do not re-litigate it.
- **Then the user trimmed hard**, over five turns, on the principle "เหลือแค่ที่โจทย์ขอ"
  (keep only what the assignment asks). Removed: residual panel and residual table,
  learning-rate comparison figure, the finite-difference gradient check, the "why the
  normal equations do not apply here" note, the Explanation slot, and the whole Summary
  section including the closed-form-vs-GD table. Each removal is its own commit with the
  reasoning in the message. **Do not propose adding any of these back** unless asked.

## How to rebuild

The notebook is generated, not hand-edited — LaTeX-heavy markdown is easier to regenerate.

```
.venv\Scripts\python.exe <scratch>\build_a2.py      # writes the .ipynb
cd lab02_gd && ..\.venv\Scripts\python.exe run_a2.py  # executes + audits, writes outputs back
cd .. && .venv\Scripts\python.exe export_pdf.py lab02_gd/A2_Gradient_Descent.ipynb
```

where `<scratch>` is
`C:\Users\kiati\AppData\Local\Temp\claude\C--Users-kiati-Documents-CPE342-ML\758c34eb-8a8c-4823-ae22-1f1946900cc4\scratchpad`.
`build_a2.py` there is in sync with what is committed. `run_a2.py` in the same folder is
copied into `lab02_gd/` to run and deleted afterwards; it reports error cells and audits
the markdown for the MathJax traps that bit lab01 (`$` opening on a digit, odd `$` parity).

`export_pdf.py` at the repo root goes HTML → headless Edge print, because VS Code's PDF
export goes through LaTeX and has no Thai font. Verify a PDF change by rendering a page:
`pdftoppm -png -r 70 -f 8 -l 8 <pdf> <out>` (MiKTeX's pdftoppm is on PATH) and looking at
the PNG — do not trust the byte count.

## Verified numbers (do not recompute from scratch)

c0 = 1.003352, c1 = 0.999271, c2 = -1.529872; MSE 7.554135e-04, SSE 0.0755, RMSE 0.0275,
R^2 0.9854; gradient norm 2.57e-08 after 5000 iterations at eta = 0.3 from init (0,0,0).
Methods 1 and 2 agree bit-for-bit; `curve_fit` differs by up to 2.89e-06 because it stops
on its own tolerance. The notebook asserts all of this, so a bad run fails loudly.

## Open question — the user has not answered

Should Method 3 change from `scipy.optimize.curve_fit` to
`scipy.optimize.minimize(loss, C_INIT, jac=gradient, method="BFGS")`?

Measured, both converge: BFGS lands within 1.33e-05 of curve_fit in 19 iterations
(L-BFGS-B 8.68e-06, CG 1.90e-04, Nelder-Mead without a gradient is off by 4.12e-02).
The trade-off put to the user: `curve_fit` never touches our gradient so it is a fully
independent check of the *answer*, whereas BFGS with `jac=gradient` consumes the task-2
derivation and so would check the *derivation* — which nothing in the notebook does any
more, since the finite-difference check was cut.

Related facts established while answering: scipy ships no fixed-step batch gradient
descent at all; sklearn 1.9.0 is installed but `SGDRegressor` is linear-only and cannot
fit c2 inside an exponent; torch/tensorflow/autograd/jax are not installed.

The user's last message before the handoff request was "启动协作", in Chinese, which does
not parse in this context and was not an answer to the above. It is still unresolved —
ask before acting on it.

## Loose ends

- `sklearn==1.9.0` is installed in `.venv` but absent from `requirements.txt`. Nothing in
  either lab imports it. Either add it or leave it; the user has not been asked.
- `lab01_ols/A1_OLS_Regression.pdf` and `lab01_ols/material/ML_2_Training_Models.ipynb`
  show as modified, and `ML_2_Training_Models.pdf` is untracked. These predate this
  session's work and were deliberately left alone. Do not sweep them into a commit.

## Working style the user has established

- Replies in Thai, casual register, blunt. Technical terms stay in English.
- Wants the deliverable minimal — if something is not in the assignment text, expect it to
  be cut. Offer removals rather than additions.
- Verify claims by running or rendering, then report the measurement. The user pushes back
  on anything asserted without evidence.
- One commit per logical change, message explaining why, `Co-Authored-By: Claude Opus 5`.

## Suggested skills

- `mattpocock-skills:grilling` — if the user opens a new assignment (A3 looks likely, given
  Telco-Churn.csv sitting in `lab02_gd/material/`). This is how A1 and A2 were both scoped,
  and the user expects the numbered-question-with-recommendation format.
- `mattpocock-skills:domain-modeling` — pairs with the above and owns `CONTEXT.md` at the
  repo root, which is a glossary and nothing else. A3 would need classification vocabulary,
  and `CONTEXT.md` currently defines only regression and iterative-fitting terms.
- `mattpocock-skills:code-review` — if the user wants the lab02 commits reviewed before
  submission.
