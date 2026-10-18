# ProofPath AI: Check Whether an AI Accuracy Claim Is Trustworthy

ProofPath AI is a fact-checking application for machine-learning results.
It checks claims such as:

> “Our AI model reaches 98% accuracy.”

It is **not a cancer detector, medical diagnosis system, or treatment
recommendation tool**. A cancer-detection model could be one example of a
model whose accuracy claim is reviewed, but ProofPath does not inspect
patients, diagnose disease, or make clinical decisions.

The application reads a model's prediction data and supporting notes, looks
for common evaluation problems, and reports whether the claim is:

- **supported** — the supplied evidence passes the core checks;
- **contradicted** — a serious problem was found; or
- **insufficient evidence** — the claim may be true, but the evidence is too
  weak or incomplete.

An explainable agent uses Amazon Bedrock or a local policy to choose **which
check to run next**. Deterministic Python tools perform every measurement and
the final verdict, so the language model cannot invent evidence or decide the
result.

## What it checks

ProofPath can detect:

- accuracy that does not match the released predictions;
- a model that is no better than always guessing the majority class;
- a test set that is too small;
- classes that are mostly misclassified despite a good overall accuracy;
- test rows that also appear in the training data;
- hyperparameters tuned on the test set; and
- a paper or report that gives a different accuracy number.

The project is domain-independent: the labels could represent spam, fraud,
objects, customer churn, or a medical condition. ProofPath checks the
**evaluation claim**, not the underlying subject.

## Setup (Python 3.10+)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python benchmark/make_cases.py      # generates 7 claims with planted flaws
pytest -q                           # runs the test suite
python benchmark/run_benchmark.py   # compares deciders
streamlit run app.py                # demo UI
```

### Run with Amazon Bedrock

The UI defaults to **Amazon Bedrock (Nova)**. The application uses the
Bedrock Runtime Converse API and defaults to `amazon.nova-lite-v1:0`.
Configure AWS credentials using the normal AWS SDK chain (for example an AWS
profile, IAM role, or environment variables), then set a region:

```bash
aws configure
set AWS_REGION=us-east-1       # PowerShell: $env:AWS_REGION="us-east-1"
set BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
streamlit run app.py
```

The required IAM permission is `bedrock:InvokeModel` for the selected model.
If Bedrock is unavailable, the agent explicitly reports the SDK error type
and continues with the local adaptive policy, so the demo remains runnable
offline. The verdict is always computed by the evidence tools.

## How it works

```
claim + artifacts  ->  Decider picks next check  ->  tool runs  ->  finding
        ^                                                              |
        +------------------------ evidence so far <-------------------+
stops on a decisive failure or when all applicable checks are done -> verdict
```

| Tool | Needs | Catches |
|---|---|---|
| class_balance | predictions CSV | accuracy no better than always guessing the majority class |
| recompute_metric | predictions CSV | claimed number differs from the released predictions |
| sample_size | predictions CSV | test set too small (Wilson 95% interval) |
| per_class | predictions CSV | accuracy hiding classes that are mostly wrong |
| leakage | predictions + train CSV | test rows that also appear in training data |
| eval_method | evaluation notes | tuning on the test set, no held-out data |
| paper_check | paper text | paper reports a different number, or gives no baseline |

Verdicts: **supported**, **contradicted**, or **insufficient_evidence**. Checks whose
input is missing are listed as *unverified*, never silently skipped.

CSV format: feature columns, plus `y_true` and `y_pred` (predictions file) or `y_true`
(training file).

## First walkthrough

1. Start the app with `streamlit run app.py`.
2. Select **Demo case** on the main page.
3. Try the `imbalance` case and click **Run evidence review**. It should be
   contradicted because always predicting the majority class already gives
   almost the claimed accuracy.
4. Try the `clean` case. Its evidence should be supported.
5. Try `tiny_test` to see **insufficient evidence**: a perfect result on only
   40 test rows is not strong enough.

The **decision brain** only chooses the next evidence check. The evidence
trail and verdict come from the deterministic Python tools.

## Optional remote decision models

### Amazon Bedrock

`BedrockDecider` sends the claim, available artifacts, and findings so far to
an Amazon Bedrock model. The model must return one available tool name. It is
never given authority to mark evidence as true or false.

### Plugging in Decider 2B

`ModelDecider` (proofpath/decider.py) sends a short state summary and the list of
remaining checks, and expects one check name back. By default it POSTs
`{"prompt": ..., "choices": [...]}` to `$DECIDER_URL` and reads `{"choice": ...}`.
If your Decider 2B deployment has another interface, change only `_default_client`.
Bad or failed answers fall back to the heuristic policy, so the demo never stalls.

```bash
export DECIDER_URL=http://localhost:8000/decide
python benchmark/run_benchmark.py   # adds a "Decider 2B" column
```

## Benchmark results (seeded, reproducible)

Mean steps to verdict across the 7 cases, 100% verdict accuracy for all three:
random 4.10, fixed checklist 4.57, adaptive 3.14. The gain comes from flawed claims:
the adaptive policy finds planted flaws in 1 to 3 steps where the checklist needs up to 7.

## Extend it

Add a function returning a `Finding` in `proofpath/tools.py`, register it in `TOOLS`,
and give it a score rule in `AdaptiveDecider._score`. Ideas: bootstrap significance
between two models, calibration error, duplicate detection with near-matches,
checking a cited dataset's license and size.

## Limits

Accuracy claims only. Inputs are CSVs plus text, so it verifies a claim only as far as
the supplied artifacts allow. Claim extraction from free text is out of scope.
The benchmark is synthetic, which gives known answers but not real-world messiness.
