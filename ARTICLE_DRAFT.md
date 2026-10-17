# ProofPath AI: an evidence detective for AI claims

ProofPath AI investigates claims such as “our model reaches 98% accuracy.”
Instead of asking a large language model to guess whether the claim is true,
it gathers reproducible evidence: it recomputes the metric, checks class
imbalance, estimates uncertainty, finds train/test leakage, reads the
evaluation protocol, and cross-checks the paper.

The meaningful decision handed to **Strands Decider 2B** is tool selection:
given the claim, available artifacts, and findings so far, which one evidence
check should run next? The choice is typed and closed-set. This lets the
small model make a fast, repeatable routing decision while deterministic
Python tools calculate every result and the agent owns the final verdict.

The generative part uses Amazon Bedrock Nova for an optional explanation path.
The architecture is intentionally hybrid: a model can route and explain, but
it cannot invent a measurement. If Decider 2B or AWS credentials are
unavailable, the local policy keeps the demo usable and reports the fallback.

The reusable recipe is documented in `DECISION_PATTERN.md`. Builders can
replace the evidence tools with their own tool list while keeping the same
typed question, validation, and safe execution boundary.

## What we learned

Small decision models are a good fit for frequent, bounded choices. A large
model is valuable for the user-facing explanation, but it is unnecessary for
choosing one item from a short list. The UI makes this visible by showing the
selected tool, rationale, evidence, and final verdict.
