# Demo video script (under 3 minutes)

**0:00–0:20 — Problem.** Show the ProofPath AI title and say: “AI claims
often look precise but hide leakage, imbalance, or test-set tuning.”

**0:20–0:45 — Architecture.** Show the flow:
claim and artifacts → Strands Decider 2B → deterministic evidence tool →
finding → next decision. Explain that Amazon Bedrock Nova handles the
generative explanation, while Python owns measurements and the verdict.

**0:45–1:35 — Live demo.** Select the `imbalance` case and click
Investigate. Highlight the Decider 2B tool choice, the class-balance result,
and the contradicted verdict. Select `clean` and show the supported verdict.

**1:35–2:10 — Reusable pattern.** Open `DECISION_PATTERN.md`. Show the typed
question, closed option set, choice validation, and safe tool execution.

**2:10–2:40 — Why it matters.** Show the step count and evidence coverage.
Explain that a small model makes fast routing decisions and never fabricates
evidence. Mention offline fallback and reproducible benchmark tests.

**2:40–2:55 — Close.** Show the public repository and the tags
`#strands-decider-2b` and `#tool-selection`.
