# Reusable pattern: Decider 2B tool selection

This project demonstrates the **tool-selection** pattern required by the
Agent Experience Hackathon.

## Typed decision

**State:** the claim, available artifacts, and findings already collected.

**Question:** `Which evidence check should run next?`

**Choices:** only the currently available deterministic checks, such as
`recompute_metric`, `class_balance`, `sample_size`, `per_class`, `leakage`,
`eval_method`, and `paper_check`.

## Wiring

[`StrandsDecider`](./proofpath/decider.py) calls the official
`strands-decider ask` CLI and parses its choice. [`run_agent`](./proofpath/agent.py)
then executes the selected Python tool. The tools calculate evidence and the
agent computes the verdict; Decider 2B cannot invent evidence or a verdict.

```python
decider = StrandsDecider()
report = run_agent(claim, decider)
```

The decider has a local adaptive fallback for development without the model
installed. The production demo pairs this decision router with
[`BedrockDecider`](./proofpath/decider.py), which uses Amazon Bedrock Nova for
the generative explanation path.

## Reuse in another agent

1. Build a compact state string from the user request and prior results.
2. Ask one typed choice question with a closed option set.
3. Validate that the returned choice is in the option set.
4. Execute the selected tool in normal application code.
5. Keep the model out of the final measurement or authorization decision.

This pattern is useful for routing support tickets, selecting retrieval
strategies, or choosing validation checks.
