from dataclasses import dataclass
from typing import Optional


@dataclass
class Claim:
    """A claim such as 'our model reaches 98% accuracy', plus whatever artifacts exist."""
    value: float                          # claimed accuracy, 0..1
    metric: str = "accuracy"
    text: str = ""                        # the claim as written
    predictions_path: Optional[str] = None  # CSV: feature cols + y_true + y_pred
    train_path: Optional[str] = None        # CSV: feature cols + y_true
    eval_notes: str = ""                  # how the authors say they evaluated
    paper_text: str = ""                  # paper / report excerpt

    @classmethod
    def from_case(cls, meta: dict, root) -> "Claim":
        """Build a Claim from a case's meta.json, resolving relative file paths against root."""
        from pathlib import Path
        d = dict(meta["claim"])
        for k in ("predictions_path", "train_path"):
            if d.get(k):
                d[k] = str(Path(root) / d[k])
        return cls(**d)
