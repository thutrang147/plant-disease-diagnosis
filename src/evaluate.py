"""Evaluation helpers and entry point for trained classifiers."""


def evaluate(config_path: str = "configs/config.yaml") -> None:
    """Load a trained checkpoint and report evaluation metrics."""
    raise NotImplementedError(
        "Evaluation workflow is a project-specific starting point; "
        "implement it after selecting a checkpoint and test dataset."
    )


if __name__ == "__main__":
    evaluate()
