"""Training entry point for plant disease classifiers."""


def train(config_path: str = "configs/config.yaml") -> None:
    """Load configuration and run the training workflow."""
    raise NotImplementedError(
        "Training workflow is a project-specific starting point; "
        "implement it after selecting the dataset and experiment."
    )


if __name__ == "__main__":
    train()
