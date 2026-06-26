"""Train command — train a model using RL."""

def train_command(dataset: str, base_model: str, algorithm: str, steps: int):
    """Run training."""
    import tensoreval as te

    print(f"Loading dataset from {dataset}...")
    datasets = te.Datasets.load_from_file(dataset)
    print(f"Loaded {len(datasets)} samples.")

    print(f"Starting training with {algorithm}...")
    print(f"  Base model: {base_model}")
    print(f"  Steps: {steps}")

    trainer = te.Training.run(
        datasets=datasets,
        base_model=base_model,
        algorithm=algorithm,
        steps=steps,
    )

    print(f"\nTraining initialized: {trainer.model_id}")
    print(f"Status: {trainer.status}")
    print(f"\nTo deploy: tensoreval deploy {trainer.model_id} --name my-agent")
