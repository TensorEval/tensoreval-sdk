"""GSM8K Environment — grade-school math evaluation.

Shows how to:
- Load a HuggingFace dataset
- Create a Verifiers-style environment
- Use the extract_boxed_answer parser
- Run evaluation
"""

import tensoreval as te


def load_environment():
    """Load the GSM8K evaluation environment."""
    # Load GSM8K dataset
    dataset = te.Datasets.from_huggingface("gsm8k", split="test", n=20, name="gsm8k_eval")

    # Define reward function
    def gsm8k_reward(state, **kwargs) -> float:
        """Check if the model's answer matches the ground truth."""
        completion = state.get("completion", [])
        answer = state.get("answer", "")
        if not completion:
            return 0.0
        last = completion[-1]
        response = last.get("content", "") if isinstance(last, dict) else str(last)
        # Extract boxed answer if present
        extracted = te.extract_boxed_answer(response, strict=True)
        if not extracted:
            extracted = response.strip()
        # Compare
        return 1.0 if extracted == answer else 0.0

    # Create rubric
    rubric = te.Rubric(funcs=[gsm8k_reward], weights=[1.0])

    # Create environment
    env = te.SingleTurnEnv(
        rubric=rubric,
        system_prompt="Solve the grade-school math problem. Reason step by step, then put your final answer within \\boxed{}.",
        dataset=dataset,
    )

    return env, dataset


def main():
    env, dataset = load_environment()
    print(f"Loaded GSM8K environment with {len(dataset)} samples")
    print(f"\nFirst sample:")
    sample = dataset[0]
    print(f"  Question: {sample.input}")
    print(f"  Answer: {sample.target}")

    print(f"\nTo evaluate:")
    print(f"  results = te.Evaluation.run(dataset, te.Grader(funcs=[gsm8k_reward]), env, model='mimo-v2.5-pro')")
    print(f"  print(results.summary())")


if __name__ == "__main__":
    main()
