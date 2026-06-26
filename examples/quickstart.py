"""TensorEval Quickstart Example.

Demonstrates:
1. Creating a dataset
2. Defining a grader
3. Running evaluation
4. Printing results
"""

import tensoreval as te


def main():
    # 1. Create a simple math dataset
    samples = [
        {"query": "What is 2 + 2?", "reference_answer": "4", "rubrics": [{"name": "correctness", "rubric": "Must answer 4", "weight": 1.0}]},
        {"query": "What is 10 * 5?", "reference_answer": "50", "rubrics": [{"name": "correctness", "rubric": "Must answer 50", "weight": 1.0}]},
        {"query": "What is 100 / 4?", "reference_answer": "25", "rubrics": [{"name": "correctness", "rubric": "Must answer 25", "weight": 1.0}]},
        {"query": "What is 7 + 8?", "reference_answer": "15", "rubrics": [{"name": "correctness", "rubric": "Must answer 15", "weight": 1.0}]},
        {"query": "What is 9 * 9?", "reference_answer": "81", "rubrics": [{"name": "correctness", "rubric": "Must answer 81", "weight": 1.0}]},
    ]
    datasets = te.Datasets.from_dicts(samples, name="math_quickstart")
    print(f"Created dataset with {len(datasets)} samples")

    # 2. Define a simple reward function
    def correct_answer(state, **kwargs) -> float:
        completion = state.get("completion", [])
        answer = state.get("answer", "")
        if completion:
            last = completion[-1]
            response = last.get("content", "") if isinstance(last, dict) else str(last)
            return 1.0 if answer in response else 0.0
        return 0.0

    # 3. Create grader and environment
    grader = te.Grader(funcs=[correct_answer], weights=[1.0])
    env = te.SingleTurnEnv(rubric=grader, system_prompt="Solve the math problem. Give your answer as a number.")

    # 4. Run evaluation (will use the model specified)
    # Note: This requires a valid API key and model endpoint
    print("\nTo run evaluation, provide a valid model and API key:")
    print('  results = te.Evaluation.run(datasets, grader, env, model="mimo-v2.5-pro")')
    print('  print(results.summary())')

    # 5. Show what the results would look like
    print("\nExample output:")
    print("  {'model': 'mimo-v2.5-pro', 'num_runs': 5, 'avg_reward': 0.80, 'pass_rate': 0.80}")

    print("\nQuickstart complete!")
    print("See gsm8k_env.py for a more complete example.")


if __name__ == "__main__":
    main()
