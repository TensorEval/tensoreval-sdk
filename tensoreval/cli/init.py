"""Init command — scaffold a new TensorEval environment."""

import os
from pathlib import Path


PRESETS = {
    "blank": {
        "description": "Minimal scaffold",
        "system_prompt": "You are a helpful assistant.",
    },
    "cx": {
        "description": "Customer support agent",
        "system_prompt": "You are a customer support agent. Resolve issues professionally and within policy.",
    },
    "coding": {
        "description": "Code generation agent",
        "system_prompt": "You are a coding assistant. Write correct, efficient code.",
    },
    "browser": {
        "description": "Web browsing agent",
        "system_prompt": "You are a web research agent. Find accurate information.",
    },
}


def init_command(name: str, preset: str = "blank"):
    """Scaffold a new environment directory."""
    if preset not in PRESETS:
        print(f"Unknown preset '{preset}'. Available: {', '.join(PRESETS.keys())}")
        return

    env_dir = Path(name)
    if env_dir.exists():
        print(f"Directory '{name}' already exists.")
        return

    env_dir.mkdir(parents=True)
    preset_config = PRESETS[preset]

    # Create env.py
    env_py = env_dir / "env.py"
    env_py.write_text(f'''"""TensorEval environment: {name}

Preset: {preset} — {preset_config["description"]}
"""

import tensoreval as te


def load_environment():
    """Load the evaluation environment."""
    dataset = te.Datasets.load_from_file("tasks.jsonl")

    async def correct_answer(state, **kwargs):
        """Simple reward function — check if answer matches."""
        completion = state.get("completion", [])
        answer = state.get("answer", "")
        if completion:
            last = completion[-1]
            response = last.get("content", "") if isinstance(last, dict) else str(last)
            return 1.0 if answer.lower().strip() in response.lower() else 0.0
        return 0.0

    rubric = te.Rubric(funcs=[correct_answer])
    env = te.SingleTurnEnv(
        rubric=rubric,
        system_prompt="""{preset_config["system_prompt"]}""",
    )
    return env
''')

    # Create tasks.jsonl
    tasks_jsonl = env_dir / "tasks.jsonl"
    tasks_jsonl.write_text('{"query": "What is 2 + 2?", "reference_answer": "4", "rubrics": [{"name": "correctness", "rubric": "Must answer 4", "weight": 1.0}]}\n')

    # Create README
    readme = env_dir / "README.md"
    readme.write_text(f"# {name}\n\n{preset_config['description']}.\n\n## Usage\n\n```bash\ntensoreval eval tasks.jsonl --model mimo-v2.5-pro\n```\n")

    print(f"Created environment '{name}' with preset '{preset}' in ./{name}/")
    print(f"  - env.py: Environment definition")
    print(f"  - tasks.jsonl: Sample tasks")
    print(f"  - README.md: Documentation")
    print(f"\nNext steps:")
    print(f"  cd {name}")
    print(f"  tensoreval eval tasks.jsonl --model mimo-v2.5-pro")
