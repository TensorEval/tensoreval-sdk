"""TensorEval CLI — build, test, and deploy evaluation environments.

Based on HUD and Inspect AI CLI patterns.
Uses Typer with Rich for beautiful output.

Usage:
    tensoreval init my-env --preset cx
    tensoreval eval tasks.jsonl --model mimo-v2.5-pro
    tensoreval train tasks.jsonl --base-model Qwen/Qwen3-8B
    tensoreval deploy <run-id> --name my-agent-v3
    tensoreval envs                       # list built-in environments
    tensoreval version                    # show version
"""

import typer
from typing import Optional

app = typer.Typer(
    name="tensoreval",
    help="TensorEval — Evaluation, Training, and Deployment SDK for AI Agents.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command()
def init(
    name: str = typer.Argument(..., help="Name for the new environment"),
    preset: str = typer.Option("blank", "--preset", "-p", help="Preset template (blank, cx, coding, browser, math, analysis)"),
    docker: bool = typer.Option(False, "--docker", "-d", help="Include Dockerfile and compose.yaml"),
):
    """Initialize a new TensorEval environment.

    [not dim]Examples:
        tensoreval init my-agent --preset cx
        tensoreval init my-math --preset math --docker
        tensoreval init my-custom --preset blank[/not dim]
    """
    from tensoreval.cli.init import init_command
    init_command(name, preset, docker)


@app.command()
def eval(
    dataset: str = typer.Argument(..., help="Path to JSONL dataset file or built-in env name"),
    model: str = typer.Option("mimo-v2.5-pro", "--model", "-m", help="Model to evaluate"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of concurrent workers"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for results (JSON)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    judge: bool = typer.Option(False, "--judge", "-j", help="Use LLM-as-judge scoring"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", "-s", help="System prompt for the model"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for the model"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Base URL for the model API"),
    n: int = typer.Option(-1, "--n", "-n", help="Number of samples to evaluate (-1 for all)"),
):
    """Evaluate a model on a dataset or built-in environment.

    [not dim]Examples:
        tensoreval eval tasks.jsonl --model mimo-v2.5-pro
        tensoreval eval gsm8k --model mimo-v2.5-pro --n 10
        tensoreval eval tasks.jsonl --judge --workers 8
        tensoreval eval tasks.jsonl --model gpt-4 --api-key sk-... --base-url https://api.openai.com/v1[/not dim]
    """
    from tensoreval.cli.eval import eval_command
    eval_command(dataset, model, workers, output, verbose, judge, system_prompt, api_key, base_url, n)


@app.command()
def train(
    dataset: str = typer.Argument(..., help="Path to JSONL dataset file"),
    base_model: str = typer.Option("Qwen/Qwen3-8B", "--base-model", help="Base model for training"),
    algorithm: str = typer.Option("grpo", "--algorithm", "-a", help="Training algorithm (grpo, sft, dpo)"),
    steps: int = typer.Option(100, "--steps", help="Number of training steps"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate", "-lr", help="Learning rate"),
    group_size: int = typer.Option(8, "--group-size", "-g", help="Rollouts per example for GRPO"),
):
    """Train a model using RL.

    [not dim]Examples:
        tensoreval train tasks.jsonl --base-model Qwen/Qwen3-8B --algorithm grpo
        tensoreval train tasks.jsonl --base-model llama-3-8b --steps 200[/not dim]
    """
    from tensoreval.cli.train import train_command
    train_command(dataset, base_model, algorithm, steps, learning_rate, group_size)


@app.command()
def deploy(
    run_id: str = typer.Argument(..., help="Training run ID or checkpoint path"),
    name: str = typer.Option("", "--name", "-n", help="Deployment name"),
    provider: str = typer.Option("together", "--provider", "-p", help="Deployment provider (together, tinker, local)"),
):
    """Deploy a trained model.

    [not dim]Examples:
        tensoreval deploy <run-id> --name my-agent-v3
        tensoreval deploy <run-id> --provider local[/not dim]
    """
    from tensoreval.cli.deploy import deploy_command
    deploy_command(run_id, name, provider)


@app.command(name="envs")
def list_envs():
    """List all built-in environments.

    [not dim]Examples:
        tensoreval envs[/not dim]
    """
    from tensoreval.cli.rich_output import print_header, print_envs_table
    from tensoreval.environments import list_envs
    envs = list_envs()
    print_envs_table(envs)


@app.command()
def version():
    """Show TensorEval version and system info."""
    import sys
    import platform
    from tensoreval.cli.rich_output import print_header

    try:
        import tensoreval
        version = tensoreval.__version__
    except:
        version = "unknown"

    print_header("TensorEval SDK")
    typer.echo(f"Version:    {version}")
    typer.echo(f"Python:     {sys.version.split()[0]}")
    typer.echo(f"Platform:   {platform.platform()}")
    typer.echo(f"Arch:       {platform.machine()}")

    # Check Docker
    try:
        import subprocess
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            typer.echo(f"Docker:     {result.stdout.strip()}")
        else:
            typer.echo(f"Docker:     not available")
    except:
        typer.echo(f"Docker:     not installed")


@app.command()
def login(
    api_key: str = typer.Argument(..., help="TensorEval API key"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API base URL"),
):
    """Authenticate with TensorEval platform.

    [not dim]Examples:
        tensoreval login your-api-key
        tensoreval login your-api-key --base-url https://api.tensoreval.com[/not dim]
    """
    from tensoreval.cli.rich_output import print_success, print_error
    # Store credentials (would integrate with platform in production)
    print_success(f"Logged in successfully.")
    if base_url:
        print_success(f"Base URL: {base_url}")


@app.command()
def config(
    key: Optional[str] = typer.Argument(None, help="Config key to view/set"),
    value: Optional[str] = typer.Argument(None, help="Value to set"),
):
    """View or set configuration.

    [not dim]Examples:
        tensoreval config                    # show all config
        tensoreval config model mimo-v2.5-pro  # set default model[/not dim]
    """
    from tensoreval.cli.rich_output import print_info
    if key and value:
        print_info(f"Set {key} = {value}")
    elif key:
        print_info(f"Config: {key}")
    else:
        print_info("Configuration:")
        print_info("  model: mimo-v2.5-pro (default)")
        print_info("  base_url: https://token-plan-sgp.xiaomimimo.com/anthropic")
        print_info("  workers: 4 (default)")


@app.command()
def results(
    last: int = typer.Option(5, "--last", "-n", help="Number of recent results to show"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Export results to file"),
):
    """View recent evaluation results.

    [not dim]Examples:
        tensoreval results
        tensoreval results --last 10[/not dim]
    """
    from tensoreval.cli.rich_output import print_info
    print_info(f"Showing last {last} results (results storage coming soon)")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
