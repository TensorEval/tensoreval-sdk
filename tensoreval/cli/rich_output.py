"""Rich CLI output for TensorEval.

Provides progress bars, colored output, result tables, and live updates.
Based on HUD and Verifiers CLI patterns.
"""

from typing import Any


def _get_console():
    """Lazy import Rich console."""
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def print_header(title: str):
    """Print a styled header."""
    console = _get_console()
    if console:
        from rich.panel import Panel
        console.print(Panel(f"[bold]{title}[/bold]", style="blue", expand=False))
    else:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")


def print_envs_table(envs: list[dict[str, str]]):
    """Print built-in environments as a formatted table."""
    console = _get_console()
    if console:
        from rich.table import Table
        from rich import box

        table = Table(title="Built-in Environments", box=box.ROUNDED)
        table.add_column("Name", style="cyan", min_width=20)
        table.add_column("Category", style="magenta", min_width=12)
        table.add_column("Difficulty", style="yellow", min_width=10)
        table.add_column("Description", style="white")

        for env in envs:
            diff = env.get("difficulty", "")
            diff_style = "[green]" if diff == "easy" else "[yellow]" if diff == "medium" else "[red]"
            table.add_row(
                env.get("name", ""),
                env.get("category", ""),
                f"{diff_style}{diff}[/]",
                env.get("description", "")[:60],
            )

        console.print(table)
        console.print(f"\n[bold]Usage:[/bold] tensoreval eval <env-name> --model mimo-v2.5-pro")
    else:
        print("\nBuilt-in Environments:")
        for env in envs:
            print(f"  {env.get('name', ''):20s} [{env.get('category', '')}] {env.get('description', '')[:50]}")


def print_results_table(summary: dict[str, Any], per_query: list[dict[str, Any]] | None = None):
    """Print evaluation results as a formatted table."""
    console = _get_console()
    if console:
        from rich.table import Table
        from rich import box

        # Summary table
        table = Table(title="Evaluation Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model", str(summary.get("model", "unknown")))
        table.add_row("Samples", str(summary.get("num_runs", 0)))
        table.add_row("Avg Reward", f"{summary.get('avg_reward', 0):.4f}")
        table.add_row("Pass Rate", f"{summary.get('pass_rate', 0):.1%}")
        table.add_row("Passed", str(summary.get("pass_count", 0)))
        table.add_row("Failed", str(summary.get("fail_count", 0)))

        console.print(table)

        # Per-query table
        if per_query:
            qtable = Table(title="Per-Query Results", box=box.ROUNDED)
            qtable.add_column("ID", style="cyan", min_width=6)
            qtable.add_column("Query", style="white", max_width=50)
            qtable.add_column("Reward", style="green", min_width=8)
            qtable.add_column("Status", style="bold", min_width=8)

            for q in per_query:
                reward = q.get("reward", 0)
                status = "[green]PASS[/green]" if reward >= 0.8 else "[red]FAIL[/red]"
                qtable.add_row(
                    str(q.get("query_id", "")),
                    str(q.get("query", ""))[:50],
                    f"{reward:.2f}",
                    status,
                )

            console.print(qtable)
    else:
        # Fallback to plain text
        print(f"\nModel:      {summary.get('model', 'unknown')}")
        print(f"Samples:    {summary.get('num_runs', 0)}")
        print(f"Avg Reward: {summary.get('avg_reward', 0):.4f}")
        print(f"Pass Rate:  {summary.get('pass_rate', 0):.1%}")
        print(f"Passed:     {summary.get('pass_count', 0)}")
        print(f"Failed:     {summary.get('fail_count', 0)}")

        if per_query:
            print("\nPer-query:")
            for q in per_query:
                reward = q.get("reward", 0)
                status = "PASS" if reward >= 0.8 else "FAIL"
                print(f"  {q.get('query_id', '')}: {str(q.get('query', ''))[:40]}  reward={reward:.2f} [{status}]")


def print_progress(message: str, current: int = 0, total: int = 0):
    """Print a progress message."""
    console = _get_console()
    if console:
        if total > 0:
            console.print(f"[cyan]{message}[/cyan] ({current}/{total})")
        else:
            console.print(f"[cyan]{message}[/cyan]")
    else:
        if total > 0:
            print(f"{message} ({current}/{total})")
        else:
            print(message)


def print_error(message: str):
    """Print an error message."""
    console = _get_console()
    if console:
        console.print(f"[red]Error: {message}[/red]")
    else:
        print(f"Error: {message}")


def print_success(message: str):
    """Print a success message."""
    console = _get_console()
    if console:
        console.print(f"[green]{message}[/green]")
    else:
        print(message)


def print_warning(message: str):
    """Print a warning message."""
    console = _get_console()
    if console:
        console.print(f"[yellow]Warning: {message}[/yellow]")
    else:
        print(f"Warning: {message}")


def print_info(message: str):
    """Print an info message."""
    console = _get_console()
    if console:
        console.print(f"[blue]{message}[/blue]")
    else:
        print(message)


def create_progress_bar(description: str, total: int):
    """Create a progress bar context manager."""
    console = _get_console()
    if console:
        from rich.progress import Progress
        return Progress(console=console)
    return None
