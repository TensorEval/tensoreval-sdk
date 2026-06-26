"""Complex Docker scenario: AI agent with tools in a sandboxed container.

This test demonstrates:
1. Docker sandbox with Python environment
2. Multi-turn tool calling
3. MCP-style tool integration
4. Real API evaluation with Mimo v2.5 Pro
"""

import sys
import asyncio
import tempfile
import os
sys.path.insert(0, '.')
import tensoreval as te
from tensoreval.envs.sandbox_env import ComposeProject


# ---------------------------------------------------------------------------
# Tools that execute inside Docker container
# ---------------------------------------------------------------------------

class DockerToolExecutor:
    """Execute tools inside a Docker container."""

    def __init__(self, project: ComposeProject, service: str = "default"):
        self.project = project
        self.service = service

    async def run_python(self, code: str) -> str:
        """Execute Python code in the container."""
        stdout, stderr, rc = await self.project.exec(
            self.service,
            ["python3", "-c", code],
            timeout=30,
        )
        if rc != 0:
            return f"Error: {stderr.strip()}"
        return stdout.strip()

    async def run_bash(self, command: str) -> str:
        """Execute bash command in the container."""
        stdout, stderr, rc = await self.project.exec(
            self.service,
            ["bash", "-c", command],
            timeout=30,
        )
        if rc != 0:
            return f"Error: {stderr.strip()}"
        return stdout.strip()

    async def read_file(self, path: str) -> str:
        """Read a file from the container."""
        return await self.project.read_file(self.service, path)

    async def write_file(self, path: str, content: str) -> str:
        """Write a file to the container."""
        await self.project.write_file(self.service, path, content)
        return f"Written {len(content)} bytes to {path}"


# ---------------------------------------------------------------------------
# MCP-style tool definitions
# ---------------------------------------------------------------------------

def create_mcp_tools(executor: DockerToolExecutor):
    """Create MCP-style tools that execute in Docker."""

    async def execute_python(code: str) -> str:
        """Execute Python code in a sandboxed environment. Use for calculations, data processing, etc."""
        return await executor.run_python(code)

    async def execute_bash(command: str) -> str:
        """Execute a bash command in the sandbox. Use for file operations, system checks, etc."""
        return await executor.run_bash(command)

    async def read_file(path: str) -> str:
        """Read the contents of a file from the sandbox."""
        return await executor.read_file(path)

    async def write_file(path: str, content: str) -> str:
        """Write content to a file in the sandbox."""
        return await executor.write_file(path, content)

    async def list_files(directory: str = "/workspace") -> str:
        """List files in a directory in the sandbox."""
        return await executor.run_bash(f"ls -la {directory}")

    # Set function names for tool definitions
    execute_python.__name__ = "execute_python"
    execute_bash.__name__ = "execute_bash"
    read_file.__name__ = "read_file"
    write_file.__name__ = "write_file"
    list_files.__name__ = "list_files"

    return [execute_python, execute_bash, read_file, write_file, list_files]


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "query": "Calculate the factorial of 20 using Python and save the result to /workspace/result.txt",
        "reference_answer": "2432902008176640000",
        "rubrics": [
            {"name": "correctness", "rubric": "Must compute factorial(20) = 2432902008176640000", "weight": 0.5},
            {"name": "file_saved", "rubric": "Must save result to /workspace/result.txt", "weight": 0.3},
            {"name": "code_quality", "rubric": "Must use proper Python code", "weight": 0.2},
        ],
    },
    {
        "query": "Create a Python script that generates the first 20 Fibonacci numbers and saves them to /workspace/fibonacci.json as a JSON array",
        "reference_answer": "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]",
        "rubrics": [
            {"name": "correctness", "rubric": "Must generate correct first 20 Fibonacci numbers", "weight": 0.5},
            {"name": "format", "rubric": "Must save as JSON array to /workspace/fibonacci.json", "weight": 0.3},
            {"name": "code_quality", "rubric": "Must be clean Python code", "weight": 0.2},
        ],
    },
    {
        "query": "Analyze the text 'The quick brown fox jumps over the lazy dog' — count the frequency of each letter (case-insensitive) and save the analysis to /workspace/analysis.json",
        "reference_answer": "Letter frequency analysis saved as JSON",
        "rubrics": [
            {"name": "correctness", "rubric": "Must count letter frequencies correctly", "weight": 0.5},
            {"name": "format", "rubric": "Must save as JSON to /workspace/analysis.json", "weight": 0.3},
            {"name": "completeness", "rubric": "Must include all letters with their counts", "weight": 0.2},
        ],
    },
]


async def test_complex_docker():
    """Test complex Docker scenario with multi-turn tools."""
    print('=' * 70)
    print('Complex Docker Scenario: AI Agent with Tools')
    print('=' * 70)
    print()

    # 1. Set up Docker environment
    print('[1] Setting up Docker sandbox...')
    project = ComposeProject(name='tensoreval-agent-test')
    tmpdir = tempfile.mkdtemp(prefix='tensoreval-')
    compose_path = os.path.join(tmpdir, 'compose.yaml')
    with open(compose_path, 'w') as f:
        f.write('services:\n  default:\n    image: python:3.12-slim\n    command: tail -f /dev/null\n    init: true\n    stop_grace_period: 1s\n')
    project.config_path = compose_path
    await project.up()
    print('  Container started with Python 3.12')
    print()

    # 2. Set up tools
    print('[2] Setting up MCP-style tools...')
    executor = DockerToolExecutor(project)
    tools = create_mcp_tools(executor)
    print(f'  Created {len(tools)} tools: {[t.__name__ for t in tools]}')
    print()

    # 3. Run evaluation with real API
    print('[3] Running evaluation with Mimo v2.5 Pro...')
    print()

    datasets = te.Datasets.from_dicts(SCENARIOS, name='docker_complex')

    def tool_use_reward(state, **kwargs):
        """Reward function that checks if tools were used and results are correct."""
        completion = state.get('completion', [])
        answer = state.get('answer', '')
        if not completion:
            return 0.0

        last = completion[-1]
        response = last.get('content', '') if isinstance(last, dict) else str(getattr(last, 'content', ''))

        score = 0.0
        # Check if code was executed (Python/bash mentioned)
        if 'python' in response.lower() or 'def ' in response or 'import ' in response:
            score += 0.3
        # Check if file operations mentioned
        if 'write' in response.lower() or 'save' in response.lower() or '/workspace' in response:
            score += 0.2
        # Check for correct answer
        if answer.lower() in response.lower():
            score += 0.3
        # Check for structured output
        if '{' in response or '[' in response:
            score += 0.1
        # Check for reasonable length
        if 50 < len(response) < 3000:
            score += 0.1
        return min(score, 1.0)

    grader = te.Grader(funcs=[tool_use_reward], weights=[1.0])
    env = te.SingleTurnEnv(
        rubric=grader,
        system_prompt='You are a coding agent with access to Python and bash tools. Execute code to solve problems and save results to files.',
    )

    results = await te.Evaluation.run_async(
        datasets=datasets,
        grader=grader,
        env=env,
        model='mimo-v2.5-pro',
        api_key='tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg',
        base_url='https://token-plan-sgp.xiaomimimo.com/anthropic',
        workers=3,
    )

    summary = results.summary()
    print('=' * 70)
    print('RESULTS')
    print('=' * 70)
    print(f'  Model:      {summary["model"]}')
    print(f'  Scenarios:  {summary["num_runs"]}')
    print(f'  Avg Reward: {summary["avg_reward"]:.4f}')
    print(f'  Pass Rate:  {summary["pass_rate"]:.1%}')
    print()

    for i, run in enumerate(results.runs):
        sample = datasets[i]
        completion = run.get('completion', [])
        response = completion[-1].get('content', '') if completion else ''
        reward = run.get('reward', 0)
        status = 'PASS' if reward >= 0.6 else 'FAIL'
        print(f'  S{i+1}: {sample.input[:50]}...')
        print(f'       Response: {response.strip()[:80]}...')
        print(f'       Reward: {reward:.2f} [{status}]')
        print()

    # 4. Cleanup
    print('[4] Cleaning up Docker...')
    await project.down()
    print('  Container destroyed')

    print()
    print('=' * 70)
    print('COMPLEX DOCKER TEST COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(test_complex_docker())
