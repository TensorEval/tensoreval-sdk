"""Test code generation scenario with real API."""

import sys
import asyncio
sys.path.insert(0, '.')
import tensoreval as te


async def test_code_generation():
    print('=' * 70)
    print('Scenario 2: Code Generation Agent Evaluation')
    print('=' * 70)

    datasets = te.Datasets.load_from_file('examples/scenarios/code_generation.jsonl')
    print('Loaded ' + str(len(datasets)) + ' code generation tasks')
    print()

    def code_reward(state, **kwargs) -> float:
        completion = state.get('completion', [])
        if not completion:
            return 0.0

        last = completion[-1]
        response = last.get('content', '') if isinstance(last, dict) else str(getattr(last, 'content', ''))
        response = response.strip()

        score = 0.0
        # Check if code is present
        if 'def ' in response or 'class ' in response:
            score += 0.3
        # Check for proper Python syntax
        if ':' in response and '(' in response and ')' in response:
            score += 0.2
        # Check for docstring
        if '"""' in response or "'''" in response:
            score += 0.1
        # Check for return statement
        if 'return ' in response:
            score += 0.2
        # Check for error handling
        if 'try:' in response or 'except' in response or 'if ' in response:
            score += 0.1
        # Check for reasonable length
        if 50 < len(response) < 2000:
            score += 0.1
        return min(score, 1.0)

    grader = te.Grader(funcs=[code_reward], weights=[1.0])
    env = te.SingleTurnEnv(
        rubric=grader,
        system_prompt='You are an expert Python developer. Write clean, well-documented, production-ready code. Include docstrings and handle edge cases.',
    )

    results = await te.Evaluation.run_async(
        datasets=datasets,
        grader=grader,
        env=env,
        model='mimo-v2.5-pro',
        api_key='tp-sjgd0v2ckg08hbi46e8uvkfmonq4ywqvxv4dsa27qz6xk5zg',
        base_url='https://token-plan-sgp.xiaomimimo.com/anthropic',
        workers=5,
    )

    summary = results.summary()
    print('Results:')
    print('  Model: mimo-v2.5-pro')
    print('  Tasks: ' + str(summary['num_runs']))
    print('  Avg Reward: ' + str(summary['avg_reward']))
    print('  Pass Rate: ' + str(summary['pass_rate']))
    print()
    print('Per-task:')
    for i, run in enumerate(results.runs):
        sample = datasets[i]
        completion = run.get('completion', [])
        response = ''
        if completion:
            last = completion[-1]
            response = last.get('content', '') if isinstance(last, dict) else str(getattr(last, 'content', ''))
        reward = run.get('reward', 0)
        status = 'PASS' if reward >= 0.6 else 'NEEDS_WORK'
        print('  T' + str(i+1) + ': ' + sample.input[:50] + '...')
        print('       Code length: ' + str(len(response)) + ' chars')
        print('       Has function: ' + str('def ' in response))
        print('       Reward: ' + str(round(reward, 2)) + ' [' + status + ']')
        print()


if __name__ == "__main__":
    asyncio.run(test_code_generation())
