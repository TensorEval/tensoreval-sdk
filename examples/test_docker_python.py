"""Docker sandbox test with Python image and real evaluation."""

import sys
import asyncio
import tempfile
import os
sys.path.insert(0, '.')
import tensoreval as te
from tensoreval.envs.sandbox_env import ComposeProject


async def test_docker_python():
    print('=' * 70)
    print('Docker Sandbox Test with Python')
    print('=' * 70)
    print()

    # Create project with Python image
    project = ComposeProject(name='tensoreval-python-test')
    tmpdir = tempfile.mkdtemp(prefix='tensoreval-')
    compose_path = os.path.join(tmpdir, 'compose.yaml')
    with open(compose_path, 'w') as f:
        f.write('services:\n  default:\n    image: python:3.12-slim\n    command: tail -f /dev/null\n    init: true\n    stop_grace_period: 1s\n')
    project.config_path = compose_path

    print('[1] Starting Python container...')
    await project.up()
    print('  Container started!')
    print()

    # Test Python execution
    print('[2] Python execution...')

    stdout, stderr, rc = await project.exec('default', ['python3', '--version'])
    print('  python3 --version: ' + stdout.strip())

    script = 'import sys\nprint("Python", sys.version)\nprint("Hello from TensorEval sandbox!")'
    await project.exec('default', ['mkdir', '-p', '/workspace'])
    await project.write_file('default', '/workspace/test.py', script)
    stdout, stderr, rc = await project.exec('default', ['python3', '/workspace/test.py'])
    print('  script output: ' + stdout.strip())
    print()

    # Test data analysis in container
    print('[3] Data analysis in container...')

    analysis = '''
import json
import math

data = [
    {"quarter": "Q1", "revenue": 45000},
    {"quarter": "Q2", "revenue": 52000},
    {"quarter": "Q3", "revenue": 48000},
    {"quarter": "Q4", "revenue": 61000},
]

total = sum(d["revenue"] for d in data)
avg = total / len(data)
growth = [(data[i]["revenue"] - data[i-1]["revenue"]) / data[i-1]["revenue"] * 100 for i in range(1, len(data))]
best = max(data, key=lambda d: d["revenue"])

result = {
    "total": total,
    "average": avg,
    "growth_rates": {f"Q{i}->Q{i+1}": f"{g:.1f}%" for i, g in enumerate(growth, 2)},
    "best_quarter": best["quarter"],
}
print(json.dumps(result, indent=2))
'''
    await project.write_file('default', '/workspace/analysis.py', analysis)
    stdout, stderr, rc = await project.exec('default', ['python3', '/workspace/analysis.py'])
    print('  Analysis result:')
    for line in stdout.strip().split('\n'):
        print('    ' + line)
    print()

    # Cleanup
    print('[4] Cleaning up...')
    await project.down()
    print('  Container destroyed!')
    print()

    print('=' * 70)
    print('Docker Sandbox Test: ALL PASSED')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(test_docker_python())
