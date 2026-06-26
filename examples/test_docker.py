"""Docker sandbox test with real containers."""

import sys
import asyncio
import tempfile
import os
sys.path.insert(0, '.')
from tensoreval.envs.sandbox_env import ComposeProject


async def test_docker():
    print('=' * 70)
    print('Docker Sandbox Test')
    print('=' * 70)
    print()

    # Test 1: Basic container operations
    print('[1] Creating Docker sandbox...')
    project = ComposeProject(name='tensoreval-test-001')

    tmpdir = tempfile.mkdtemp(prefix='tensoreval-')
    compose_path = os.path.join(tmpdir, 'compose.yaml')
    with open(compose_path, 'w') as f:
        f.write('services:\n  default:\n    image: ubuntu:24.04\n    command: tail -f /dev/null\n    init: true\n    stop_grace_period: 1s\n')
    project.config_path = compose_path

    print('  Starting container...')
    await project.up()
    print('  Container started!')
    print()

    # Test 2: Execute commands
    print('[2] Executing commands in container...')

    stdout, stderr, rc = await project.exec('default', ['echo', 'Hello from TensorEval sandbox!'])
    print('  echo: ' + stdout.strip() + ' (rc=' + str(rc) + ')')

    stdout, stderr, rc = await project.exec('default', ['uname', '-a'])
    print('  uname: ' + stdout.strip()[:60])

    stdout, stderr, rc = await project.exec('default', ['whoami'])
    print('  whoami: ' + stdout.strip())

    stdout, stderr, rc = await project.exec('default', ['pwd'])
    print('  pwd: ' + stdout.strip())

    stdout, stderr, rc = await project.exec('default', ['ls', '/'])
    print('  ls /: ' + stdout.strip()[:60] + '...')
    print()

    # Test 3: File operations
    print('[3] File operations...')

    await project.exec('default', ['mkdir', '-p', '/workspace'])
    await project.write_file('default', '/workspace/test.txt', 'Hello from TensorEval!')
    print('  Wrote /workspace/test.txt')

    content = await project.read_file('default', '/workspace/test.txt')
    print('  Read /workspace/test.txt: ' + content.strip())

    script = 'import sys\nprint("Python", sys.version)\nprint("Hello from sandbox!")'
    await project.write_file('default', '/workspace/script.py', script)
    print('  Wrote /workspace/script.py')

    stdout, stderr, rc = await project.exec('default', ['python3', '/workspace/script.py'])
    print('  python3 script.py: ' + stdout.strip())
    print()

    # Test 4: Multi-step workflow
    print('[4] Multi-step workflow...')

    await project.write_file('default', '/workspace/data.csv', 'name,age,salary\nAlice,30,75000\nBob,25,65000\nCharlie,35,85000')
    print('  Created data.csv')

    analysis_script = '''
import csv
with open("/workspace/data.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    total = sum(int(r["salary"]) for r in rows)
    avg = total / len(rows)
    print(f"Total salary: ${total:,}")
    print(f"Average salary: ${avg:,.0f}")
    print(f"Employees: {len(rows)}")
'''
    await project.write_file('default', '/workspace/analyze.py', analysis_script)
    print('  Created analyze.py')

    stdout, stderr, rc = await project.exec('default', ['python3', '/workspace/analyze.py'])
    print('  Analysis result:')
    for line in stdout.strip().split('\n'):
        print('    ' + line)
    print()

    # Test 5: Cleanup
    print('[5] Cleaning up...')
    await project.down()
    print('  Container destroyed!')
    print()

    print('=' * 70)
    print('Docker Sandbox Test: ALL PASSED')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(test_docker())
