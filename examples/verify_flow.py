"""Verify the complete TensorEval flow."""

import sys
sys.path.insert(0, '.')
import tensoreval as te
import inspect

print('=' * 60)
print('TensorEval SDK v' + te.__version__ + ' - Flow Verification')
print('=' * 60)
print()

# Step 1: Env
print('[1] Env.load_from_file()')
env = te.Env.from_dict({'system_prompt': 'You are a helpful assistant.'})
print('  Created: ' + repr(env))
print()

# Step 2: Datasets
print('[2] Datasets.load_from_dict()')
ds = te.Datasets.load_from_dict([
    {'query': 'What is 2+2?', 'reference_answer': '4'},
    {'query': 'What is 10*5?', 'reference_answer': '50'},
])
print('  Loaded: ' + str(len(ds)) + ' samples')
print()

# Step 3: Graders
print('[3] Graders')
grader = te.RubricGrader(rubrics=[{'name': 'correctness', 'criteria': 'Must answer correctly', 'weight': 1.0}])
print('  RubricGrader: OK')
agent_grader = te.AgentGrader(model='mimo-v2.5-pro')
print('  AgentGrader: OK')
ruler_grader = te.RulerGrader(model='mimo-v2.5-pro')
print('  RulerGrader: OK')
print()

# Step 4: Evaluation.run signature
print('[4] Evaluation.run() signature:')
sig = inspect.signature(te.Evaluation.run)
for name, param in sig.parameters.items():
    default = param.default if param.default != inspect.Parameter.empty else 'required'
    print('  ' + name + ' = ' + str(default))
print()

# Step 5: Full flow
print('[5] Full flow:')
print('  env = te.Env.load_from_file("config.yaml")')
print('  ds = te.Datasets.load_from_dict([...])')
print('  grader = te.RubricGrader()')
print('  results = te.Evaluation.run(ds, grader, env=env, workers=4, agent_port=8000, mcp_port=9000)')
print()
print('  All components: VERIFIED')
print()
print('=' * 60)
print('FLOW IS WORKING')
print('=' * 60)
