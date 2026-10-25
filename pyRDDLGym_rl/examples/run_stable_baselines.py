'''In this example, the stable baselines package is used to train an RL agent.

The syntax for running this example is:

    python run_stable_baselines.py <domain> <instance> <method> [<steps>] [<learning_rate>]

where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <method> is the algorithm to train (e.g. PPO, DQN etc.)
    <steps> is the number of trials to simulate for training
    <learning_rate> is the learning rate to use to train the agent
'''
import sys

import pyRDDLGym

from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv
from pyRDDLGym_rl.core.sb3_trainer import METHODS, StableBaselinesTrainer


def main(domain, instance, method, steps=200000, learning_rate=None):

    # set up the environment
    env = pyRDDLGym.make(domain, instance, base_class=SimplifiedActionRDDLEnv,
                         action_mode='auto')

    # train the agent
    kwargs = {}
    if learning_rate is not None:
        kwargs['learning_rate'] = learning_rate
    trainer = StableBaselinesTrainer(env, method, **kwargs)
    trainer.train(steps)

    # wrap the agent in a RDDL policy and evaluate
    trainer.agent().evaluate(env, episodes=1, verbose=True, render=True)
    env.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python run_stable_baselines.py <domain> <instance> <method> [<steps>] [<learning_rate>]')
        exit(1)
    if args[2].lower() not in METHODS:
        print(f'<method> in {sorted(METHODS)}')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1], 'method': args[2]}
    if len(args) >= 4: kwargs['steps'] = int(args[3])
    if len(args) >= 5: kwargs['learning_rate'] = float(args[4])
    main(**kwargs)
