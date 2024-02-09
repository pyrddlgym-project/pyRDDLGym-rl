'''In this example, the rllib package is used to train an RL agent.
    
The syntax for running this example is:

    python run_rllib.py <domain> <instance> <method> [<iters>]
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <method> is the algorithm to train (e.g. PPO, DQN etc.)
    <iters> is the number of iterations of training
'''
import sys

from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig

import pyRDDLGym
from pyRDDLGym_rl.core.agent import RLLibAgent
from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv

METHODS = {'dqn': DQNConfig, 'ppo': PPOConfig, 'sac': SACConfig}


def main(domain, instance, method, iters=20):
    
    # set up the environment
    def env_creator(env_config):
        return pyRDDLGym.make(env_config['domain'], env_config['instance'],
                              base_class=SimplifiedActionRDDLEnv)    

    register_env('RLLibEnv', env_creator)
    
    # create agent
    env_config = {'domain': domain, 'instance': instance}
    config = METHODS[method]()
    config = config.rollouts(num_rollout_workers=1)
    config = config.environment('RLLibEnv', env_config=env_config)
    algo = config.build()
    
    # train agent
    for n in range(iters):
        result = algo.train()
        print(f'iteration {n + 1} '
              f'/ min {result["episode_reward_min"]} '
              f'/ mean {result["episode_reward_mean"]} '
              f'/ max {result["episode_reward_max"]}')
    
    # wrap the agent in a RDDL policy and evaluate
    RLLibAgent(algo).evaluate(env_creator(env_config),
                              episodes=1, verbose=True, render=True)
            
        
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python run_rllib.py <domain> <instance> <method> [<iters>]')
        exit(1)
    if args[2] not in METHODS:
        print(f'<method> in {set(METHODS.keys())}')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1], 'method': args[2]}
    if len(args) >= 4: kwargs['iters'] = int(args[3])
    main(**kwargs)
