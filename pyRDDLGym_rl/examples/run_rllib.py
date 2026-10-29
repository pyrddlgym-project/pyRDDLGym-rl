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

from pyRDDLGym_rl.core.rllib_trainer import METHODS, RLLibTrainer


def main(domain, instance, method, iters=30):

    # train the agent
    env_kwargs = {'action_mode': 'box'} if method.lower() == 'sac' else None
    trainer = RLLibTrainer(domain, instance, method, env_kwargs=env_kwargs)
    trainer.train(iters)

    # wrap the agent in a RDDL policy and evaluate
    trainer.agent().evaluate(trainer.env, episodes=1, verbose=True, render=True)
    trainer.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python run_rllib.py <domain> <instance> <method> [<iters>]')
        exit(1)
    if args[2].lower() not in METHODS:
        print(f'<method> in {sorted(METHODS)}')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1], 'method': args[2]}
    if len(args) >= 4: kwargs['iters'] = int(args[3])
    main(**kwargs)
