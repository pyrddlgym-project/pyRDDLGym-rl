import numbers
from typing import Any, Dict, Optional, Type, Union

from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.registry import ALGORITHMS

import pyRDDLGym
from pyRDDLGym_rl.core.agent import RLLibAgent
from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv
from pyRDDLGym_rl.core.progress import TrainingProgress

# all algorithms registered with rllib, keyed by lowercase name, e.g. 'ppo';
# they are only imported when used
METHODS = {name.lower(): name for name in ALGORITHMS}

ENV_NAME = 'pyRDDLGym_rl'


def _env_creator(env_config):
    env_config = dict(env_config)
    domain = env_config.pop('domain')
    instance = env_config.pop('instance')

    # the summary is printed by the trainer's own environment, not every env runner
    env_config['print_summary'] = False
    return pyRDDLGym.make(domain, instance, base_class=SimplifiedActionRDDLEnv,
                          **env_config)


def rllib_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    '''Returns the episode and learner statistics reported by rllib in the result
    of a training iteration, keyed by 'section/name' as rllib names them.

    :param result: the dictionary returned by rllib's Algorithm.train()
    '''

    # episode statistics are under env_runners, or at the top level before ray 2.10
    stats = result.get('env_runners', result)
    sections = {'env_runners': {name: value for (name, value) in stats.items()
                                if name.startswith(('episode', 'num_episodes'))}}

    # learner statistics are per module in the new API stack, else per policy
    learners = result.get('learners', None)
    if learners is not None:
        for (module, stats) in learners.items():
            if module != '__all_modules__':
                sections[f'learners/{module}'] = stats
    else:
        for (policy, stats) in result.get('info', {}).get('learner', {}).items():
            sections[f'info/learner/{policy}'] = stats.get('learner_stats', stats)

    # keep only the numbers directly in each section, as python numbers
    metrics = {}
    for (section, stats) in sections.items():
        for (name, value) in stats.items():
            if isinstance(value, numbers.Number) and not isinstance(value, bool):
                if hasattr(value, 'item'):
                    value = value.item()
                metrics[f'{section}/{name}'] = value
    return metrics


class RLLibTrainer:
    '''Trains a rllib agent on a RDDL environment, showing the training progress
    and metrics in the console.'''

    def __init__(self, domain: str, instance: str,
                 method: Union[str, Type[Algorithm], Type[AlgorithmConfig],
                               AlgorithmConfig]='ppo',
                 env_kwargs: Optional[Dict[str, Any]]=None,
                 progress: bool=True,
                 progress_kwargs: Optional[Dict[str, Any]]=None,
                 **training_kwargs) -> None:
        '''Creates a new trainer.

        :param domain: the RDDL domain, as passed to pyRDDLGym.make()
        :param instance: the RDDL instance, as passed to pyRDDLGym.make()
        :param method: the algorithm to train, either a name in METHODS (e.g. 'ppo',
        case insensitive), a rllib algorithm or algorithm config class, or a
        config instance on which other settings (e.g. resources or env runners)
        have already been set; if the config already has an env_to_module_connector,
        it must flatten the Dict observations
        :param env_kwargs: other arguments passed to pyRDDLGym.make() to create
        each SimplifiedActionRDDLEnv, e.g. the action_mode, which is 'auto' by default
        :param progress: whether to show a progress bar and table of metrics
        :param progress_kwargs: arguments passed to the TrainingProgress display
        :param **training_kwargs: arguments passed to the config's training(),
        e.g. the learning rate lr
        '''
        if isinstance(method, str):
            if method.lower() not in METHODS:
                raise ValueError(f'Method <{method}> is not one of {sorted(METHODS)}.')
            method = ALGORITHMS[METHODS[method.lower()]]()[0]
        if isinstance(method, type) and issubclass(method, Algorithm):
            config = method.get_default_config()
        elif isinstance(method, type):
            config = method()
        else:
            config = method

        # the environment is created by name in each env runner, where mixed real
        # and discrete actions are a single Box since most algorithms don't support Dict
        env_kwargs = {'action_mode': 'auto', **(env_kwargs or {})}
        self.env = pyRDDLGym.make(domain, instance, base_class=SimplifiedActionRDDLEnv,
                                  **env_kwargs)
        register_env(ENV_NAME, _env_creator)
        env_config = {'domain': domain, 'instance': instance, **env_kwargs}
        config = config.environment(ENV_NAME, env_config=env_config)

        # the new API stack (default in recent rllib) must flatten Dict observations,
        # unless a connector was already set
        if getattr(config, 'enable_env_runner_and_connector_v2', False) \
        and getattr(config, '_env_to_module_connector', None) is None:
            from ray.rllib.connectors.env_to_module import FlattenObservations
            config = config.env_runners(
                env_to_module_connector=lambda *_args, **_kwargs: FlattenObservations())
        if training_kwargs:
            config = config.training(**training_kwargs)

        self.config = config
        self.progress = progress
        self.progress_kwargs = progress_kwargs or {}
        self.algo = getattr(config, 'build_algo', config.build)()

    def train(self, iters: int) -> Dict[str, Any]:
        '''Trains the agent and returns the result of the last training iteration.

        :param iters: the number of training iterations
        '''
        result = None
        if self.progress:
            with TrainingProgress(type(self.algo).__name__, total=iters, unit='iter',
                                  **self.progress_kwargs) as display:
                for n in range(iters):
                    result = self.algo.train()
                    display.update(n + 1, rllib_metrics(result))
        else:
            for _ in range(iters):
                result = self.algo.train()
        return result

    def agent(self, deterministic: bool=True) -> RLLibAgent:
        '''Returns the trained agent as a RDDL policy, e.g. for evaluation.

        :param deterministic: whether to choose actions deterministically
        '''
        return RLLibAgent(self.algo, deterministic=deterministic)
