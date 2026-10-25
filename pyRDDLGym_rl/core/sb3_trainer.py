import inspect
import numpy as np
import sys
from typing import Any, Dict, Iterable, Optional, Tuple, Type, Union

import gymnasium as gym
import stable_baselines3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter

from pyRDDLGym_rl.core.agent import StableBaselinesAgent
from pyRDDLGym_rl.core.progress import TrainingProgress


def _find_algorithms(*modules) -> Dict[str, Type[BaseAlgorithm]]:
    return {name.lower(): value
            for module in modules for (name, value) in vars(module).items()
            if inspect.isclass(value) and issubclass(value, BaseAlgorithm)}


# all algorithms exported by stable baselines and sb3_contrib (if installed),
# keyed by lowercase name, e.g. 'ppo'
METHODS = _find_algorithms(stable_baselines3)
try:
    import sb3_contrib
    METHODS.update(_find_algorithms(sb3_contrib))
except ImportError:
    pass


class _ProgressWriter(KVWriter):
    '''A stable baselines logger output that sends the logged values to a training
    progress display, i.e. the values that would otherwise be printed in a block
    to stdout.'''

    def __init__(self, display: TrainingProgress, hidden_keys: Iterable[str]) -> None:
        self.display = display
        self.hidden_keys = set(hidden_keys)

    def write(self, key_values: Dict[str, Any],
              key_excluded: Dict[str, Tuple[str, ...]], step: int=0) -> None:
        for (key, value) in key_values.items():
            excluded = key_excluded.get(key, None)
            if key in self.hidden_keys or (
                excluded is not None and ('stdout' in excluded or 'log' in excluded)):
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (int, float, str)):
                self.display.metrics[key] = value

    def close(self) -> None:
        pass


class TrainingProgressCallback(BaseCallback):
    '''A stable baselines callback for displaying progress and metrics in a bar.'''

    # these are already shown by the progress bar
    DEFAULT_HIDDEN_KEYS = ('time/fps', 'time/time_elapsed', 'time/total_timesteps')

    def __init__(self, columns: int=3,
                 refresh_per_second: float=3,
                 hidden_keys: Iterable[str]=DEFAULT_HIDDEN_KEYS,
                 verbose: int=0) -> None:
        '''Creates a new progress callback.

        :param columns: maximum number of metrics to show per row of the table,
        which is reduced if the rows do not fit in the console
        :param refresh_per_second: how many times per second to redraw the display
        :param hidden_keys: logged metrics that are not shown in the table
        :param verbose: verbosity level of the callback
        '''
        super(TrainingProgressCallback, self).__init__(verbose=verbose)
        self.columns = columns
        self.refresh_per_second = refresh_per_second
        self.hidden_keys = hidden_keys

    def _on_training_start(self) -> None:

        # remove timesteps that were done in previous training sessions
        self._start = self.model.num_timesteps
        self._display = TrainingProgress(
            type(self.model).__name__,
            total=self.locals['total_timesteps'] - self._start,
            unit='step',
            columns=self.columns,
            refresh_per_second=self.refresh_per_second)

        # capture the logged metrics, and stop them from being printed to stdout
        logger = self.model.logger
        self._writer = _ProgressWriter(self._display, self.hidden_keys)
        self._stdout_formats = [
            output for output in logger.output_formats
            if isinstance(output, HumanOutputFormat) and output.file is sys.stdout]
        logger.output_formats = [
            output for output in logger.output_formats
            if output not in self._stdout_formats] + [self._writer]
        self._display.start()

    def _on_step(self) -> bool:
        self._display.update(self.model.num_timesteps - self._start)
        return True

    def _on_training_end(self) -> None:
        self._display.update(self.model.num_timesteps - self._start)
        self._display.stop()

        # restore the original logger outputs
        logger = self.model.logger
        logger.output_formats = [
            output for output in logger.output_formats
            if output is not self._writer] + self._stdout_formats


class StableBaselinesTrainer:
    '''Trains a stable baselines agent on an environment, showing the training
    progress and metrics in the console.'''

    def __init__(self, env: gym.Env,
                 method: Union[str, Type[BaseAlgorithm]]='ppo',
                 policy: Union[str, Type]='MultiInputPolicy',
                 progress: bool=True,
                 progress_kwargs: Optional[Dict[str, Any]]=None,
                 **model_kwargs) -> None:
        '''Creates a new trainer.

        :param env: the environment to train on
        :param method: the algorithm to train, either a name in METHODS (e.g. 'ppo',
        case insensitive) or a stable baselines algorithm class
        :param policy: the policy passed to the algorithm
        :param progress: whether to show a progress bar and table of metrics
        instead of the blocks of metrics printed by stable baselines
        :param progress_kwargs: arguments passed to the TrainingProgressCallback
        :param **model_kwargs: other arguments passed to the algorithm, e.g. the
        learning rate
        '''
        if isinstance(method, str):
            if method.lower() not in METHODS:
                raise ValueError(f'Method <{method}> is not one of {sorted(METHODS)}.')
            method = METHODS[method.lower()]
        self.env = env
        self.progress = progress
        self.progress_kwargs = progress_kwargs or {}
        self.model = method(policy, env, **model_kwargs)

    def train(self, total_timesteps: int, callback=None, **learn_kwargs) -> BaseAlgorithm:
        '''Trains the agent and returns the trained stable baselines model.

        :param total_timesteps: the number of environment steps to train for
        :param callback: callback or list of callbacks called during training,
        in addition to the progress display
        :param **learn_kwargs: other arguments passed to the algorithm's learn()
        '''
        if self.progress:
            if callback is None:
                callback = []
            elif not isinstance(callback, list):
                callback = [callback]
            callback = callback + [TrainingProgressCallback(**self.progress_kwargs)]
        return self.model.learn(total_timesteps=total_timesteps, callback=callback,
                                **learn_kwargs)

    def agent(self, deterministic: bool=True) -> StableBaselinesAgent:
        '''Returns the trained agent as a RDDL policy, e.g. for evaluation.

        :param deterministic: whether to choose actions deterministically
        '''
        return StableBaselinesAgent(self.model, deterministic=deterministic)
