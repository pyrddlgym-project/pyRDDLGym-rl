import atexit
import numpy as np
import sys
from typing import Any, Dict, Iterable, Tuple

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter

try:
    from rich.console import Group
    from rich.live import Live
    from rich.padding import Padding
    from rich.progress import (
        BarColumn,
        Progress,
        ProgressColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn
    )
    from rich.table import Table
    from rich.text import Text
except ImportError as error:
    raise ImportError(
        'TrainingProgressCallback requires the rich package: '
        'it can be installed with \'pip install rich\'.') from error


class _LatestValuesWriter(KVWriter):
    '''A stable baselines logger output that keeps the most recently logged values,
    i.e. the values that would otherwise be printed in a block to stdout.'''

    def __init__(self) -> None:
        self.values = {}

    def write(self, key_values: Dict[str, Any],
              key_excluded: Dict[str, Tuple[str, ...]], step: int=0) -> None:
        for (key, value) in key_values.items():
            excluded = key_excluded.get(key, None)
            if excluded is not None and ('stdout' in excluded or 'log' in excluded):
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (int, float, str)):
                self.values[key] = value

    def close(self) -> None:
        pass


class _SpeedColumn(ProgressColumn):
    '''Renders the number of environment steps per second.'''

    def render(self, task):
        speed = task.finished_speed or task.speed
        text = f'{speed:,.0f} steps/s' if speed else '- steps/s'
        return Text(text, style='progress.data.speed')


class TrainingProgressCallback(BaseCallback):
    '''A stable baselines callback that replaces the blocks of training metrics
    printed to the console by a progress bar, followed by the same metrics in a
    compact table that is updated in place.'''

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
        self.hidden_keys = set(hidden_keys)
        self._live = None

    def _on_training_start(self) -> None:

        # capture the logged metrics, and stop them from being printed to stdout
        logger = self.model.logger
        self._writer = _LatestValuesWriter()
        self._stdout_formats = [
            output for output in logger.output_formats
            if isinstance(output, HumanOutputFormat) and output.file is sys.stdout]
        logger.output_formats = [
            output for output in logger.output_formats
            if output not in self._stdout_formats] + [self._writer]

        # remove timesteps that were done in previous training sessions
        self._start = self.model.num_timesteps
        total = self.locals['total_timesteps'] - self._start
        self._progress = Progress(
            TextColumn('[bold]{task.description}'),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn('{task.completed:,.0f}/{task.total:,.0f} steps'),
            _SpeedColumn(),
            TimeElapsedColumn(),
            TextColumn('<'),
            TimeRemainingColumn())
        self._task = self._progress.add_task(type(self.model).__name__, total=total)
        self._live = Live(get_renderable=self._renderable,
                          refresh_per_second=self.refresh_per_second)
        self._console = self._live.console
        self._live.start()
        atexit.register(self._stop)

    def _on_step(self) -> bool:
        self._progress.update(self._task, completed=self.model.num_timesteps - self._start)
        return True

    def _on_training_end(self) -> None:
        self._progress.update(self._task, completed=self.model.num_timesteps - self._start)
        self._stop()

        # restore the original logger outputs
        logger = self.model.logger
        logger.output_formats = [
            output for output in logger.output_formats
            if output is not self._writer] + self._stdout_formats

    def _stop(self) -> None:
        if self._live is not None:
            self._live.stop()

            # rich only ends the output with a new line when writing to a terminal
            if not self._console.is_terminal:
                self._console.line()
            self._live = None
            atexit.unregister(self._stop)

    @staticmethod
    def _format_value(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, int):
            return f'{value:,}'
        elif abs(value) >= 1000:
            return f'{value:,.1f}'
        else:
            return f'{value:.4g}'

    def _metrics_table(self):

        # group metrics by their section (e.g. rollout, time, train) as in the blocks
        sections = {}
        for (key, value) in sorted(self._writer.values.items()):
            if key not in self.hidden_keys:
                section, _, name = key.rpartition('/')
                sections.setdefault(section, []).append((name, value))
        if not sections:
            return Padding(Text('waiting for the first metrics...', style='dim'),
                           (0, 0, 0, 2))

        # fit as many (name, value) pairs per row as the console width allows
        items = [item for section_items in sections.values() for item in section_items]
        name_width = max(len(name) for (name, _) in items)
        value_width = max(len(self._format_value(value)) for (_, value) in items)
        free_width = self._console.width - 2 - max(map(len, sections)) - 2
        columns = max(1, min(self.columns, free_width // (name_width + value_width + 4)))

        # each row is a section name, then several (name, value) pairs
        table = Table.grid(padding=(0, 2))
        table.add_column(style='bold', no_wrap=True)
        for _ in range(columns):
            table.add_column(style='dim', no_wrap=True)
            table.add_column(justify='right', no_wrap=True)
        for (section, items) in sections.items():
            for start in range(0, len(items), columns):
                row = [section if start == 0 else '']
                for (name, value) in items[start:start + columns]:
                    row.extend([name, self._format_value(value)])
                table.add_row(*row)
        return Padding(table, (0, 0, 0, 2))

    def _renderable(self):
        return Group(self._progress, self._metrics_table())
