import atexit
from typing import Any, Dict, Optional

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
        'The training progress display requires the rich package: '
        'it can be installed with \'pip install rich\'.') from error


class _SpeedColumn(ProgressColumn):
    '''Renders the speed of training, e.g. steps per second or seconds per iteration.'''

    def __init__(self, unit: str) -> None:
        super(_SpeedColumn, self).__init__()
        self.unit = unit

    def render(self, task):
        speed = task.finished_speed or task.speed
        if not speed:
            text = f'- {self.unit}s/s'
        elif speed >= 1:
            text = f'{speed:,.0f} {self.unit}s/s'
        else:
            text = f'{1 / speed:,.1f} s/{self.unit}'
        return Text(text, style='progress.data.speed')


class TrainingProgress:
    '''A console display with a progress bar, followed by a table of the latest
    training metrics grouped by section, which are both updated in place.'''

    def __init__(self, description: str, total: float,
                 unit: str='step',
                 columns: int=3,
                 refresh_per_second: float=3) -> None:
        '''Creates a new training progress display.

        :param description: the text shown before the progress bar
        :param total: the total amount of training (e.g. steps or iterations)
        :param unit: the unit of training progress (e.g. step or iter)
        :param columns: maximum number of metrics to show per row of the table,
        which is reduced if the rows do not fit in the console
        :param refresh_per_second: how many times per second to redraw the display
        '''
        self.columns = columns
        self.refresh_per_second = refresh_per_second
        self.metrics = {}
        self._progress = Progress(
            TextColumn('[bold]{task.description}'),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn(f'{{task.completed:,.0f}}/{{task.total:,.0f}} {unit}s'),
            _SpeedColumn(unit),
            TimeElapsedColumn(),
            TextColumn('<'),
            TimeRemainingColumn())
        self._task = self._progress.add_task(description, total=total)
        self._live = None

    def __enter__(self) -> 'TrainingProgress':
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def start(self) -> None:
        '''Starts drawing the display in the console.'''
        self._live = Live(get_renderable=self._renderable,
                          refresh_per_second=self.refresh_per_second)
        self._console = self._live.console
        self._live.start()
        atexit.register(self.stop)

    def update(self, completed: float, metrics: Optional[Dict[str, Any]]=None) -> None:
        '''Updates the progress, and optionally replaces the metrics shown.

        :param completed: the amount of training completed
        :param metrics: dictionary of metrics, keyed by 'section/name'
        '''
        self._progress.update(self._task, completed=completed)
        if metrics is not None:
            self.metrics = metrics

    def stop(self) -> None:
        '''Stops drawing the display, leaving its last state in the console.'''
        if self._live is not None:
            self._live.stop()

            # rich only ends the output with a new line when writing to a terminal
            if not self._console.is_terminal:
                self._console.line()
            self._live = None
            atexit.unregister(self.stop)

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

        # group metrics by their section, which is everything before the last /
        sections = {}
        for (key, value) in sorted(self.metrics.items()):
            section, _, name = key.rpartition('/')
            sections.setdefault(section, []).append((name, value))
        if not sections:
            return Padding(Text('waiting for the first metrics...', style='dim'), (0, 0, 0, 2))

        # each section is a header, followed by its metrics indented below it
        renderables = []
        for (section, items) in sections.items():
            renderables.append(Text(section, style='bold'))
            renderables.append(Padding(self._section_table(items), (0, 0, 0, 2)))
        return Padding(Group(*renderables), (0, 0, 0, 2))

    def _section_table(self, items):
        cells = [(name, self._format_value(value)) for (name, value) in items]

        # use as many (name, value) pairs per row as fit in the console width
        for columns in range(min(self.columns, len(cells)), 0, -1):
            rows = [cells[start:start + columns] for start in range(0, len(cells), columns)]
            widths = [max(len(row[column][i]) for row in rows if column < len(row))
                      for column in range(columns) for i in (0, 1)]
            if 4 + sum(widths) + 2 * (len(widths) - 1) <= self._console.width:
                break

        table = Table.grid(padding=(0, 2))
        for _ in range(columns):
            table.add_column(style='dim', no_wrap=True)
            table.add_column(justify='right', no_wrap=True)
        for row in rows:
            table.add_row(*[text for cell in row for text in cell])
        return table

    def _renderable(self):
        return Group(self._progress, self._metrics_table())
