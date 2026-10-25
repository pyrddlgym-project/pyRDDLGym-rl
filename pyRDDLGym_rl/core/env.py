from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
import numpy as np
import termcolor
import typing
import warnings

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.debug.exception import (
    RDDLInvalidActionError,
    RDDLNotImplementedError,
    RDDLTypeError
)
from pyRDDLGym.core.env import RDDLEnv
from pyRDDLGym.core.simulator import RDDLSimulator


class SimplifiedActionRDDLEnv(RDDLEnv):
    '''A gym environment class for RDDL domains, modified to compact the
    action space to avoid the use of Dict space.'''

    ACTION_MODES = ('dict', 'box', 'discrete')

    def __init__(self, domain: str,
                 instance: str,
                 enforce_action_constraints: bool=False,
                 enforce_action_count_non_bool: bool=False,
                 debug_path: str=None,
                 log_path: str=None,
                 backend: typing.Type=RDDLSimulator,
                 backend_kwargs: typing.Optional[typing.Dict]=None,
                 action_mode: str='dict',
                 num_bins: int=11,
                 normalize_actions: bool=False,
                 print_summary: bool=True,
                 **kwargs) -> None:
        '''Creates a new gym environment from the given RDDL domain + instance.

        :param domain: the RDDL domain, or a compiled RDDLLiftedModel
        :param instance: the RDDL instance
        :param enforce_action_constraints: whether to raise an exception if the
        action constraints are violated
        :param enforce_action_count_non_bool: whether to include non-bool actions
        in check that number of nondef actions don't exceed max-nondef-actions
        (the simplified action space only enforces max-nondef-actions on bool
        actions, so setting this to True can raise exceptions during training)
        :param debug_path: absolute path to file where debug log is saved,
        excluding the file extension, None means no debugging
        :param log_path: absolute path to file where simulation log is saved,
        excluding the file extension, None means no logging
        :param backend: the subclass of RDDLSimulator to use as backend for
        simulation (currently supports numpy and Jax)
        :param backend_kwargs: dictionary of additional named arguments to
        pass to backend (must not include logger)
        :param action_mode: how actions are represented, one of the following,
        where if max-nondef-actions is less than the number of bool actions, the
        bool actions are replaced by max-nondef-actions selectors, each choosing
        one bool action to set or none, so that exactly the valid combinations
        of bool actions can be chosen:
        'dict': bool, int and enum actions are a Discrete or MultiDiscrete space
        and real actions are a Box space, combined in a Dict space only if both
        are present (supported by e.g. RLlib);
        'box': all actions are a single Box space, where each int or bool action
        is one coordinate in [-1, 1] split into equal intervals (one per value),
        each enum action (and each bool action selector) is one coordinate
        in [-1, 1] per value of which the argmax is
        selected, and real actions keep their bounds (supported by e.g. PPO, SAC, TD3);
        'discrete': all actions are a single Discrete or MultiDiscrete space,
        where each real action is discretized into num_bins evenly spaced values
        between its bounds (supported by e.g. PPO, A2C)
        :param num_bins: number of values of each real action if action_mode is 'discrete'
        :param normalize_actions: whether real actions are represented in [-1, 1]
        and rescaled to their bounds, which requires all bounds to be finite,
        and has no effect if action_mode is 'discrete': this is only useful for
        algorithms that clip rather than rescale actions (e.g. PPO, A2C in
        stable baselines), since off-policy stable baselines algorithms and
        RLlib already rescale actions
        :param print_summary: whether to print a summary of the observation and
        action spaces when the environment is created
        :param **kwargs: other arguments to pass to RDDLEnv (e.g. preprocessor)
        '''
        if not kwargs.pop('vectorized', True):
            raise ValueError(
                'SimplifiedActionRDDLEnv only supports vectorized=True.')
        if action_mode not in self.ACTION_MODES:
            raise ValueError(
                f'Action mode <{action_mode}> is not valid, '
                f'must be one of {self.ACTION_MODES}.')
        if action_mode == 'discrete' and num_bins < 2:
            raise ValueError(f'num_bins must be at least 2, got {num_bins}.')
        self.action_mode = action_mode
        self.num_bins = num_bins
        self.normalize_actions = normalize_actions

        # objects must be returned as integer indices to be valid observations
        backend_kwargs = dict(backend_kwargs or {})
        backend_kwargs.setdefault('objects_as_strings', False)

        super(SimplifiedActionRDDLEnv, self).__init__(
            domain=domain,
            instance=instance,
            enforce_action_constraints=enforce_action_constraints,
            enforce_action_count_non_bool=enforce_action_count_non_bool,
            vectorized=True,
            debug_path=debug_path,
            log_path=log_path,
            backend=backend,
            backend_kwargs=backend_kwargs,
            **kwargs)

        # the default backend RNG is shared by all simulators, so create a fresh one
        if 'rng' not in backend_kwargs and hasattr(self.sampler, 'rng'):
            self.sampler.rng = np.random.default_rng()

        # observations are returned as arrays of at least one dimension
        self._shapes = {var: np.shape(np.atleast_1d(values[0]))
                        for (var, values) in self._bounds.items()}
        if self.sampler.is_pomdp:
            state_ranges = self.model.observ_ranges
        else:
            state_ranges = self.model.state_ranges
        self.observation_space = Dict(dict(self._rddl_to_gym_bounds(state_ranges)))

        # construct the gym action space
        self.action_space = self._rddl_to_gym_bounds_act(self._action_ranges)
        if print_summary:
            print(self.summarize_spaces())

    def _rddl_to_gym_bounds_act(self, ranges):

        # compute whether or not a constraint must be placed on boolean actions
        total_bool, total_non_bool = 0, 0
        for (var, prange) in ranges.items():
            if prange == 'bool':
                total_bool += np.prod(self._shapes[var], dtype=np.int64)
            else:
                total_non_bool += np.prod(self._shapes[var], dtype=np.int64)
                
        # if max-nondef-actions constrains the bool actions, each selector chooses
        # one bool action to set or none, which allows exactly the valid choices
        bool_constraint = self.max_allowed_actions < total_bool
        bool_selectors = int(self.max_allowed_actions) if bool_constraint else 0

        # max-nondef-actions is only enforced on bool actions by the action space
        max_non_default = (bool_selectors if bool_constraint else total_bool) + total_non_bool
        if not self.enforce_count_non_bool and total_non_bool \
        and max_non_default > self.max_allowed_actions:
            warnings.warn(
                f'max-nondef-actions = {self.max_allowed_actions} is not enforced on '
                f'non-bool actions, so up to {max_non_default} actions can be '
                f'non-default in each step: learned policies may not be valid '
                f'in the original domain.', stacklevel=3)

        # collect information about action ranges
        locational = {}
        count_disc, count_cont, count_bool = 0, 0, 0
        disc_start, disc_nelem, disc_ordinal = [], [], []
        cont_low, cont_high = [], []
        for (var, prange) in ranges.items():
            num_elements = np.prod(self._shapes[var], dtype=np.int64)

            # boolean actions without constraint are stored as Discrete
            if prange == 'bool':
                if bool_constraint:
                    locational[var] = ('discrete', (count_bool, num_elements))
                    count_bool += num_elements
                else:
                    disc_start.extend([0] * num_elements)
                    disc_nelem.extend([2] * num_elements)
                    disc_ordinal.extend([True] * num_elements)
                    locational[var] = ('discrete', (count_disc, num_elements))
                    count_disc += num_elements

            # integer actions are stored as Discrete, and must be bounded
            elif prange == 'int':
                low, high = self._bounds[var]
                low = np.ravel(low, order='C')
                high = np.ravel(high, order='C')
                if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
                    raise RDDLNotImplementedError(
                        f'Int action-fluent <{var}> must have finite lower and '
                        f'upper bounds specified in action-preconditions, '
                        f'got bounds [{low}, {high}].')
                low = np.ceil(low).astype(np.int64)
                high = np.floor(high).astype(np.int64)
                if np.any(high < low):
                    raise RDDLInvalidActionError(
                        f'Int action-fluent <{var}> has an empty range, '
                        f'got bounds [{low}, {high}].')
                disc_start.extend(low.tolist())
                disc_nelem.extend((high - low + 1).tolist())
                disc_ordinal.extend([True] * num_elements)
                locational[var] = ('discrete', (count_disc, num_elements))
                count_disc += num_elements

            # enum-valued actions are stored as Discrete
            elif prange in self.model.enum_types:
                num_objects = len(self.model.type_to_objects[prange])
                disc_start.extend([0] * num_elements)
                disc_nelem.extend([num_objects] * num_elements)
                disc_ordinal.extend([False] * num_elements)
                locational[var] = ('discrete', (count_disc, num_elements))
                count_disc += num_elements

            # real actions are stored as Box
            elif prange == 'real':
                low, high = self._bounds[var]
                low = np.ravel(low, order='C').tolist()
                high = np.ravel(high, order='C').tolist()
                cont_low.extend(low)
                cont_high.extend(high)
                locational[var] = ('continuous', (count_cont, num_elements))
                count_cont += num_elements

            # not a valid action type
            else:
                raise RDDLTypeError(
                    f'Type <{prange}> of fluent <{var}> is not valid, '
                    f'must be an enumerated or primitive type (real, int, bool).')

        # boolean actions with constraint are stored as selectors in the last place
        if bool_constraint:
            disc_start.extend([0] * bool_selectors)
            disc_nelem.extend([count_bool + 1] * bool_selectors)
            disc_ordinal.extend([False] * bool_selectors)
            count_disc += bool_selectors
        if not count_disc and not count_cont:
            raise RDDLInvalidActionError(
                'RDDL action specification resulted in an empty action space.')

        # continuous bounds are kept in float64 to clip actions exactly
        self._locational = locational
        self._bool_constraint = bool_constraint
        self._bool_selectors = bool_selectors
        self._disc_start = np.asarray(disc_start, dtype=np.int64)
        self._disc_nelem = np.asarray(disc_nelem, dtype=np.int64)
        self._cont_low = np.asarray(cont_low, dtype=np.float64)
        self._cont_high = np.asarray(cont_high, dtype=np.float64)
        if self.normalize_actions and self.action_mode != 'discrete':
            unbounded = self._unbounded_real_actions()
            if unbounded:
                raise RDDLNotImplementedError(
                    f'Real action-fluents {unbounded} must have finite lower and '
                    f'upper bounds specified in action-preconditions for '
                    f'normalize_actions=True.')

        # build the space for the requested action mode
        if self.action_mode == 'box':
            space = self._box_action_space(np.asarray(disc_ordinal, dtype=bool))
        elif self.action_mode == 'discrete':
            space = self._discrete_action_space()
        else:
            space = self._dict_action_space()

        # log information
        if self.logger is not None:
            act_info = '\n\t'.join(
                f'{act}: action_tensor={key}, start={start}, count={count}'
                for (act, (key, (start, count))) in locational.items()
            )
            bound_info = (f'\tdiscrete_start={disc_start}, '
                          f'discrete_n={disc_nelem}\n'
                          f'\tcontinuous_low={cont_low}, '
                          f'continuous_high={cont_high}')
            self.logger.log(f'[info] computed gym action space:\n'
                            f'{bound_info}\n'
                            f'\t{act_info}\n'
                            f'[info] final space ({self.action_mode}): {space}\n')
        return space

    @staticmethod
    def _discrete_space(nelem):
        if len(nelem) == 1:
            return Discrete(int(nelem[0]))
        else:
            return MultiDiscrete(nelem)

    def _cont_space(self, low, high):
        return Box(np.asarray(low, dtype=np.float32),
                   np.asarray(high, dtype=np.float32), dtype=np.float32)

    def _unbounded_real_actions(self):
        return [var for (var, (key, (start, count))) in self._locational.items()
                if key == 'continuous' and not (
                    np.all(np.isfinite(self._cont_low[start:start + count])) and
                    np.all(np.isfinite(self._cont_high[start:start + count])))]

    def _agent_cont_bounds(self):
        if self.normalize_actions:
            return -np.ones(self._cont_low.size), np.ones(self._cont_high.size)
        else:
            return self._cont_low, self._cont_high

    def _denormalize_cont_action(self, cont):
        if not self.normalize_actions or cont is None:
            return cont
        fraction = (np.clip(np.atleast_1d(cont), -1., 1.) + 1.) / 2.
        return self._cont_low + fraction * (self._cont_high - self._cont_low)

    def _dict_action_space(self):
        spaces = {}
        if self._disc_nelem.size:
            spaces['discrete'] = self._discrete_space(self._disc_nelem)
        if self._cont_low.size:
            spaces['continuous'] = self._cont_space(*self._agent_cont_bounds())
        space = Dict(spaces)
        self._action_keys = list(space.keys())
        if len(self._action_keys) == 1:
            space = space[self._action_keys[0]]
        return space

    def _box_action_space(self, disc_ordinal):

        # ordinal actions are one coordinate split into equal intervals, categorical
        # actions are one coordinate per value of which the argmax is selected
        widths = np.where(disc_ordinal, 1, self._disc_nelem)
        offsets = np.cumsum(widths) - widths
        self._box_widths, self._box_offsets = widths, offsets
        self._box_ordinal = (np.flatnonzero(disc_ordinal), offsets[disc_ordinal])
        self._box_categorical = [
            (i, offsets[i], widths[i]) for i in np.flatnonzero(~disc_ordinal)]
        self._box_cont_start = int(np.sum(widths))

        cont_low, cont_high = self._agent_cont_bounds()
        low = np.concatenate([-np.ones(self._box_cont_start), cont_low])
        high = np.concatenate([np.ones(self._box_cont_start), cont_high])
        return self._cont_space(low, high)

    def _discrete_action_space(self):
        unbounded = self._unbounded_real_actions()
        if unbounded:
            raise RDDLNotImplementedError(
                f'Real action-fluents {unbounded} must have finite lower and upper '
                f'bounds specified in action-preconditions for action_mode=discrete.')
        bins = np.full(self._cont_low.size, self.num_bins, dtype=np.int64)
        return self._discrete_space(np.concatenate([self._disc_nelem, bins]))

    def _decode_box_action(self, action):
        action = np.ravel(np.asarray(action, dtype=np.float64))
        disc = np.zeros(self._disc_nelem.size, dtype=np.int64)
        index, offset = self._box_ordinal
        nelem = self._disc_nelem[index]
        fraction = (np.clip(action[offset], -1., 1.) + 1.) / 2.
        disc[index] = np.minimum(np.floor(fraction * nelem), nelem - 1)
        for (i, start, width) in self._box_categorical:
            disc[i] = np.argmax(action[start:start + width])
        return disc, self._denormalize_cont_action(action[self._box_cont_start:])

    def _decode_discrete_action(self, action):
        action = np.atleast_1d(action).astype(np.int64)
        num_disc = self._disc_nelem.size
        fraction = action[num_disc:] / (self.num_bins - 1)
        cont = self._cont_low + fraction * (self._cont_high - self._cont_low)
        return action[:num_disc], cont

    def _decode_dict_action(self, action):
        if not isinstance(action, dict):
            action = {self._action_keys[0]: action}
        cont = self._denormalize_cont_action(action.get('continuous', None))
        return action.get('discrete', None), cont

    def _gym_to_rddl_actions(self, gym_actions):

        # actions of non-Dict spaces can also be passed as {'action': action}
        if isinstance(gym_actions, dict) and 'action' in gym_actions \
        and not isinstance(self.action_space, Dict):
            gym_actions = gym_actions['action']

        # decode into discrete indices and continuous values
        if self.action_mode == 'box':
            disc, cont = self._decode_box_action(gym_actions)
        elif self.action_mode == 'discrete':
            disc, cont = self._decode_discrete_action(gym_actions)
        else:
            disc, cont = self._decode_dict_action(gym_actions)
        noop_actions = self.sampler.noop_actions

        # process all actions except if active max-nondef-actions constraint
        actions = {}
        for (var, prange) in self._action_ranges.items():
            if self._bool_constraint and prange == 'bool':
                continue
            key, (start, count) = self._locational[var]
            if key == 'discrete':
                if disc is None:
                    continue
                action = np.atleast_1d(disc)[start:start + count]
                action = action + self._disc_start[start:start + count]
            else:
                if cont is None:
                    continue
                action = np.atleast_1d(cont)[start:start + count]
                action = np.clip(action, self._cont_low[start:start + count],
                                 self._cont_high[start:start + count])
            dtype = RDDLValueInitializer.NUMPY_TYPES.get(
                prange, RDDLValueInitializer.INT)
            correct_shape = np.shape(noop_actions[var])
            actions[var] = np.reshape(
                action, correct_shape, order='C').astype(dtype)

        # process the active max-nondef-actions constraint: set the bool actions
        # chosen by any selector, where choosing the same action twice sets it once
        if self._bool_constraint and self._bool_selectors and disc is not None:
            indices = np.atleast_1d(disc)[-self._bool_selectors:]
            for (var, prange) in self._action_ranges.items():
                if prange == 'bool':
                    _, (start, count) = self._locational[var]
                    chosen = indices[(indices >= start) & (indices < start + count)]
                    if chosen.size:
                        default_value = self.model.variable_defaults[var]
                        action = np.full(shape=count, fill_value=default_value, dtype=bool)
                        action[chosen - start] = not default_value
                        correct_shape = np.shape(noop_actions[var])
                        actions[var] = np.reshape(action, correct_shape, order='C')

        return actions

    @staticmethod
    def _format_number(value):
        if np.issubdtype(np.asarray(value).dtype, np.integer):
            return str(int(value))
        return repr(float(value))

    @staticmethod
    def _format_range(low, high):
        low, high = np.ravel(low), np.ravel(high)
        fmt = SimplifiedActionRDDLEnv._format_number
        if not low.size:
            return '[]'
        text = f'[{fmt(np.min(low))}, {fmt(np.max(high))}]'
        if np.any(low != low[0]) or np.any(high != high[0]):
            text += ' (varies)'
        return text

    @staticmethod
    def _format_array(values, max_items=6):
        values = [str(value) for value in np.ravel(values)]
        if len(values) > max_items:
            values = values[:2] + ['...'] + values[-2:]
        return f'[{", ".join(values)}]'

    @staticmethod
    def _format_space(space):
        fmt = SimplifiedActionRDDLEnv._format_array
        if isinstance(space, Box):
            low, high = space.low, space.high
            if np.all(low == low.flat[0]) and np.all(high == high.flat[0]):
                bounds = f'{low.flat[0]}, {high.flat[0]}'
            else:
                bounds = f'{fmt(low)}, {fmt(high)}'
            return f'Box({bounds}, {space.shape}, {space.dtype})'
        elif isinstance(space, Discrete):
            return f'Discrete({space.n})'
        elif isinstance(space, MultiDiscrete):
            return f'MultiDiscrete({fmt(space.nvec)})'
        elif isinstance(space, Dict):
            keys = 'key' if len(space) == 1 else 'keys'
            return f'Dict with {len(space)} {keys}'
        else:
            return str(space)

    @staticmethod
    def _format_rows(rows, indent):
        if not rows:
            return ''
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
        result = ''
        for row in rows:
            cells = [cell.ljust(width) for (cell, width) in zip(row[:-1], widths)]
            description = termcolor.colored(row[-1], 'dark_grey')
            result += ' ' * indent + '  '.join(cells + [description]) + '\n'
        return result

    def _action_location(self, var):
        key, (start, count) = self._locational[var]
        if not count:
            return key, 0, 0

        # bool actions with constraint share the last discrete actions (selectors),
        # and in a box space each selector is a group of coordinates of an argmax
        if self._bool_constraint and self._action_ranges[var] == 'bool':
            first = self._disc_nelem.size - self._bool_selectors
            last = self._disc_nelem.size
            if self.action_mode == 'box':
                first = self._box_offsets[first]
                last = self._box_cont_start
            return key, int(first), int(last)

        # map to the coordinates of the action space for the current action mode
        if self.action_mode == 'box':
            if key == 'discrete':
                first = self._box_offsets[start]
                last = self._box_offsets[start + count - 1] + \
                    self._box_widths[start + count - 1]
            else:
                first = self._box_cont_start + start
                last = first + count
        elif self.action_mode == 'discrete' and key == 'continuous':
            first = self._disc_nelem.size + start
            last = first + count
        else:
            first, last = start, start + count
        return key, int(first), int(last)

    def _action_description(self, var, prange):
        _, (start, count) = self._locational[var]
        box = self.action_mode == 'box'
        if prange == 'bool' and self._bool_constraint:
            none = self._disc_nelem[-1] - 1
            values = f'{start}' if count == 1 else f'{start}-{start + count - 1}'
            if box:
                description = (f'bool, values {values} of argmax over each '
                               f'{none + 1} coordinates, {none} is none')
            else:
                description = f'bool, values {values} of {none + 1}, {none} is none'
            description += f' (max-nondef-actions={self._bool_selectors})'
        elif prange == 'bool':
            description = 'bool'
            if box:
                description += ', true if > 0'
        elif prange == 'int':
            low = self._disc_start[start:start + count]
            high = low + self._disc_nelem[start:start + count] - 1
            description = f'int in {self._format_range(low, high)}'
            if box:
                description += ', equal intervals'
        elif prange in self.model.enum_types:
            objects = self.model.type_to_objects[prange]
            if len(objects) > 6:
                objects = objects[:3] + ['...'] + objects[-2:]
            description = f'{prange} in {{{", ".join(objects)}}}'
            if box:
                description += ', argmax'
        else:
            description = 'real in ' + self._format_range(
                self._cont_low[start:start + count], self._cont_high[start:start + count])
            if self.action_mode == 'discrete':
                description += f', {self.num_bins} bins'
            elif self.normalize_actions:
                description += ', rescaled from [-1, 1]'
        return f'[{description}]'

    def _action_rows(self, key, space):
        rows = []
        for (var, prange) in self._action_ranges.items():
            var_key, first, last = self._action_location(var)
            if key is not None and var_key != key:
                continue
            if isinstance(space, Discrete) or last <= first:
                index = '-'
            elif last - first == 1:
                index = f'[{first}]'
            else:
                index = f'[{first}:{last}]'
            params = self.model.variable_params[var]
            label = f'{var}({", ".join(params)})' if params else var
            rows.append((first, (label, index, self._action_description(var, prange))))
        return [row for (_, row) in sorted(rows, key=lambda row: row[0])]

    def summarize_spaces(self) -> str:
        '''Returns a summary of the environment settings and the observation and
        action spaces, showing how each RDDL fluent maps to each space.
        '''
        max_nondef = self.max_allowed_actions
        if not np.isfinite(max_nondef):
            max_nondef = 'pos-inf'
        result = (f'[INFO] environment settings:\n'
                  f'    domain                       ={self.model.domain_name}\n'
                  f'    instance                     ={self.model.instance_name}\n'
                  f'    horizon                      ={self.horizon}\n'
                  f'    discount                     ={self.discount}\n'
                  f'    max_nondef_actions           ={max_nondef}\n'
                  f'    action_mode                  ={self.action_mode}\n')
        if self.action_mode == 'discrete':
            result += f'    num_bins                     ={self.num_bins}\n'
        else:
            result += f'    normalize_actions            ={self.normalize_actions}\n'
        result += (f'    enforce_action_constraints   ={self.enforce_action_constraints}\n'
                   f'    enforce_action_count_non_bool={self.enforce_count_non_bool}\n')

        # observation space: one entry per state or observ-fluent
        fluent_type = 'observ-fluents' if self.sampler.is_pomdp else 'state-fluents'
        if self.sampler.is_pomdp:
            ranges = self.model.observ_ranges
        else:
            ranges = self.model.state_ranges
        rows = []
        for (var, space) in self.observation_space.items():
            params = self.model.variable_params[var]
            description = f'[{ranges[var]}, over ({", ".join(params)})]' \
                if params else f'[{ranges[var]}]'
            rows.append((f'{var}:', self._format_space(space), description))
        result += (f'[INFO] observation space ({fluent_type}):\n'
                   f'    {self._format_space(self.observation_space)}\n')
        result += self._format_rows(rows, indent=8)

        # action space: one entry per action-fluent, nested in Dict sub-spaces
        result += (f'[INFO] action space (action_mode={self.action_mode}):\n'
                   f'    {self._format_space(self.action_space)}\n')
        if isinstance(self.action_space, Dict):
            for (key, space) in self.action_space.items():
                result += f'        {key}: {self._format_space(space)}\n'
                result += self._format_rows(self._action_rows(key, space), indent=12)
        else:
            result += self._format_rows(self._action_rows(None, self.action_space), indent=8)
        return result

    def _gym_observation(self, obs):
        '''Casts observations to the dtypes of the observation space, and
        replaces missing observations (i.e. at reset of a POMDP) by defaults.'''
        result = {}
        for (var, space) in self.observation_space.items():
            value = obs.get(var, None)
            if value is None or np.asarray(value).dtype == object:
                value = self.sampler.subs[var]

            # saturate values outside the range of the dtype instead of overflow
            if np.issubdtype(space.dtype, np.integer):
                limits = np.iinfo(space.dtype)
            else:
                limits = np.finfo(space.dtype)
            value = np.clip(value, limits.min, limits.max)
            result[var] = np.reshape(value, space.shape).astype(space.dtype)
        return result

    def reset(self, seed=None, options=None):
        obs, info = super(SimplifiedActionRDDLEnv, self).reset(seed=seed, options=options)
        return self._gym_observation(obs), info

    def step(self, actions):
        actions = self._gym_to_rddl_actions(actions)
        obs, reward, terminated, truncated, info = \
            super(SimplifiedActionRDDLEnv, self).step(actions)
        return self._gym_observation(obs), reward, terminated, truncated, info
