from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
import numpy as np
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

    def __init__(self, domain: str,
                 instance: str,
                 enforce_action_constraints: bool=False,
                 enforce_action_count_non_bool: bool=False,
                 debug_path: str=None,
                 log_path: str=None,
                 backend: typing.Type=RDDLSimulator,
                 backend_kwargs: typing.Optional[typing.Dict]=None,
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
        :param **kwargs: other arguments to pass to RDDLEnv (e.g. preprocessor)
        '''
        if not kwargs.pop('vectorized', True):
            raise ValueError(
                'SimplifiedActionRDDLEnv only supports vectorized=True.')

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
        self.observation_space = self._rddl_to_gym_bounds(state_ranges)

        # construct the gym action space
        self.action_space, self._action_info = \
            self._rddl_to_gym_bounds_act(self._action_ranges)

    def _rddl_to_gym_bounds_act(self, ranges):

        # compute whether or not a constraint must be placed on boolean actions
        total_bool, total_non_bool = 0, 0
        for (var, prange) in ranges.items():
            if prange == 'bool':
                total_bool += np.prod(self._shapes[var], dtype=np.int64)
            else:
                total_non_bool += np.prod(self._shapes[var], dtype=np.int64)
        if self.max_allowed_actions < total_bool:
            if self.max_allowed_actions != 1:
                raise RDDLNotImplementedError(
                    'Simplification of bool action space with max-nondef-actions '
                    'other than 1 or pos-inf is not currently supported.')
            bool_constraint = True
        else:
            bool_constraint = False

        # max-nondef-actions is only enforced on bool actions by the action space
        max_non_default = (1 if bool_constraint else total_bool) + total_non_bool
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
        disc_start, disc_nelem = [], []
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
                locational[var] = ('discrete', (count_disc, num_elements))
                count_disc += num_elements

            # enum-valued actions are stored as Discrete
            elif prange in self.model.enum_types:
                num_objects = len(self.model.type_to_objects[prange])
                disc_start.extend([0] * num_elements)
                disc_nelem.extend([num_objects] * num_elements)
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

        # boolean actions with constraint are stored in the last place instead
        if bool_constraint:
            disc_start.append(0)
            disc_nelem.append(count_bool + 1)
            count_disc += 1

        # discrete space
        if len(disc_nelem) == 1:
            disc_space = Discrete(disc_nelem[0])
        elif len(disc_nelem) > 1:
            disc_space = MultiDiscrete(disc_nelem)
        else:
            disc_space = None

        # real space
        if count_cont:
            cont_space = Box(np.asarray(cont_low), np.asarray(cont_high),
                             dtype=np.float32)
        else:
            cont_space = None

        # simplify space
        combined_space = Dict()
        if disc_space is not None:
            combined_space['discrete'] = disc_space
        if cont_space is not None:
            combined_space['continuous'] = cont_space
        if not combined_space:
            raise RDDLInvalidActionError(
                'RDDL action specification resulted in an empty action space.')
        keys = list(combined_space.keys())
        if len(keys) == 1:
            combined_space = combined_space[keys[0]]

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
                            f'[info] final space: {combined_space}\n')

        # continuous bounds are kept in float64 to clip actions exactly
        return combined_space, \
            (locational, keys, bool_constraint, np.asarray(disc_start, dtype=np.int64),
             np.asarray(cont_low, dtype=np.float64),
             np.asarray(cont_high, dtype=np.float64))

    def _gym_to_rddl_actions(self, gym_actions):
        locational, keys, bool_constraint, disc_start, cont_low, cont_high = \
            self._action_info
        if len(keys) == 1 and not (
            isinstance(gym_actions, dict) and keys[0] in gym_actions):
            if isinstance(gym_actions, dict):
                gym_actions = gym_actions['action']
            gym_actions = {keys[0]: gym_actions}
        noop_actions = self.sampler.noop_actions

        # process all actions except if active max-nondef-actions constraint
        actions = {}
        for (var, prange) in self._action_ranges.items():
            if not (bool_constraint and prange == 'bool'):
                key, (start, count) = locational[var]
                action_key = gym_actions.get(key, None)
                if action_key is not None:
                    action = np.atleast_1d(action_key)[start:start + count]
                    if key == 'discrete':
                        action = action + disc_start[start:start + count]
                    else:
                        action = np.clip(action, cont_low[start:start + count],
                                         cont_high[start:start + count])
                    dtype = RDDLValueInitializer.NUMPY_TYPES.get(
                        prange, RDDLValueInitializer.INT)
                    correct_shape = np.shape(noop_actions[var])
                    actions[var] = np.reshape(
                        action, correct_shape, order='C').astype(dtype)

        # process the active max-nondef-actions constraint
        action_key = gym_actions.get('discrete', None)
        if bool_constraint and action_key is not None:
            index = np.atleast_1d(action_key)[-1]
            for (var, prange) in self._action_ranges.items():
                if prange == 'bool':
                    _, (start, count) = locational[var]
                    index_in_var = index - start
                    if 0 <= index_in_var < count:
                        default_value = self.model.variable_defaults[var]
                        action = np.full(shape=count, fill_value=default_value, dtype=bool)
                        action[index_in_var] ^= True
                        correct_shape = np.shape(noop_actions[var])
                        actions[var] = np.reshape(action, correct_shape, order='C')
                        break

        return actions

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
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        obs, info = super(SimplifiedActionRDDLEnv, self).reset(seed=seed, options=options)
        return self._gym_observation(obs), info

    def step(self, actions):
        actions = self._gym_to_rddl_actions(actions)
        obs, reward, terminated, truncated, info = \
            super(SimplifiedActionRDDLEnv, self).step(actions)
        return self._gym_observation(obs), reward, terminated, truncated, info
