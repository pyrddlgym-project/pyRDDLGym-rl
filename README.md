<div align="center">

# pyRDDLGym-rl

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
[![PyPI Version](https://img.shields.io/pypi/v/pyRDDLGym-rl.svg)](https://pypi.org/project/pyRDDLGym-rl/)
[![Documentation Status](https://readthedocs.org/projects/pyrddlgym/badge/?version=latest)](https://pyrddlgym.readthedocs.io/en/latest/sb.html)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

**Train deep reinforcement learning agents from Stable Baselines 3 and RLlib on any RDDL problem.**

[Why pyRDDLGym-rl?](#why-pyrddlgym-rl) &nbsp;&middot;&nbsp; [Installation](#installation) &nbsp;&middot;&nbsp; [Examples](#running-the-basic-examples) &nbsp;&middot;&nbsp; [Creating an environment](#creating-an-environment)

</div>

## Why pyRDDLGym-rl?

- 🔌 **Any problem, any algorithm.** Every problem in [rddlrepository](https://github.com/pyrddlgym-project/rddlrepository), or your own RDDL files, trains with any algorithm in [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) (and sb3-contrib) or [RLlib](https://docs.ray.io/en/latest/rllib/index.html).
- 🎛️ **Action spaces that fit.** RDDL actions become a dictionary, a single box (for PPO, SAC or TD3) or a discrete space (for DQN), and ``auto`` avoids the mixed spaces most algorithms cannot handle.
- 🛡️ **Constraints handled.** An action that breaks the action preconditions or the limit on concurrent actions is replaced by a no-op instead of crashing training, and flagged in ``info["action_valid"]``.
- 🔍 **Nothing hidden.** A summary shows how every state and action fluent maps to the observation and action spaces.
- 📊 **Watch it train.** A live display in the terminal tracks progress and returns as the agent learns.
- 🎬 **Back to pyRDDLGym.** A trained agent is a pyRDDLGym policy, ready to evaluate, visualize and record like any other.

> [!NOTE]
> If your environment has differentiable dynamics and a differentiable reward, try the gradient-based [JAX planner](https://github.com/pyrddlgym-project/pyRDDLGym-jax).

## Installation

To run the basic examples you will need ``pyRDDLGym>=2.1``, ``rddlrepository>=2.0`` and one of the supported reinforcement learning frameworks:
- ``stable-baselines3>=2.2.1``
- ``ray[rllib]>=2.9.2``

You can install this package, together with all of its requirements via pip:

```shell
pip install stable-baselines3  # need one of these two
pip install -U "ray[rllib]"
pip install rddlrepository pyRDDLGym-rl
```

or equivalently, with the reinforcement learning framework installed as an extra:

```shell
pip install rddlrepository "pyRDDLGym-rl[sb3]"    # or "pyRDDLGym-rl[rllib]"
```

## Running the Basic Examples

### Stable Baselines 3

To run the stable-baselines3 example, navigate to the install directory of pyRDDLGym-rl, and type:

```shell
python -m pyRDDLGym_rl.examples.run_stable_baselines <domain> <instance> <method> <steps> <learning_rate>
```

where:
- ``<domain>`` is the name of the domain in rddlrepository, or a path pointing to a ``domain.rddl`` file
- ``<instance>`` is the name of the instance in rddlrepository, or a path pointing to an ``instance.rddl`` file
- ``<method>`` is the RL algorithm to use
- ``<steps>`` is the (optional) number of samples to generate from the environment for training, and
- ``<learning_rate>`` is the (optional) learning rate to specify for the algorithm.

### RLLib

To run the RLlib example, from the install directory of pyRDDLGym-rl, type:

```shell
python -m pyRDDLGym_rl.examples.run_rllib <domain> <instance> <method> <iters>
```

where:
- ``<domain>`` is the name of the domain in rddlrepository, or a path pointing to a ``domain.rddl`` file
- ``<instance>`` is the name of the instance in rddlrepository, or a path pointing to an ``instance.rddl`` file
- ``<method>`` is the RL algorithm to use
- ``<iters>`` is the (optional) number of iterations of training

## Creating an Environment

You can create an environment wrapper to use with your own RL implementations, or a package that is not currently supported by us:

```python
import pyRDDLGym
from pyRDDLGym_rl.core.env import SimplifiedActionRDDLEnv
env = pyRDDLGym.make("domain name", "instance name", base_class=SimplifiedActionRDDLEnv)
```

This creates an instance of ``gymnasium.Env`` in which the action space is simplified by concatenating all continuous, discrete and boolean action fluents into single tensors.
You can then use this environment as you would in [pyRDDLGym](https://github.com/pyrddlgym-project/pyRDDLGym).
