# pyRDDLGym-rl

Author: [Mike Gimelfarb](https://mike-gimelfarb.github.io)

This repository provides wrappers for deep reinforcement learning algorithms (i.e. Stable Baselines 3 and RLlib) to work with pyRDDLGym.

> [!NOTE]  
> If your environment has differentiable dynamics and a differentiable reward, try the gradient-based [JAX planner](https://github.com/pyrddlgym-project/pyRDDLGym-jax).

## Contents

- [Installation](#installation)
- [Running the Basic Examples](#running-the-basic-examples)
  - [Stable Baselines 3](#stable-baselines-3)
  - [RLlib](#rllib)
- [Creating an Environment](#creating-an-environment)

## Installation

To run the basic examples you will need ``pyRDDLGym>=2.0``, ``rddlrepository>=2.0`` and one of the supported reinforcement learning frameworks:
- ``stable-baselines3>=2.2.1``
- ``ray[rllib]>=2.9.2``

You can install this package, together with all of its requirements via pip:

```shell
pip install stable-baselines3  # need one of these two
pip install -U "ray[rllib]"
pip install rddlrepository pyRDDLGym-rl
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
- ``<method>`` is the RL algorithm to use [a2c, ddpg, dqn, ppo, sac, td3]
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
- ``<method>`` is the RL algorithm to use [dqn, ppo, sac]
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
