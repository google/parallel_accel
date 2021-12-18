# Copyright 2021 The ParallelAccel Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# Lint as: python3
"""ASIC Simulator prototype for LinearAlgebra."""
import jax
import numpy as np
from asic_la.sharded_probability_function import (
    ShardedDiscretedProbabilityFunction,
    permute,
    invert_permutation,
)
from asic_la.sharded_probability_function import complex_workaround as cw
from asic_la import asic_simulator_helpers as helpers
from jax.interpreters.pxla import ShardedDeviceArray
from typing import Text, List, Tuple, Dict

AXIS_NAME = helpers.AXIS_NAME


def get_indices(N: int, nsteps: int) -> np.ndarray:
    """
    Helper function: divides `N` steps into `nsteps` subiterations.
    The returned array `cumsum` is such that the following two code
    fragments iterate through all elements of `array`:

    ```python
    for n in range(1, len(cumsum)):
      for m in array[cumsum[n - 1]:cumsum[n]]:
        continue
    ```

    Args:
      N: An integer.
      nsteps: The desired number of subarrays of an
        array of length `N`.
    Returns:
      np.ndarray: An array.
    """
    if nsteps > N:
        return np.arange(N)
    strides = [N // nsteps] * nsteps
    if N % nsteps != 0:
        strides.append(N % nsteps)
    cumsum = np.append(0, np.cumsum(strides))
    return cumsum


def get_final_state_in_steps(
    building_blocks: Tuple[cw.ComplexDeviceArray],
    axes: Tuple[Tuple[int]],
    num_discretes: int,
    nsteps: int,
    verbose: bool = False,
) -> Tuple[ShardedDeviceArray, Tuple[int], Tuple[int]]:
    """
    Get the final probability function after applying `building_blocks`. This functions compiles
    the computation into `nsteps` blocks of `pmapped` code. While being
    potentially slower than just calling `get_final_states`, is uses a lot less
    memory on the ASIC. The returned state has aligned axes.

    Args:
      building_blocks: List of tuples of 2D tensors. Tuples must be in (real, imag) format
        of the corresponding complex tensor.
      axes: List of tuples of ints. Represents the axes in which the building_block will
        apply. This functions assumes that all building_blocks were applied to the right.
        (i.e, state = state @ building_block).
      num_discretes: The number of discretes needed for the simulation.
      nsteps: The number of pmaped blocks into which the while computation
        should be divided.
      verbose: A verbosity flag.

    Returns:
      ShardedDiscretedProbabilityFunction: The final state, with aligned axes.
    """

    ndev = jax.local_device_count()
    state = jax.pmap(
        lambda x: ShardedDiscretedProbabilityFunction.zero_state(num_discretes),
        axis_name=AXIS_NAME,
    )(np.arange(ndev))
    state = apply_building_blocks_in_steps(state, building_blocks, axes, nsteps, verbose)
    return helpers.pmapped_align_axes(state, None)


def apply_building_blocks_in_steps(
    state: ShardedDiscretedProbabilityFunction,
    building_blocks: Tuple[cw.ComplexDeviceArray],
    axes: Tuple[Tuple[int]],
    nsteps: int,
    verbose: bool = False,
) -> Tuple[ShardedDeviceArray, Tuple[int], Tuple[int]]:
    """
    Get the probability function after applying `building_blocks` to `state`. This functions
    compiles the computation into `nsteps` blocks of `pmapped` code. While
    being potentially slower than just calling `get_final_states`, is uses a
    lot less memory on the ASIC.

    Args:
      state: A ShardedDiscretedProbabilityFunction.
      building_blocks: List of tuples of 2D tensors. Tuples must be in (real, imag) format
        of the corresponding complex tensor.
      axes: The discrete axes on which the building_block is applied.
      num_discretes: The number of discretes needed for the simulation.
      nsteps: The number of pmaped blocks into which the while computation
        should be divided.
      verbose: A verbosity flag.

    Returns:
      ShardedDiscretedProbabilityFunction: The result of applying the `building_blocks` to `state`.
    """
    cumsum = get_indices(len(building_blocks), nsteps)
    for n in range(1, len(cumsum)):
        if verbose:
            print(f"at step {n} in `apply_building_blocks_in_step`")
        tmp_building_blocks = building_blocks[cumsum[n - 1] : cumsum[n]]
        tmp_axes = axes[cumsum[n - 1] : cumsum[n]]
        state = helpers.pmapped_apply_building_blocks(state, tmp_building_blocks, tmp_axes)
    return state
