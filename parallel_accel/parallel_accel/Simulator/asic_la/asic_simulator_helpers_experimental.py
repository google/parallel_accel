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
import functools
import random
import time
import jax
import jax.numpy as jnp
import numpy as np
import asic_la
from asic_la.sharded_probability_function import ShardedDiscretedProbabilityFunction
import asic_la.sharded_probability_function.complex_workaround as cw
from asic_la.sharded_probability_function import invert_permutation
from asic_la import utils
from asic_la.preprocessor import preprocessor
from jax.interpreters.pxla import ShardedDeviceArray
from typing import Text, List, Tuple, Dict
import asic_la.asic_simulator_helpers as helpers

AXIS_NAME = asic_la.sharded_probability_function.jax_wrappers.AXIS_NAME


@functools.partial(
    jax.pmap, axis_name=AXIS_NAME, static_broadcasted_argnums=(1, 3, 5)
)
def distributed_compute_expectations(
    building_blocks: Tuple[cw.ComplexDeviceArray],
    operating_axes: Tuple[Tuple[int]],
    pbaxisums: Tuple[Tuple[cw.ComplexDeviceArray]],
    pbaxisums_operating_axes: Tuple[Tuple[Tuple[int]]],
    pbaxisum_coeffs: Tuple[Tuple[float]],
    num_discretes: int,
) -> ShardedDeviceArray:
    """
    Compute the expectation values of several observables given in `pbaxisums`.
    This function uses a single pmap and can be memory intesive for
    pbaxisums with many long prob-basis-axis-strings.

    Args:
      building_blocks: The building_blocks in super-matrix format (i.e. 128x128)
      operating_axes: The discrete axes on which `building_blocks` act.
      pbaxisums: Supermatrices of large_block representation of pauli sum
        operators. A single pbaxistring is represented as an innermost list
        of matrix-large_blocks. The outermost list iterates through different
        prob-basis-axis-sums, the intermediate list iterates through pbaxistrings
        within a pbaxisum.
      pbaxisums_operating_axes: The discrete axes on which the pbaxisums act.
      pbaxisum_coeffs: The coefficients of the
        prob-basis-axis-strings appearing in the union of all prob-basis-axis-sum operators.
      num_discretes: The number of discretes needed for the simulation.
      num_params: The number of parameters on which the acyclic_graph depends.

    Returns:
      ShardedDeviceArray: The expectation values.
    """
    num_pbaxisums = len(pbaxisums)
    expectation_values = jnp.zeros(num_pbaxisums)
    final_state = helpers.get_final_state(building_blocks, operating_axes, num_discretes)

    for m, pbaxisum in enumerate(pbaxisums):
        pbaxisum_op_axes = pbaxisums_operating_axes[m]
        pbaxisum_coeff = pbaxisum_coeffs[m]

        # `psi` is brought into natural discrete order
        # don't forget to also align the axes here!
        coeff = pbaxisum_coeff[0]
        psi = helpers.apply_building_blocks(
            final_state, pbaxisum[0], pbaxisum_op_axes[0]
        ).align_axes()
        expectation_value = (
            helpers.scalar_product_real(psi, final_state) * coeff
        )

        for n in range(1, len(pbaxisum)):
            pbaxistring = pbaxisum[n]
            op_axes = pbaxisum_op_axes[n]
            coeff = pbaxisum_coeff[n]
            psi = helpers.apply_building_blocks(
                final_state, pbaxistring, op_axes
            ).align_axes()
            expectation_value += (
                helpers.scalar_product_real(psi, final_state) * coeff
            )

        # at this point all `psis` are in natural discrete ordering,
        # with the same `labels` values as `final_state` (i.e.
        # `labels = [0,1,2,3,..., num_discretes - 1]`). They also all have the
        # same (sorted) `perm` ordering due to the call to `align_axes()`.

        # compute the expectation values. Note that `psi` and `final_state`
        # have identical `perm` and `labels`.
        expectation_values = expectation_values.at[m].set(
            expectation_value.real[0]
        )

    return expectation_values


@functools.partial(
    jax.pmap,
    axis_name=AXIS_NAME,
    static_broadcasted_argnums=(2, 4, 6, 7),
    out_axes=(None, None),
)
def distributed_compute_gradients(
    building_blocks: Tuple[cw.ComplexDeviceArray],
    gradients: Tuple[Dict[int, cw.ComplexDeviceArray]],
    operating_axes: Tuple[Tuple[int]],
    pbaxisums: Tuple[Tuple[cw.ComplexDeviceArray]],
    pbaxisums_operating_axes: Tuple[Tuple[Tuple[int]]],
    pbaxisum_coeffs: Tuple[Tuple[float]],
    num_discretes: int,
    num_params: int,
) -> Tuple[ShardedDeviceArray, ShardedDeviceArray]:
    """
    Compute the gradients of expectation values of a parametrized symplectic acyclic_graph
    for a list of self_adjoint observables.

    Args:
      building_blocks: The building_blocks in super-matrix format (i.e. 128x128)
      gradients: The gradients in super-matrix format (i.e. 128x128).
        The element `gradients[n]` is a dictionary mapping integers, which
        encode the sympy.Symbols on which `building_blocks[n]` depends, to
        the derivative of `building_blocks[n]` with respect to that symbol.
      operating_axes: The discrete axes on which `building_blocks` act.
      pbaxisums: Supermatrices of large_block
        representation of pauli sums operators. A single pbaxistring is
        represented as an innermost list of matrix-large_blocks. The outermost
        list iterates through different prob-basis-axis-sums, the intermediate list iterates
        through pbaxistrings within a pbaxisum.
      pbaxisums_operating_axes: The discrete axes on which the pbaxisums act.
      pbaxisum_coeffs: The coefficients of the
        prob-basis-axis-strings appearing in the union of all prob-basis-axis-sum operators.
      num_discretes: The number of discretes needed for the simulation.
      num_params: The number of parameters on which the acyclic_graph depends.

    Returns:
      jax.ShardedDeviceArray: A (`num_params`, `len(pbaxisums)`) shaped array
        storing the gradients per pbaxisum operator and acyclic_graph parameter.
      jax.ShardedDeviceArray: The expectation value of each observable in
        `pbaxisums`, in corresponding order.
    """
    num_pbaxisums = len(pbaxisums)
    accumulated_gradients = jnp.zeros((num_params, num_pbaxisums))
    expectation_values = jnp.zeros(num_pbaxisums)

    # we need `final_state` to be in natural discrete ordering
    # because that's the ordering that the pbaxisums are expecting.
    # `final_state` also has its axes aligned
    final_state = helpers.get_final_state(building_blocks, operating_axes, num_discretes)

    psis = []
    for m, pbaxisum in enumerate(pbaxisums):
        pbaxisum_op_axes = pbaxisums_operating_axes[m]
        pbaxisum_coeff = pbaxisum_coeffs[m]

        psi = ShardedDiscretedProbabilityFunction.zeros(num_discretes, final_state.perm)
        for n, pbaxistring in enumerate(pbaxisum):
            op_axes = pbaxisum_op_axes[n]
            coeff = pbaxisum_coeff[n]
            tmp = helpers.apply_building_blocks(final_state, pbaxistring, op_axes)
            psi = tmp * coeff + psi

        # at this point all `psis` have the same `perm` as `final_state` (i.e.
        # Now compute the expectation values.
        expectation_values = expectation_values.at[m].set(
            helpers.scalar_product_real(psi, final_state)
        )
        psis.append(psi)

    reversed_axes = list(reversed(operating_axes))
    reversed_gradients = list(reversed(gradients))
    # now the backwards pass
    for m, building_block in enumerate(reversed(building_blocks)):
        axes = reversed_axes[m]
        final_state = final_state.discrete_dot(building_block.conj().transpose((1, 0)), axes)
        # Note that `final_state ` has an arbitrary `perm` order at this point.
        for k, grad in reversed_gradients[m]:
            # apply the gradient large_block
            # FIXME : If we can change `ShardedDiscretedProbabilityFunction.discrete_dot`
            # such that `tmp_state` has the same `perm` as `psi` we can remove all
            # align_axes calls
            tmp_state = final_state.discrete_dot(grad, axes)
            for n, psi in enumerate(psis):
                # we need to align the axes of both `tmp_state` to `psi`
                gradient_value = helpers.scalar_product_real(
                    psi, tmp_state.align_axes(psi.perm)
                )
                accumulated_gradients = jax.ops.index_add(
                    accumulated_gradients,
                    jax.ops.index[k, n],
                    2 * gradient_value,
                )

        # finally unfold `psis` backwards by one step.
        for n, psi in enumerate(psis):
            psi = psi.discrete_dot(building_block.conj().transpose((1, 0)), axes)
            psis[n] = psi
    return accumulated_gradients, expectation_values
