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
from asic_la.sharded_probability_function import (
    ShardedDiscretedProbabilityFunction,
    permute,
    invert_permutation,
)
from asic_la.sharded_probability_function import complex_workaround as cw
from asic_la import utils
from asic_la.preprocessor import preprocessor
from jax.interpreters.pxla import ShardedDeviceArray
from typing import Text, List, Tuple, Dict, Sequence

AXIS_NAME = asic_la.sharded_probability_function.jax_wrappers.AXIS_NAME


def apply_building_blocks(
    state: ShardedDiscretedProbabilityFunction,
    building_blocks: Tuple[cw.ComplexDeviceArray],
    axes: Tuple[Tuple[int]],
) -> ShardedDiscretedProbabilityFunction:
    """Get the final probability function after applying all building_blocks
    The perm attribute of the returned ShardedDiscretedProbabilityFunction is not aligned.

    Args:
      state: The initial state.
      building_blocks: The super-building_block matrices, packed into ComplexDeviceArray.
      axes: Tuple of tuples of ints. Represents the discrete axes on which the building_block will
        apply. This functions assumes that all building_blocks were applied to the right
        (i.e, state = state @ building_block).

    Returns:
      ShardedDiscretedProbabilityFunction: The final probabilityfunction.
    """
    for building_block, l in zip(building_blocks, axes):
        state = state.discrete_dot(building_block, l)
    return state


@functools.partial(jax.pmap, static_broadcasted_argnums=2, axis_name=AXIS_NAME)
def pmapped_apply_building_blocks(state, building_blocks, axes):
    """
    Pmapped version of `apply_building_blocks`
    """
    return apply_building_blocks(state, building_blocks, axes)


def get_final_state(
    building_blocks: Tuple[cw.ComplexDeviceArray],
    axes: Tuple[Tuple[int]],
    num_discretes: int,
) -> ShardedDiscretedProbabilityFunction:
    """
    Get the final probability function of a symplectic acyclic_graph after applying all building_blocks
    to the initial state. The `perm` attribute of the returned
    ShardedDiscretedProbabilityFunction is aligned.

    Args:
      building_blocks: The seven-discrete super-building_blocks of the acyclic_graph.
      axes: List of tuples of ints. Represents the discrete axes on which the super-building_block
        acts.
      num_discretes: The number of discretes needed for the simulation.

    Returns:
      ShardedDiscretedProbabilityFunction: The final probabilityfunction.
    """
    state = ShardedDiscretedProbabilityFunction.zero_state(num_discretes)
    state = apply_building_blocks(state, building_blocks, axes)
    state = state.align_axes()
    return state


def scalar_product_real(
    state1: ShardedDiscretedProbabilityFunction,
    state2: ShardedDiscretedProbabilityFunction,
    precision=jax.lax.Precision.HIGHEST,
) -> float:
    """
    Compute the real part of the scalar product between `state1` and `state2`.

    Args:
      state1: A probabilityfunction.
      state2: A probabilityfunction.
      precision: jax.lax.Precision argument.

    Returns:
      float: The real part of the scalar product between `state1` and `state2`.
    """
    assert (
        state1.perm == state2.perm
    ), "state1 and state2 have different `perm` attributes."
    axes = list(range(state1.free_num_discretes + 2))
    a = state1.concrete_tensor.conj()
    b = state2.concrete_tensor
    tmp = jnp.tensordot(
        a.real, b.real, (axes, axes), precision=precision
    ) - jnp.tensordot(a.imag, b.imag, (axes, axes), precision=precision)
    return jax.lax.psum(tmp, axis_name=AXIS_NAME)


pmapped_scalar_product_real = jax.pmap(
    scalar_product_real,
    static_broadcasted_argnums=2,
    axis_name=AXIS_NAME,
    out_axes=None,
)


@functools.partial(jax.pmap, static_broadcasted_argnums=1, axis_name=AXIS_NAME)
def pmapped_align_axes(
    state: ShardedDiscretedProbabilityFunction, perm: Tuple[int] = None
) -> ShardedDiscretedProbabilityFunction:
    """
    Pmapped version of applying perm-ordering using `align_axes`.

    Args:
      state: The probabilityfunction.

    Returns:
      ShardedDiscretedProbabilityFunction: The resulting probabilityfunction.
    """
    return state.align_axes(perm)


@functools.partial(
    jax.pmap, axis_name=AXIS_NAME, static_broadcasted_argnums=2, out_axes=None
)
def distributed_compute_pbaxistring_expectation(
    state: ShardedDiscretedProbabilityFunction,
    pbaxistring: Tuple[cw.ComplexDeviceArray],
    pbaxistring_operating_axes: Tuple[Tuple[int]],
    pbaxistring_coeff: float,
) -> ShardedDeviceArray:
    """
    Compute the expectation value of a prob-basis-axis-string operator.

    Args:
      psi: A probabilityfunction.
      pbaxistring: The parsed and preprocessed
        prob-basis-axis-string operator, represented as a tuple of broadcasted super-building_block
        matrices of shape (jax.local_device_count(), 128, 128).
      pbaxistring_operating_axes: The discrete axes on which the building_blocks in
        `pbaxistring` act.
      pbaxistring_coeff: The coefficient of the pbaxistring.

    Returns:
      ShardedDeviceArray: An array with 1 element containing the expectation
        value.
    """
    psi = apply_building_blocks(state, pbaxistring, pbaxistring_operating_axes)
    psi = psi.align_axes()
    return scalar_product_real(psi, state) * pbaxistring_coeff


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
    This function uses a single pmap and can be memory intensive for
    pbaxisums with many long prob-basis-axis-strings.

    Args:
      building_blocks: The building_blocks in broadcasted  super-matrix format, i.e. of shape
        (jax.local_device_count(), 128, 128).
      operating_axes: The discrete axes on which `building_blocks` act.
      pbaxisums: Supermatrices of large_block representation of pauli sum
        operators. A single pbaxistring is represented as an innermost list
        of matrix-large_blocks. The outermost list iterates through different
        prob-basis-axis-sums, the intermediate list iterates through pbaxistrings
        within a pbaxisum.
      pbaxisum_coeffs: The coefficients of the prob-basis-axis-strings appearing in the
        union of all prob-basis-axis-sum operators.
      num_discretes: The number of discretes needed for the simulation.

    Returns:
      ShardedDeviceArray: The expectation values.
    """
    final_state = apply_building_blocks(
        ShardedDiscretedProbabilityFunction.zero_state(num_discretes), building_blocks, operating_axes
    )

    expectation_values = jnp.zeros(len(pbaxisums))

    for m, pbaxisum in enumerate(pbaxisums):
        pbaxisum_op_axes = pbaxisums_operating_axes[m]
        pbaxisum_coeff = pbaxisum_coeffs[m]

        expectation_value = 0.0
        for n, pbaxistring in enumerate(pbaxisum):
            op_axes = pbaxisum_op_axes[n]
            coeff = pbaxisum_coeff[n]
            psi = apply_building_blocks(final_state, pbaxistring, op_axes).align_axes(
                final_state.perm
            )
            expectation_value += scalar_product_real(psi, final_state) * coeff
        expectation_values = expectation_values.at[m].set(
            expectation_value.real[0]
        )

    return expectation_values


@functools.partial(
    jax.pmap, axis_name=AXIS_NAME, static_broadcasted_argnums=(1, 2)
)
def compute_final_state(
    building_blocks: Tuple[cw.ComplexDeviceArray],
    operating_axes: Tuple[Tuple[int]],
    num_discretes: int,
) -> ShardedDeviceArray:
    """
    Pmapped function for computing the final state of a symplectic computation.

    Args:
      building_blocks: The super-building_block matrices of the symplectic acyclic_graph in broadcasted format,
        i.e. of shape (jax.local_device_count(), 128, 128).
      operating_axes: The discrete axes on which `building_blocks` act.
      num_discretes: The number of discretes of the symplectic acyclic_graph.

    Returns:
      ShardedDiscretedProbabilityFunction: The final probabilityfunction.
    """
    return get_final_state(building_blocks, operating_axes, num_discretes)


@functools.partial(jax.pmap, static_broadcasted_argnums=3, axis_name=AXIS_NAME)
def apply_building_blocks_multiply_and_add(
    accumulator: ShardedDiscretedProbabilityFunction,
    state: ShardedDiscretedProbabilityFunction,
    building_blocks: Tuple[cw.ComplexDeviceArray],
    operating_axes: Tuple[Tuple[int]],
    coeff: cw.ComplexDeviceArray,
) -> Tuple[ShardedDiscretedProbabilityFunction, Tuple[int], Tuple[int]]:
    """
    Apply `building_blocks` to a `ShardedDiscretedProbabilityFunction`, multiply the result by `coeff`
    and add the result to `accumulator`.

    Args:
      accumulator: The probabilityfunction to which the result should be added.
      state: The state to which the building_blocks should be applied.
      building_blocks: The super-building_block matrices that are to be applied to the state.
      operating_axes: The discrete axes on which `building_blocks` act.
      coeff: A complex coefficient.

    Returns:
      ShardedDiscretedProbabilityFunction: The resulting of the operation.
    """
    y = apply_building_blocks(state, building_blocks, operating_axes)
    # The result of __add__ has a perm attribute equal to the second argument.
    # We want to keep `accumulator`'s perm here, so it's added from
    # the right.
    return y * coeff + accumulator


@functools.partial(
    jax.pmap, static_broadcasted_argnums=3, axis_name=AXIS_NAME, out_axes=None
)
def _get_gradients(
    bra: ShardedDiscretedProbabilityFunction,
    ket: ShardedDiscretedProbabilityFunction,
    gradients: Tuple[cw.ComplexDeviceArray],
    operating_axes: Tuple[int],
) -> Tuple[ShardedDeviceArray, Tuple[int], Tuple[int]]:
    """
    Compute the gradients at step `n`.

    Args:
      bra: The conjubuilding_blockd (bottom) probabilityfunction.
      ket: The unconjubuilding_blockd (top) probabilityfunction.
      gradients: The supergradienst of the large_block-matrix at step `n`.
      operating_axes: The discrete axes on which `building_blocks[n]` acts.

    Returns:
      ShardedDeviceArray: The gradient values corresponding to
        `gradients`.
    """
    gradient_values = jnp.zeros(len(gradients))
    for n, grad in enumerate(gradients):
        tmpket = apply_building_blocks(ket, (grad,), (operating_axes,)).align_axes(
            bra.perm
        )
        gradient_value = 2 * scalar_product_real(bra, tmpket)
        gradient_values = gradient_values.at[n].set(gradient_value)
    return gradient_values


@jax.partial(jax.pmap, static_broadcasted_argnums=(1, 2), axis_name=AXIS_NAME)
def pmapped_get_zeros(_, num_discretes, perm):
    return ShardedDiscretedProbabilityFunction.zeros(num_discretes, perm)


def compute_gradients_multiple_pmaps(
    building_blocks: Tuple[cw.ComplexDeviceArray],
    gradients: Tuple[Tuple[int, cw.ComplexDeviceArray]],
    operating_axes: Tuple[Tuple[int]],
    pbaxisum: Tuple[Tuple[cw.ComplexDeviceArray]],
    pbaxisum_operating_axes: Tuple[Tuple[Tuple[int]]],
    pbaxisum_coeffs: Tuple[float],
    num_discretes: int,
    num_params: int,
) -> Tuple[ShardedDeviceArray, ShardedDeviceArray]:
    """
    Compute the gradients of expectation values of a parametrized symplectic acyclic_graph
    for a self_adjoint observable given by `pbaxisum`. This function uses mulitple pmaps
    to compute the result.

    Args:
      building_blocks: The building_blocks in broadcasted super-matrix format, i.e. of shape
        (jax.local_device_count(), 128x128)
      gradients: The gradients in super-matrix format (i.e. 128x128), broadcasted to all
        local ASIC devices. The element `gradients[n]` is a dictionary mapping integers,
        which encode the sympy.Symbols on which `building_blocks[n]` depends, to
        the derivative of `building_blocks[n]` with respect to that symbol.
      operating_axes: The discrete axes on which `building_blocks` act.
      pbaxisum: Supermatrices of large_block
        representation of a pauli sums operator. A single pbaxistring is
        represented as the innermost tuple of matrix-large_blocks. The outer tuple
        iterates through pbaxistrings within a pbaxisum.
      pbaxisum_operating_axes: The discrete axes on which the super-building_block matrices
        in `pbaxisum` act.
      pbaxisum_coeffs: The coefficients of the
        prob-basis-axis-strings appearing in the prob-basis-axis-sum operator.
      num_discretes: The number of discretes needed for the simulation.
      num_params: The number of parameters on which the acyclic_graph depends.

    Returns:
      np.ndarray: A n array of length `num_params`  holding the gradients.
      float: The expectation value of `pbaxisum`.
    """
    accumulated_gradients = np.zeros(num_params)
    # get the final state (it is ordered)
    final_state = compute_final_state(building_blocks, operating_axes, num_discretes)

    # apply all pbaxistrings one by one and sum it up
    psi = pmapped_get_zeros(
        np.arange(jax.local_device_count()), num_discretes, final_state.perm
    )
    for n, p in enumerate(pbaxisum):
        psi = apply_building_blocks_multiply_and_add(
            psi, final_state, p, pbaxisum_operating_axes[n], pbaxisum_coeffs[n]
        )
        # `psi` is still in natural discrete ordering, ie small to large.
        assert (
            psi.perm == final_state.perm
        ), "`psi` and `final_state` have different `perm` attributes."

    # compute the expectation value.
    expectation_value = pmapped_scalar_product_real(
        psi, final_state, jax.lax.Precision.HIGHEST
    )

    reversed_axes = list(reversed(operating_axes))
    reversed_gradients = list(reversed(gradients))

    # now the backwards pass
    for m, building_block in enumerate(reversed(building_blocks)):
        axes = reversed_axes[m]
        # first unfold `final_state` backwards by `building_block`.
        # Note that composition of `ShardedDiscretedProbabilityFunction.discrete_dot`
        # with its inverse does not leave `perm` invariant, i.e.
        # FIXME : if we can change `ShardedDiscretedProbabilityFunction.discrete_dot`
        # such that the above call preserves `perm` we can get rid of a lot of
        # `align_axes()` calls below by ensuring that the initial `final_state`
        # and all initial `psis` have identical `perm`.
        final_state = pmapped_apply_building_blocks(
            final_state, (building_block.conj().transpose((0, 2, 1)),), (axes,)
        )
        param_keys = [int(t[0][0]) for t in reversed_gradients[m]]
        gradients_tuple = tuple([t[1] for t in reversed_gradients[m]])
        gradient_values = _get_gradients(
            psi, final_state, gradients_tuple, axes
        )
        # accumulate the gradients
        for n, p in enumerate(param_keys):
            accumulated_gradients[p] += gradient_values[n]

        # finally unfold `psi` backwards by one step.
        psi = pmapped_apply_building_blocks(
            psi, (building_block.conj().transpose((0, 2, 1)),), (axes,)
        )
    return accumulated_gradients, expectation_value


@functools.partial(
    jax.pmap,
    axis_name=AXIS_NAME,
    static_broadcasted_argnums=(1, 2, 3),
    out_axes=(None, None),
)
def distributed_get_samples(
    building_blocks: Tuple[cw.ComplexDeviceArray],
    axes: Tuple[Tuple[int]],
    num_discretes: int,
    repetitions: int,
    prng_key,
):
    """Calculate bitstring samples from the given contraction of building_blocks.

    Distributed across a node or cluster.

    This function is split out from the TN contraction in order to utilize numpy
    for building_block merging and drastically reduce the XLA compile time.

    Args:
      building_blocks: The building_blocks in super-matrix format (i.e. 128x128), broadcasted to all local
        ASIC devices
      axes: The discrete axes on which the building_blocks act.
      num_discretes: The number of discretes.
      repetitions: Number of samples to take.
      prng_key: The PRNGKey to use for random sampling. This key has to be broadcasted
        to all local ASIC devices. All global devices need to receive the same key.

    Returns:
      A DeviceArray of the sampled bitstrings as uints.
    """
    return get_final_state(building_blocks, axes, num_discretes).sample(
        repetitions, prng_key
    )


def get_amplitudes_from_state(state, global_bitstrings, local_bitstrings):
    """Calculate the amplitudes of `state` for the given bitstrings.

    Args:
      state: The distributed probability function.
      global_bitstrings: The global part of the bitstrings
        (i.e. the bitstrings of the global discretes).
      local_bitstrings: The local part of the bitstrings
        (i.e. the bitstrings of the local discretes).

    Returns: Two arrays, the real part and the imaginary part of the amplitudes.
    """

    real_part = state.concrete_tensor.real.ravel()
    imag_part = state.concrete_tensor.imag.ravel()

    @jax.vmap
    def inner(global_bitstring, local_bitstring):
        # Grab real and imaginary parts.
        real = jnp.where(
            global_bitstring == jax.lax.axis_index(AXIS_NAME),
            real_part[
                local_bitstring],
            0.0)
        imag = jnp.where(global_bitstring == jax.lax.axis_index(AXIS_NAME),
                         imag_part[local_bitstring], 0.0)
        # Psum is required to replicate results across all cores.
        real = jax.lax.psum(real, AXIS_NAME)
        imag = jax.lax.psum(imag, AXIS_NAME)
        return real, imag

    return inner(global_bitstrings, local_bitstrings)


@functools.partial(
    jax.pmap,
    axis_name=AXIS_NAME,
    static_broadcasted_argnums=(3, 4),
    in_axes=(0, None, None),
    out_axes=(None, None),
)
def _pmapped_get_amplitudes(
    building_blocks: Tuple[cw.ComplexDeviceArray],
    global_bitstrings: "jnp.DeviceArray[jnp.uint32]",
    local_bitstrings: "jnp.DeviceAarray[jnp.uint32]",
    axes: Tuple[Tuple[int]],
    num_discretes: int,
):
    """Calculate the amplitudes of the given bitstrings for the final state
    of a symplectic computation as given by `building_blocks` and `axes`.

    Defined to be efficiently distributed across a node or cluster.

    Args:
      building_blocks: The building_blocks in super-matrix format (i.e. 128x128), broadcasted to
        all local ASIC devices
      global_bitstrings: The global part of the bitstrings
        (i.e. the bitstrings of the global discretes).
      local_bitstrings: The local part of the bitstrings
        (i.e. the bitstrings of the local discretes).
      axes: The discrete axes on which the building_blocks act.
      num_discretes: Number of discretes.

    Returns: Two arrays, the real part and the imaginary part of the amplitudes.
    """
    state = get_final_state(building_blocks, axes, num_discretes)
    return get_amplitudes_from_state(state, global_bitstrings, local_bitstrings)

def get_samples(
    building_blocks: Tuple[np.ndarray],
    prng_key,
    discrete_indices_per_building_block: Tuple[Tuple[int]],
    num_discretes: int,
    repetitions: int,
):
    """Get a full bitstring sample from a symplectic acyclic_graph.

    Args:
      building_blocks: Tuple of tensors of the symplectic building_block.
      prng_key: A jax.random.PRNGKey.
      discrete_indices_per_building_block: Each inner tuple contains the linear locations
        of the discretes in the original sequence of discretes on which the building_blocks act.
        The original sequence of discretes was passed to `parser.parse` but is not
        needed any more at this point.
      num_discretes: Number of discretes in the simulation.
      repetitions: How many samples to take from the final probability function.

    Returns:
      An array of integer samples from the final probabilityfunction. The binary
        representation of the int is the sampled bitstring.
    """
    building_blocks, _, operating_axes = preprocessor.preprocess(
        building_blocks,
        [dict()] * len(building_blocks),
        discrete_indices_per_building_block,
        num_discretes,
        max_discrete_support=7,
    )
    # Here, we need to broadcast the prngkey and the building_blocks
    # to include an extra dimension equal to the number of local
    # ASIC cores. This axis is what is sliced over during the jax.pmap.
    num_cores = jax.local_device_count()
    broadcasted_building_blocks = preprocessor.canonicalize_building_blocks(
        building_blocks, broadcasted_shape=num_cores
    )
    broadcasted_prng_key = np.broadcast_to(
        prng_key, (num_cores,) + prng_key.shape
    )
    global_samples, local_samples = distributed_get_samples(
        broadcasted_building_blocks,
        utils.to_tuples_of_ints(operating_axes),
        num_discretes,
        repetitions,
        broadcasted_prng_key,
    )
    num_local_discretes = num_discretes - int(np.log2(jax.device_count()))
    samples = (
        np.asarray(global_samples, np.int64) << num_local_discretes
    ) + np.asarray(local_samples, np.int64)
    return samples


def get_amplitudes(
    building_blocks: Tuple[np.ndarray],
    discrete_indices_per_building_block: Tuple[Tuple[int]],
    num_discretes: int,
    bitstrings: Sequence[int],
):
    """Get complex amplitudes from the final probability function.

    Args:
      building_blocks: Tuple of tensors of the symplectic building_block.
      discrete_indices_per_building_block: Each inner tuple contains the linear locations
        of the discretes in the original sequence of discretes on which the building_blocks act.
        The original sequence of discretes was passed to `parser.parse` but is not
        needed any more at this point.
      num_discretes: Number of discretes in the simulation.
      bitstrings: Which bitstrings to sample from. Should be an aray of ints.

    Returns:
      np.ndarray: The complex amplitudes.
    """
    bitstrings = np.array(bitstrings).astype(np.uint64)
    num_global = int(np.round(np.log2(jax.device_count())))
    if num_global > 32:
        raise ValueError(
            f"got num_global = {num_global}. Maximum number of global"
            "discretes must not exceed 32"
        )
    num_local = num_discretes - num_global
    if num_local > 32:
        raise ValueError(
            f"got num_local = {num_local}. Maximum number of local"
            "discretes must not exceed 32"
        )

    # Global discretes are always the first N discretes after axes alignment,
    # and the given bitstrings are in big endian, so we shift the bits
    # down to grab them.
    global_bitstrings = bitstrings >> num_local
    local_bitstrings = bitstrings & (2 ** num_local - 1)

    building_blocks, _, operating_axes = preprocessor.preprocess(
        building_blocks,
        [dict()] * len(building_blocks),
        discrete_indices_per_building_block,
        num_discretes,
        max_discrete_support=7,
    )

    # Here, we need to broadcast the prngkey and the building_blocks
    # to include an extra dimension equal to the number of local
    # ASIC cores. This axis is what is sliced over during the jax.pmap.
    broadcasted_building_blocks = preprocessor.canonicalize_building_blocks(
        building_blocks, broadcasted_shape=jax.local_device_count()
    )
    real, imag = _pmapped_get_amplitudes(
        broadcasted_building_blocks,
        global_bitstrings,
        local_bitstrings,
        utils.to_tuples_of_ints(operating_axes),
        num_discretes,
    )

    return np.array(real + imag * 1j, dtype=np.complex64)
