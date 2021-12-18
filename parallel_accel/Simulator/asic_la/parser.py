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
"""Parser Functions (LinearAlgebra -> numpy/jax)"""
from functools import partial
import numbers
from operator import mul
import time
from typing import Any, Set, Tuple, Sequence, Text, Union, Callable, Dict, List

import linear_algebra
import jax
import jax.numpy as jnp
import numpy as np
import sympy
import graph_helper_tool as tn

from parallel_accel.shared import logger
import asic_la.config as config
import asic_la.sharded_probability_function.complex_workaround as cw


Array = Any
LinearAlgebraDiscreted = linear_algebra.Qid
# NOTE : linear_algebra.cond_rotate_x is linear_algebra.cond_x_angle
SIMPLE_BUILDING_BLOCKS = {
    linear_algebra.flip_x_axis_angle,
    linear_algebra.flip_x_axis_angle_square,
    linear_algebra.flip_y_axis_angle,
    linear_algebra.flip_y_axis_angle_square,
    linear_algebra.flip_z_axis_angle,
    linear_algebra.flip_z_axis_angle_square,
    linear_algebra.flip_pi_over_4_axis_angle,
    linear_algebra.cond_rotate_z,
    linear_algebra.cond_rotate_x,
    linear_algebra.cond_x_angle,
    linear_algebra.swap_angle,
    linear_algebra.imaginary_swap_angle,
}
BUILDING_BLOCKS_WITH_ANGLES = {linear_algebra.x_axis_two_angles, linear_algebra.imaginary_swap_two_angles}
SUPPORTED_PARAMETERIZABLE_GATE_TYPES = (
    SIMPLE_BUILDING_BLOCKS | BUILDING_BLOCKS_WITH_ANGLES | {linear_algebra.EmptyBuildingBlock, linear_algebra.rotate_on_xy_plane}
)

# the jax backend is reconfigured in parse_gradients to use CPU by default
# This flag can be set to `None`, but maybe change to 'cpu' explicitly later.
JAX_PREPRO_BACKEND = config.JAX_PREPRO_BACKEND

log = logger.get_logger(__name__)


def parse(
    acyclic_graph: linear_algebra.Graph,
    discretes: Sequence[linear_algebra.Qid],
    param_resolver: linear_algebra.ParamResolver = None,
    dtype=np.complex64,
    quiet: bool = False,
):
    """
    Resolve the acyclic_graph and compute the gradients of each building_block with respect to its
    parameters. The integers in the returned list `discrete_indices_per_building_block`
    are linear indices into the list of passed discretes.

    Args:
      acyclic_graph: The linear_algebra symplectic acyclic_graph.
      discretes: The linear_algebra.Discreteds of the full system.
      param_resolver: Optional parameter resolver used for resolving a possibly
        variabled acyclic_graph.
      dtype: The desired dtype of the returned building_blocks.
      quiet: Flag that suppresses logging when True. Default is False.


    Returns:
      building_blocks (list): A list of resolved non-observation building_blocks in the acyclic_graph.
      gradients (list[dict]): A list of gradients. `gradients[n]`
        maps the symbols on which `building_blocks[n]` depends to the respective
        gradient of `building_blocks[n]` with respect to this symbol.
      discrete_indices_per_building_block (tuple[tuple[int]]): A sequence of tuples, one for
        each non-observation building_block in the acyclic_graph, holding the discrete labels on
        which the corresponding building_block acts. Discreted labels here refers to a
        fixed labelling of the original discretes used in `discretes`. That is,
        if the original discretes were given as a (not necessarily ordered) sequence
        `discretes`, the labels are given by
        ```
        labels = [str(q) for q in range(len(discretes))]
        ```
        i.e. each discrete in `discretes` gets tagged with a label according to its
        position in `discretes`. The returned acyclic_graph is defined on these auxilliary
        labels.
    """
    # clear the cache of jax.lib.xla_bridge.get_backend and
    # and manually set the jax config to cpu
    config.set_jax_config_to_cpu()
    if param_resolver is None:
        param_resolver = linear_algebra.ParamResolver({})
    t0 = time.time()
    building_blocks = []
    gradients = []
    symbols = set()
    discrete_to_index = {q: i for i, q in enumerate(discretes)}
    discrete_indices_per_building_block = []
    for op in acyclic_graph.all_operations():
        resolved = linear_algebra.resolve_parameters(op, param_resolver)
        if linear_algebra.has_unitary(resolved):
            unitary = linear_algebra.unitary(resolved)
            unitary = unitary.reshape((2, 2) * len(resolved.discretes))
            matrix_gradients = get_autodiff_gradient(op.building_block, param_resolver)
            grad = {
                k: v.reshape((2, 2) * len(resolved.discretes)).astype(dtype)
                for k, v in matrix_gradients.items()
            }
            building_blocks.append(unitary.astype(dtype))
            gradients.append(grad)
            symbols.update(grad.keys())
            discrete_indices_per_building_block.append(
                tuple([discrete_to_index[q] for q in resolved.discretes])
            )
        elif not linear_algebra.is_observation(resolved):
            raise NotImplementedError(
                "support for operations other than unitary building_blocks and "
                "observations is not yet implemented"
            )

    if not quiet:
        log.info(
            "parsing finished",
            num_building_blocks=len(building_blocks),
            num_grad_building_blocks=len([g for g in gradients if len(g) > 0]),
            num_params=len(symbols),
            num_discretes=len(discretes),
            parsing_time=time.time() - t0,
        )
    # clear the cache of jax.lib.xla_bridge.get_backend and
    # reset jax config to its value prior to calling
    # config.set_jax_config() above,
    config.reset_jax_to_former_config()
    return building_blocks, gradients, tuple(discrete_indices_per_building_block)


def parse_pbaxisums(
    pbaxisums: List[linear_algebra.ProbBasisAxisSum],
    discretes: Sequence[linear_algebra.Qid],
    dtype=np.complex64,
):
    """
    Parse a list of linear_algebra.ProbBasisAxisSum operators.

    Args:
      pbaxisums: A list of ProbBasisAxisSum operators.
      discretes: The linear_algebra.Discreteds of the full system.
      dtype: An optional dtype for the resulting arrays.

    Returns:
      List[List[List[np.ndarray]]]: The matrix building_blocks. Each entry in the list
        represents a ProbBasisAxisString as a list of np.ndarrays.
      List[List[np.ndarray]]: The coefficients of the ProbBasisAxiss strings.
      List[List[List[Tuple[int]]]]: Operating labels, discrete index on which each
        building_block of the ProbBasisAxisSum representation acts.
    """

    building_blocks = []
    coeffs = []
    oplabels = []
    t0 = time.time()
    for pbaxisum in pbaxisums:
        tmpbuilding_blocks, tmpcoeffs, tmpaxes = _parse_pbaxisum(pbaxisum, discretes, dtype)
        building_blocks.append(tmpbuilding_blocks)
        coeffs.append(tmpcoeffs)
        oplabels.append(tmpaxes)
    log.info(
        "pbaxisum parsing finished",
        num_pbaxis=len(pbaxisums),
        num_discretes=len(discretes),
        parsing_time=time.time() - t0,
    )
    return building_blocks, coeffs, oplabels


def get_symbol_expressions(linear_algebra_building_block: linear_algebra.Gate) -> Set[sympy.Basic]:
    """
    Obtain a set of parameter-symbols of `linear_algebra_building_block`.
    As opposed to `linear_algebra.parameter_symbols`, this function
    also return expressions of symbols.

    Args:
      linear_algebra_building_block: One of the supported linear_algebra building_blocks.
    Returns:
      set[sympy.Basic]: Set of sympy symbols or expressions.
    """
    symbols = set()
    if isinstance(linear_algebra_building_block, tuple(SIMPLE_BUILDING_BLOCKS)):
        if isinstance(linear_algebra_building_block._exponent, sympy.Basic):
            symbols |= {linear_algebra_building_block._exponent}
    elif isinstance(linear_algebra_building_block, (linear_algebra.x_axis_two_angles, linear_algebra.imaginary_swap_two_angles)):
        if isinstance(linear_algebra_building_block._exponent, sympy.Basic):
            symbols |= {linear_algebra_building_block._exponent}
        if isinstance(linear_algebra_building_block._phase_exponent, sympy.Basic):
            symbols |= {linear_algebra_building_block._phase_exponent}
    elif isinstance(linear_algebra_building_block, linear_algebra.rotate_on_xy_plane):
        if isinstance(linear_algebra_building_block.theta, sympy.Basic):
            symbols |= {linear_algebra_building_block.theta}
        if isinstance(linear_algebra_building_block.phi, sympy.Basic):
            symbols |= {linear_algebra_building_block.phi}
    else:
        raise TypeError(f"building_block type {type(linear_algebra_building_block)} not supported")
    return symbols


def assert_is_variabled(linear_algebra_building_block: linear_algebra.Gate):
    """
    Check if `linear_algebra_building_block` is parametrized or not.

    Args:
      linear_algebra_building_block: One of the supported linear_algebra building_blocks.

    Raises:
      ValueError: If the building_block is not parametrized
    """
    if (
        not is_variabled(linear_algebra_building_block)
        or len(linear_algebra.parameter_symbols(linear_algebra_building_block)) == 0
    ):
        raise ValueError("Gate is not variabled.")


def assert_is_allowed_expression(val):
    """
    Assert that `val` is an allowed sympy-expresssion.
    `val` can be either a sympy.Symbol, a numeric constant,
    or an expression of the form symbol * constant,
    constant * symbol or symbol / constant.

    Args:
      val: Any

    Raises:
      ValueError: If `val` is not an allowed sympy-expression.
    """

    msg = (
        f"Parameters can only be sympy.Symbols, not expressions. "
        f"Got an exponent = {val} of type(exponent) = {type(val)}"
    )
    val = sympy.sympify(val)
    if not len(val.free_symbols) < 2:
        raise ValueError("only expression with one free symbol are allowed")
    if isinstance(val, sympy.Basic):
        if val.is_number:
            return
        if val.is_symbol:
            return
    args = val.args
    if val.func is sympy.Mul:
        if (
            (args[0].is_symbol and args[1].is_number)
            or (args[1].is_symbol and args[0].is_number)
            or (args[0].is_number and args[1].is_number)
        ):
            return
        if args[0].is_symbol and args[1].func is sympy.Pow:
            if args[1].args[0].is_number and args[1].args[1].is_number:
                return
    raise ValueError(msg)


def convert_linear_algebra_eigen_components_to_complex_device_array(
    eigen_components: List[Tuple[float, np.ndarray]]
) -> List[Tuple[float, cw.ComplexDeviceArray]]:
    eig_comps = []
    for coeff, comp in eigen_components:
        parsed_comp = cw.ComplexDeviceArray(
            jnp.array(comp.real, dtype=jnp.float64),
            jnp.array(comp.imag, dtype=jnp.float64),
        )

        eig_comps.append((coeff, parsed_comp))
    return eig_comps


def _resolve_eigenbuilding_block_params(
    linear_algebra_building_block: linear_algebra.Gate, param_resolver: linear_algebra.ParamResolver
) -> Tuple[float, float, Sequence[Tuple[float, cw.ComplexDeviceArray]]]:
    """
    Resolve the parameters of `linear_algebra_building_block`

    Args:
      linear_algebra_building_block: A building_block of one the types in SIMPLE_BUILDING_BLOCKS

    Returns:
      tuple: The resolved parameters of `linear_algebra_building_block`

    """
    exponent = param_resolver.value_of(linear_algebra_building_block._exponent)
    global_shift = linear_algebra_building_block._global_shift
    # seperate real and imaginary parts of the component-matrices
    # into a ComplexDeviceArray
    eigen_components = convert_linear_algebra_eigen_components_to_complex_device_array(
        linear_algebra_building_block._eigen_components()
    )
    return exponent, global_shift, eigen_components


def _resolve_phased_xpow_params(
    linear_algebra_building_block: linear_algebra.x_axis_two_angles, param_resolver: linear_algebra.ParamResolver
) -> Tuple[float, float, float]:
    """
    Resolve the parameters of `linear_algebra_building_block`

    Args:
      linear_algebra_building_block: A linear_algebra.x_axis_two_angles.

    Returns:
      tuple: The resolved parameters of `linear_algebra_building_block`

    """

    phase_exponent = param_resolver.value_of(linear_algebra_building_block._phase_exponent)
    exponent = param_resolver.value_of(linear_algebra_building_block._exponent)
    global_shift = linear_algebra_building_block._global_shift
    return phase_exponent, exponent, global_shift


def _resolve_phased_iswappow_params(
    linear_algebra_building_block: linear_algebra.imaginary_swap_two_angles, param_resolver: linear_algebra.ParamResolver
) -> Tuple[float, float, float, Sequence[Tuple[float, cw.ComplexDeviceArray]]]:
    """
    Resolve the parameters of `linear_algebra_building_block`

    Args:
      linear_algebra_building_block: A linear_algebra.imaginary_swap_two_angles.

    Returns:
      tuple: The resolved parameters of `linear_algebra_building_block`

    """

    phase_exponent = param_resolver.value_of(linear_algebra_building_block._phase_exponent)
    exponent = param_resolver.value_of(linear_algebra_building_block._exponent)
    global_shift = linear_algebra_building_block._global_shift
    iswap_eigen_components = (
        convert_linear_algebra_eigen_components_to_complex_device_array(
            linear_algebra_building_block._iswap._eigen_components()
        )
    )
    return phase_exponent, exponent, global_shift, iswap_eigen_components


def _resolve_fsimbuilding_block_params(
    linear_algebra_building_block: linear_algebra.rotate_on_xy_plane, param_resolver: linear_algebra.ParamResolver
) -> Tuple[float, float]:
    """
    Resolve the parameters of `linear_algebra_building_block`

    Args:
      linear_algebra_building_block: A linear_algebra.rotate_on_xy_plane.

    Returns:
      tuple: The resolved parameters of `linear_algebra_building_block`

    """
    theta = param_resolver.value_of(linear_algebra_building_block.theta)
    phi = param_resolver.value_of(linear_algebra_building_block.phi)
    return theta, phi


@partial(jax.jit, backend=JAX_PREPRO_BACKEND)
def _eigenbuilding_block_unitary_(
    exponent: float,
    global_shift: float,
    eigen_components: Sequence[Tuple[float, cw.ComplexDeviceArray]],
) -> cw.ComplexDeviceArray:
    """
    Compute the unitary matrix representation of a building_block with type in
    SIMPLE_BUILDING_BLOCKS.
    LinearAlgebra uses an eigen-building_block decomposition of building_blocks into a set of
    (float, np.ndarray(..., dtype=complex) pairs. The first entry in
    each tuple is the eigenvalue_exponent_factor, and is always real.
    The second is a complex prjector into the eigenspace to eigenvalue
    exp(1j * eigenvalue_exponent_factor).

    Args:
      exponent: The resolved value of the building_block._exponent attribute if the linear_algebra building_block.
      global_shift: The value of the building_block._global_shift attribute if the linear_algebra building_block.
      eigen_components: The return value of building_block._eigen_components(),
        with real and imaginary parts of arrays decomposed into
        ComplexDeviceArray.

    Returns:
      Tuple[jnp.DeviceArray]: Real and imaginary part of the
        unitary matrix representation.
    """
    summands = []
    for half_turn, component in eigen_components:
        # NOTE : half_turn is a float
        complex_coefficient = cw.ComplexDeviceArray(
            jnp.cos(
                jnp.pi * (half_turn + global_shift) * jnp.float64(exponent)
            ),
            jnp.sin(
                jnp.pi * (half_turn + global_shift) * jnp.float64(exponent)
            ),
        )
        summand = complex_coefficient * component
        summands.append(summand)
    result = summands[0]
    for s in summands[1:]:
        result += s
    return result


@partial(jax.jit, backend=JAX_PREPRO_BACKEND)
def _phased_xpow_unitary_(
    phase_exponent: float, exponent: float, global_shift: float
) -> cw.ComplexDeviceArray:
    """
    Compute the unitary matrix representation of a linear_algebra.x_axis_two_angles.

    Args:
      phase_exponent: The resolved value of the building_block._phase_exponent
        attribute if the linear_algebra building_block.
      exponent: The resolved value of the building_block._exponent attribute
        if the linear_algebra building_block.
      global_shift: The value of the building_block._global_shift attribute
        if the linear_algebra building_block.

    Returns:
      Tuple[jnp.DeviceArray]: Real and imaginary part of the
        unitary matrix representation.
    """
    z_eigen_components = convert_linear_algebra_eigen_components_to_complex_device_array(
        linear_algebra.flip_z_axis._eigen_components()
    )
    x_eigen_components = convert_linear_algebra_eigen_components_to_complex_device_array(
        linear_algebra.flip_x_axis._eigen_components()
    )

    z = _eigenbuilding_block_unitary_(phase_exponent, 0.0, z_eigen_components)
    x = _eigenbuilding_block_unitary_(exponent, 0.0, x_eigen_components)
    p = cw.ComplexDeviceArray(
        jnp.cos(jnp.pi * global_shift * exponent),
        jnp.sin(jnp.pi * global_shift * exponent),
    )
    result = (
        cw.dot(
            cw.dot(z, x, precision=jax.lax.Precision.HIGHEST),
            cw.conj(z),
            precision=jax.lax.Precision.HIGHEST,
        )
        * p
    )
    return result


@partial(jax.jit, backend=JAX_PREPRO_BACKEND)
def _phased_iswappow_eigen_components(
    phase_exponent: float,
    iswap_eigen_components: Sequence[Tuple[float, cw.ComplexDeviceArray]],
) -> Sequence[Tuple[float, cw.ComplexDeviceArray]]:
    """
    Compute the eigen components of the linear_algebra.imaginary_swap_two_angles.

    Args:
      phase_exponent: The resolved value of the building_block._phase_exponent
        attribute if the linear_algebra building_block.
      iswap_components: The return value of building_block._iswap._eigen_components(),
        with real and imaginary parts of arrays decomposed into
        ComplexDeviceArray.

    Returns:
      Sequence[Tuple[float, ComplexDeviceArray]]: eigen components
        of the linear_algebra.imaginary_swap_two_angles.
    """

    phase = cw.ComplexDeviceArray(
        jnp.cos(jnp.pi * phase_exponent), jnp.sin(jnp.pi * phase_exponent)
    )
    phase_matrix_real = jnp.diag(
        jnp.array([1, phase.real, phase.conj().real, 1])
    )
    phase_matrix_imag = jnp.diag(
        jnp.array([0, phase.imag, phase.conj().imag, 0])
    )
    phase_matrix = cw.ComplexDeviceArray(phase_matrix_real, phase_matrix_imag)
    inverse_phase_matrix = cw.conj(phase_matrix)
    eigen_components = []
    for eigenvalue, projector in iswap_eigen_components:
        # NOTE  eigenvalue is a float
        new_projector = phase_matrix @ projector @ inverse_phase_matrix
        eigen_components.append((eigenvalue, new_projector))
    return eigen_components


@partial(jax.jit, backend=JAX_PREPRO_BACKEND)
def _phased_iswappow_unitary_(
    phase_exponent: float,
    exponent: float,
    global_shift: float,
    iswap_eigen_components: Sequence[Tuple[float, cw.ComplexDeviceArray]],
) -> cw.ComplexDeviceArray:
    """
    Compute the unitary matrix representation of a linear_algebra.imaginary_swap_two_angles.

    Args:
      phase_exponent: The resolved value of the building_block._phase_exponent attribute
        if the linear_algebra building_block.
      exponent: The resolved value of the building_block._exponent attribute
        if the linear_algebra building_block.
      global_shift: The value of the building_block._global_shift attribute if
        the linear_algebra building_block.
      iswap_components: The return value of linear_algebra.imaginary_swap_two_angles
        (to be passed into `_phased_iswappow_eigen_components`),
        with real and imaginary parts of arrays decomposed into
        ComplexDeviceArray.

    Returns:
      Tuple[jnp.DeviceArray]: Real and imaginary part of the
        unitary matrix representation.
    """

    eigen_components = _phased_iswappow_eigen_components(
        phase_exponent, iswap_eigen_components
    )
    return _eigenbuilding_block_unitary_(exponent, global_shift, eigen_components)


@partial(jax.jit, backend=JAX_PREPRO_BACKEND)
def _fsimbuilding_block_unitary_(theta, phi) -> cw.ComplexDeviceArray:
    """
    Compute the unitary matrix representation of a linear_algebra.rotate_on_xy_plane.

    Args:
      theta: The resolved value of the building_block.theta attribute
        of the linear_algebra building_block.
      phi: The resolved value of the building_block.phi attribute
        of the linear_algebra building_block.

    Returns:
      Tuple[jnp.DeviceArray]: Real and imaginary part of the
        unitary matrix representation.
    """

    a = jnp.cos(theta)
    b = -jnp.sin(theta)
    c_real = jnp.cos(phi)
    c_imag = -jnp.sin(phi)
    real_part = jnp.array(
        [
            [1, 0, 0, 0],
            [0, a, 0, 0],
            [0, 0, a, 0],
            [0, 0, 0, c_real],
        ]
    )
    imag_part = jnp.array(
        [
            [0, 0, 0, 0],
            [0, 0, b, 0],
            [0, b, 0, 0],
            [0, 0, 0, c_imag],
        ]
    )
    return cw.ComplexDeviceArray(real_part, imag_part)


def _resolve_building_block_params(
    linear_algebra_building_block: linear_algebra.Gate, param_resolver: linear_algebra.ParamResolver
):
    """
    Resolve the parameteres of `linear_algebra_building_block`.

    Args:
      linear_algebra_building_block: One of the supported building_block types.
      param_resolver: A linear_algebra.ParamResolver which resolves the parameters
       of `linear_algebra_building_block`.

    Returns:
      The exact return type varies depending on the passed building_block type.
      The unpacked returned value is typically passed to the returned
      value of `_building_block_unitary_getter_`.

    """
    symbols = linear_algebra.parameter_symbols(linear_algebra_building_block)  # pylint: disable=no-member
    if isinstance(linear_algebra_building_block, tuple(SIMPLE_BUILDING_BLOCKS)):
        if len(symbols) > 1:
            raise ValueError(
                f"{type(linear_algebra_building_block)} can only have a single parameter "
                f"dependence. Found dependence on {symbols}."
            )
        return _resolve_eigenbuilding_block_params(linear_algebra_building_block, param_resolver)
    if isinstance(linear_algebra_building_block, linear_algebra.x_axis_two_angles):
        return _resolve_phased_xpow_params(linear_algebra_building_block, param_resolver)
    if isinstance(linear_algebra_building_block, linear_algebra.imaginary_swap_two_angles):
        return _resolve_phased_iswappow_params(linear_algebra_building_block, param_resolver)
    if isinstance(linear_algebra_building_block, linear_algebra.rotate_on_xy_plane):
        return _resolve_fsimbuilding_block_params(linear_algebra_building_block, param_resolver)
    if isinstance(linear_algebra_building_block, linear_algebra.EmptyBuildingBlock):
        return [linear_algebra_building_block]

    raise ValueError(f"Gate of type {type(linear_algebra_building_block)} not supported.")


def _building_block_unitary_getter_(linear_algebra_building_block: linear_algebra.Gate) -> Callable:
    """
    Return a function which can be used to
    compute the unitary matrix representation of `linear_algebra_building_block`.

    Args:
      linear_algebra_building_block: One of the supported building_block types.

    Returns:
      Callable: A function which can be used to compute
        the unitary matrix representation of `linear_algebra_building_block`.
        The arguments to this function have to be obtained
        from _resolve_building_block_params.

        ```python
        args = _resolve_building_block_params(linear_algebra_building_block, resolver)
        matrix = _building_block_unitary_getter_(*args)
        ```

    """

    symbols = linear_algebra.parameter_symbols(linear_algebra_building_block)
    if isinstance(linear_algebra_building_block, tuple(SIMPLE_BUILDING_BLOCKS)):
        if len(symbols) > 1:
            raise ValueError(
                f"{type(linear_algebra_building_block)} can only have a single parameter "
                f"dependence. Found dependence on {symbols}."
            )
        return _eigenbuilding_block_unitary_

    if isinstance(linear_algebra_building_block, linear_algebra.x_axis_two_angles):
        return _phased_xpow_unitary_

    if isinstance(linear_algebra_building_block, linear_algebra.imaginary_swap_two_angles):
        return _phased_iswappow_unitary_

    if isinstance(linear_algebra_building_block, linear_algebra.rotate_on_xy_plane):
        return _fsimbuilding_block_unitary_

    if isinstance(linear_algebra_building_block, linear_algebra.EmptyBuildingBlock):
        return linear_algebra.unitary
    raise ValueError(f"Gate of type {type(linear_algebra_building_block)} not supported.")


def _get_jacfwd_argnums(
    linear_algebra_building_block: linear_algebra.Gate,
) -> Tuple[Tuple[sympy.Symbol, ...], Tuple[int, ...]]:
    """
    Compute the `argnums` argument to jax.jacfwd
    for `linear_algebra_building_block`. This is then used to compute
    the gradient of `linear_algebra_building_block` with respect to
    its parameters.

    Args:
      linear_algebra_building_block: One of the supported building_block types.

    Returns:
      Tuple[sympy.Symbol,...], Tuple[int,...]:
      The building_block parameter-symbols and the positions
      where their resolved values appear in the
      arguments to the callable returned by
      `_building_block_unitary_getter_`
    """
    try:
        assert_is_variabled(linear_algebra_building_block)
    except ValueError as err:
        raise ValueError(
            f"Gradient computation requires a "
            f"parametrized linear_algebra-building_block. The building_block "
            f"of type {type(linear_algebra_building_block)}"
            f"is not variabled."
        ) from err

    expressions = get_symbol_expressions(linear_algebra_building_block)
    # for now we only allow symbols and expressions
    # of the form symbol * constant, constant * symbol
    # symbol/constant (those are necessary to support
    # variabled rotation building_blocks). In principle
    # the code should work for arbitrary expressions
    # but I haven't tested it extensively.
    for expression in expressions:
        try:
            assert_is_allowed_expression(expression)
        except ValueError as err:
            raise ValueError(
                f"Gradient computation not supported for "
                f"expression {expression}"
            ) from err

    if isinstance(linear_algebra_building_block, tuple(SIMPLE_BUILDING_BLOCKS)):
        if len(expressions) > 1:
            raise ValueError(
                f"{type(linear_algebra_building_block)} can only have a single parameter "
                f"dependence. Found dependence on {expressions}."
            )
        expressions = (linear_algebra_building_block._exponent,)
        argnums = (0,)

    elif isinstance(linear_algebra_building_block, (linear_algebra.x_axis_two_angles, linear_algebra.imaginary_swap_two_angles)):
        if len(expressions) > 2:
            raise ValueError(
                f"{type(linear_algebra_building_block)} can only have two parameter "
                f"dependences. Found dependence on {expressions}."
            )

        expressions = tuple()
        argnums = tuple()
        # we allow at most two different parameters
        # the order of elements in argnums and symbols matters
        # and has to match the order required by _phased_xpow_unitary
        if isinstance(linear_algebra_building_block._phase_exponent, sympy.Basic):
            argnums += (0,)
            expressions += (linear_algebra_building_block._phase_exponent,)

        if isinstance(linear_algebra_building_block._exponent, sympy.Basic):
            argnums += (1,)
            expressions += (linear_algebra_building_block._exponent,)

    elif isinstance(linear_algebra_building_block, linear_algebra.rotate_on_xy_plane):
        if len(expressions) > 2:
            raise ValueError(
                f"{type(linear_algebra_building_block)} can only have two parameter "
                f"dependences. Found dependence on {expressions}."
            )

        expressions = tuple()
        argnums = tuple()
        # we allow at most two different parameters
        # the order of elements in argnums and symbols matters
        # and has to match the order required by _phased_xpow_unitary
        if isinstance(linear_algebra_building_block.theta, sympy.Basic):
            argnums += (0,)
            expressions += (linear_algebra_building_block.theta,)

        if isinstance(linear_algebra_building_block.phi, sympy.Basic):
            argnums += (1,)
            expressions += (linear_algebra_building_block.phi,)

    else:
        raise ValueError(
            f"gradient computation for building_block of type {type(linear_algebra_building_block)} "
            f"not supported."
        )

    return expressions, argnums


def get_unitary(
    linear_algebra_building_block: linear_algebra.Gate, param_resolver: linear_algebra.ParamResolver
) -> jnp.DeviceArray:
    """
    Compute the unitary matrix representation of `linear_algebra_building_block`.

    Args:
      linear_algebra_building_block: One of the supported building_block types.
      param_resolver: A linear_algebra.ParamResolver which resolves the parameters
       of `linear_algebra_building_block`.

    Returns:
      jnp.DeviceArray: The unitary matrix representation of `linear_algebra_building_block`.

    """
    if not linear_algebra.is_variabled(linear_algebra_building_block):
        return np.asarray(linear_algebra.unitary(linear_algebra_building_block))

    if not isinstance(linear_algebra_building_block, tuple(SUPPORTED_PARAMETERIZABLE_GATE_TYPES)):
        raise TypeError(
            f"building_block of type {type(linear_algebra_building_block)} is" f" currently not supported"
        )
    # grab the function responsible for computing the unitary
    fun = _building_block_unitary_getter_(linear_algebra_building_block)
    # grab the parameters to fun
    args = _resolve_building_block_params(linear_algebra_building_block, param_resolver)
    # compute the unitary
    unitary = fun(*args)
    return np.array(unitary.real) + 1.0j * np.array(
        unitary.imag
    )  # cast to numpy array


@partial(jax.jit, static_argnums=(0, 1), backend=JAX_PREPRO_BACKEND)
def jacobian(
    fun: Callable, argnums: Tuple[int, ...], args: Any
) -> Dict[sympy.Symbol, jnp.DeviceArray]:
    return jax.jacfwd(fun, argnums)(*args)


def is_variabled(linear_algebra_building_block) -> bool:
    """
    Check if `linear_algebra_building_block` is variabled.
    For certain cases `linear_algebra.is_partameterized` insufficient because
    it returns True if the building_block depens only on constant symbols.

    Args:
      linear_algebra_building_block: A linear_algebra building_block.

    Returns:
      bool: Whether or not the building_block is variabled.
    """
    if linear_algebra.is_variabled(linear_algebra_building_block):
        expressions = linear_algebra.parameter_symbols(linear_algebra_building_block)
        for exp in expressions:
            if len(exp.free_symbols) > 0:
                return True
    return False


def get_autodiff_gradient(
    linear_algebra_building_block: linear_algebra.Gate, param_resolver: linear_algebra.ParamResolver
) -> Dict[sympy.Symbol, jnp.DeviceArray]:
    """
    Compute the gradient of the unitary matrix representation of `linear_algebra_building_block`
    with respect to its parameter-symbols.

    Args:
      linear_algebra_building_block: One of the supported building_block types.
      param_resolver: A linear_algebra.ParamResolver which resolves the parameters
       of `linear_algebra_building_block`.

    Returns:
      dict[sympy.Symbol, jnp.DeviceArray]: A dictionary mapping each of the
        symbols on which `linear_algebra_building_block` depends to the derivative of `linear_algebra_building_block`
        with respect to that symbol.
    """
    if (
        not is_variabled(linear_algebra_building_block)
        or len(linear_algebra.parameter_symbols(linear_algebra_building_block)) == 0
    ):
        return dict()

    if not isinstance(linear_algebra_building_block, tuple(SUPPORTED_PARAMETERIZABLE_GATE_TYPES)):
        raise TypeError(
            f"building_block of type {type(linear_algebra_building_block)} is" f" currently not supported"
        )

    # grab the function responsible for computing the unitary
    fun = _building_block_unitary_getter_(linear_algebra_building_block)
    # grab the parameters to fun
    args = _resolve_building_block_params(linear_algebra_building_block, param_resolver)
    # grab the the building_block's symbols and the argnums needed to compute the gradient
    expressions, argnums = _get_jacfwd_argnums(linear_algebra_building_block)
    free_symbols = set()
    for exp in expressions:
        free_symbols |= exp.free_symbols
    derivs = jacobian(fun, argnums, args)
    outer_derivatives = []
    for n in range(len(argnums)):
        outer_derivatives.append(
            np.array(derivs[n].real) + 1.0j * np.array(derivs[n].imag)
        )

    derivatives = {}
    for symbol in free_symbols:
        for n, expr in enumerate(expressions):
            deriv = param_resolver.value_of(sympy.diff(expr, symbol))
            if not isinstance(deriv, numbers.Number):
                raise ValueError(
                    f"derivative for expression {expr} evaluates "
                    f"to {deriv} instead of a numerical constant."
                )
            if symbol in derivatives:
                derivatives[symbol] += outer_derivatives[n] * deriv
            else:
                derivatives[symbol] = outer_derivatives[n] * deriv

    return derivatives


def get_finitediff_gradient(linear_algebra_building_block, param_resolver, eps=1e-6):
    """
    Compute the gradient of the unitary matrix representation of `linear_algebra_building_block`
    using a numerical derivative.

    Args:
      linear_algebra_building_block: One of the supported building_block types.
      param_resolver: A linear_algebra.ParamResolver which resolves the parameters
       of `linear_algebra_building_block`.

    Returns:
      dict[Text, jnp.DeviceArray]: A dictionary mapping each of the
        parameter names on which `linear_algebra_building_block` depends to the derivative
        of `linear_algebra_building_block` with respect to that symbol.
    """

    params = list(linear_algebra_building_block._parameter_names_())
    derivatives = {}
    for p in params:
        resolver = {
            k: v for k, v in param_resolver.param_dict.items()
        }  # pylint: disable=unnecessary-comprehension
        resolver[p] = resolver[p] + eps
        G0 = linear_algebra.unitary(linear_algebra.resolve_parameters(linear_algebra_building_block, param_resolver))
        G1 = linear_algebra.unitary(linear_algebra.resolve_parameters(linear_algebra_building_block, resolver))
        derivatives[p] = (G1 - G0) / eps
    return derivatives


def _parse_pbaxisum(
    pbaxisum: linear_algebra.ProbBasisAxisSum, discretes: Sequence[linear_algebra.Qid], dtype=np.complex64
):
    """
    Parse a linear_algebra.ProbBasisAxisSum operators.

    Args:
      pbaxisum: A ProbBasisAxisSum operator.
      discretes: The ordered linear_algebra.Discreteds of the full system.

    Returns:
      List[List[np.ndarray]]: The matrix building_blocks. Each entry in the list
        represents a ProbBasisAxisString as a list of np.ndarrays.
      List[np.ndarray]: The coefficients of the ProbBasisAxiss strings, each
        coefficient packed into a np.ndarray

      List[List[Tuple[int]]]: Operating axes, discrete index on which each building_block
        of the ProbBasisAxisSum representation acts.
    """
    all_building_blocks = []
    all_coefficients = []
    all_oplabels = []
    resolver = linear_algebra.ParamResolver({})
    for string in pbaxisum:
        acyclic_graph = linear_algebra.Graph()
        for discrete, building_block in string.items():
            acyclic_graph += [building_block(discrete)]

        building_blocks, _, oplabels = parse(acyclic_graph, discretes, resolver, dtype, quiet=True)
        all_building_blocks.append(building_blocks)
        all_coefficients.append(np.array([string.coefficient]))
        all_oplabels.append(oplabels)
    return all_building_blocks, all_coefficients, all_oplabels
