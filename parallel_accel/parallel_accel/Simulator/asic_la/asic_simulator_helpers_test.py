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
"""Tests for asic_la.asic_simulator.
"""
import linear_algebra
import pytest
import math
import jax
import asic_la.utils as utils
from jax.config import config

config.update("jax_enable_x64", True)
import numpy as np
from asic_la import asic_simulator
import asic_la.asic_simulator_helpers as helpers
import asic_la.piecewise_pmapped_functions as ppf
import asic_la.asic_simulator_helpers_experimental as helpers_experimental
from asic_la.asic_simulator_helpers import AXIS_NAME

from asic_la.parser import parse_pbaxisums, parse
from asic_la.preprocessor.preprocessor import (
    preprocess,
    preprocess_pbaxisums,
    canonicalize_gradients,
    canonicalize_building_blocks,
)

from asic_la.sharded_probability_function import ShardedDiscretedProbabilityFunction
from asic_la.testutils import (
    build_random_acyclic_graph,
    generate_raw_pbaxistring,
    generate_pbaxisum,
    to_array,
)
from asic_la.sharded_probability_function import invert_permutation


@pytest.mark.parametrize("depth", [30])
@pytest.mark.parametrize("Nparams", [10])
@pytest.mark.parametrize("Nexponents", [10])
def test_get_final_state_in_steps(depth, Nparams, Nexponents):
    N = 21
    tar = 7
    acyclic_graph, discretes, resolver = build_random_acyclic_graph(
        Nparams=Nparams, Nexponents=Nexponents, depth=depth, N=N
    )
    resolved_acyclic_graph = linear_algebra.resolve_parameters(acyclic_graph, resolver)
    building_blocks, gradients, op_axes = parse(resolved_acyclic_graph, discretes, dtype=np.complex128)

    supermatrices, _, superaxes = preprocess(
        building_blocks, gradients, op_axes, N, max_discrete_support=tar
    )
    canonical_superaxes = utils.canonicalize_ints(superaxes)
    canonical_supermatrices = canonicalize_building_blocks(
        supermatrices, broadcasted_shape=jax.device_count()
    )

    assert len(supermatrices) > 1

    state = np.zeros(2 ** N).astype(np.complex128)
    state[0] = 1.0
    state = state.reshape((2,) * N)
    simulator = linear_algebra.Simulator(dtype=np.complex128)
    expected = simulator.simulate(
        resolved_acyclic_graph, discrete_order=discretes, initial_state=state.ravel()
    )

    asic_result = ppf.get_final_state_in_steps(
        canonical_supermatrices, canonical_superaxes, N, len(supermatrices)
    )
    assert asic_result.perm == tuple(range(N))

    actual = to_array(asic_result.concrete_tensor)
    np.testing.assert_allclose(np.ravel(actual), expected.final_state_vector)


@pytest.mark.parametrize("depth", [30])
@pytest.mark.parametrize("Nparams", [10])
@pytest.mark.parametrize("Nexponents", [10])
def test_get_final_state(depth, Nparams, Nexponents):
    N = 21
    tar = 7
    acyclic_graph, discretes, resolver = build_random_acyclic_graph(
        Nparams=Nparams, Nexponents=Nexponents, depth=depth, N=N
    )
    resolved_acyclic_graph = linear_algebra.resolve_parameters(acyclic_graph, resolver)
    building_blocks, gradients, op_axes = parse(resolved_acyclic_graph, discretes, dtype=np.complex128)

    supermatrices, _, superaxes = preprocess(
        building_blocks, gradients, op_axes, N, max_discrete_support=tar
    )
    state = np.zeros(2 ** N).astype(np.complex128)
    state[0] = 1.0
    state = state.reshape((2,) * N)
    simulator = linear_algebra.Simulator(dtype=np.complex128)
    expected = simulator.simulate(
        resolved_acyclic_graph, discrete_order=discretes, initial_state=state.ravel()
    )
    asic_result = jax.pmap(
        lambda x: helpers.get_final_state(supermatrices, superaxes, N),
        axis_name=AXIS_NAME,
    )(np.arange(jax.device_count()))
    assert asic_result.perm == tuple(range(N))

    actual = to_array(asic_result.concrete_tensor)
    np.testing.assert_allclose(np.ravel(actual), expected.final_state_vector)


@pytest.mark.parametrize("depth", [30])
@pytest.mark.parametrize("Nparams", [20])
@pytest.mark.parametrize("Nexponents", [10])
def test_apply_building_blocks(depth, Nparams, Nexponents):
    N = 21
    target = 7
    discretes = linear_algebra.LinearSpace.range(N)

    acyclic_graph, discretes, resolver = build_random_acyclic_graph(
        Nparams=Nparams, Nexponents=Nexponents, depth=depth, N=N
    )
    building_blocks, gradients, op_axes = parse(
        linear_algebra.resolve_parameters(acyclic_graph, resolver), discretes, dtype=np.complex128
    )
    supermatrices, _, superaxes = preprocess(
        building_blocks, gradients, op_axes, N, max_discrete_support=target
    )
    intermediate_state = jax.pmap(
        lambda x: helpers.get_final_state(supermatrices, superaxes, N),
        axis_name=AXIS_NAME,
    )(np.arange(jax.device_count()))
    actual_final_state = jax.pmap(
        lambda x: helpers.apply_building_blocks(x, supermatrices, superaxes).align_axes(),
        axis_name=AXIS_NAME,
    )(intermediate_state)

    state = np.zeros(2 ** N)
    state[0] = 1.0
    state /= np.linalg.norm(state)
    simulator = linear_algebra.Simulator(dtype=np.complex128)
    linear_algebra_result = simulator.simulate(
        linear_algebra.resolve_parameters(acyclic_graph + acyclic_graph, resolver),
        discrete_order=discretes,
        initial_state=state.ravel(),
    )
    expected_final_state = linear_algebra_result.final_state_vector
    np.testing.assert_allclose(
        expected_final_state, to_array(actual_final_state.concrete_tensor).ravel()
    )


@pytest.mark.parametrize("depth", [30])
@pytest.mark.parametrize("Nparams", [20])
@pytest.mark.parametrize("Nexponents", [10])
def test_apply_pbaxistring(depth, Nparams, Nexponents):
    N = 21
    target = 7
    discretes = linear_algebra.LinearSpace.range(N)

    acyclic_graph, discretes, resolver = build_random_acyclic_graph(
        Nparams=Nparams, Nexponents=Nexponents, depth=depth, N=N
    )
    building_blocks, gradients, op_axes = parse(
        linear_algebra.resolve_parameters(acyclic_graph, resolver), discretes, dtype=np.complex128
    )
    supermatrices, _, superaxes = preprocess(
        building_blocks, gradients, op_axes, N, max_discrete_support=target
    )

    intermediate_state = jax.pmap(
        lambda x: helpers.get_final_state(supermatrices, superaxes, N),
        axis_name=AXIS_NAME,
    )(np.arange(jax.device_count()))
    coeff, rawpbaxistring, prob_basis_axis_discretes = generate_raw_pbaxistring(discretes, N)
    pbaxistring = linear_algebra.ProbBasisAxisString(
        coeff, [p(q) for p, q in zip(rawpbaxistring, prob_basis_axis_discretes)]
    )
    pbaxisums = [sum([pbaxistring])]
    prob_basis_axis_building_blocks, _, prob_basis_axis_opaxes = parse_pbaxisums(pbaxisums, discretes)
    superpaulimats, superpauliaxes = preprocess_pbaxisums(
        prob_basis_axis_building_blocks, prob_basis_axis_opaxes, num_discretes=N, max_discrete_support=target
    )

    actual_final_state = jax.pmap(
        lambda x: helpers.apply_building_blocks(
            x, superpaulimats[0][0], superpauliaxes[0][0]
        ).align_axes(),
        axis_name=AXIS_NAME,
    )(intermediate_state)

    state = np.zeros(2 ** N)
    state[0] = 1.0
    state /= np.linalg.norm(state)
    simulator = linear_algebra.Simulator(dtype=np.complex128)

    # NOTE : linear_algebra.ProbBasisAxisString and + operator for linear_algebra.Graphs
    # use different logic for ordering building_blocks.
    acyclic_graph_2 = acyclic_graph + [p(q) for q, p in pbaxistring.items()]
    linear_algebra_result = simulator.simulate(
        linear_algebra.resolve_parameters(acyclic_graph_2, resolver),
        discrete_order=discretes,
        initial_state=state.ravel(),
    )
    expected_final_state = linear_algebra_result.final_state_vector
    np.testing.assert_allclose(
        expected_final_state, to_array(actual_final_state.concrete_tensor).ravel()
    )


@pytest.mark.parametrize("depth", [30])
@pytest.mark.parametrize("Nparams", [10])
@pytest.mark.parametrize("Nexponents", [10])
def test_inverse_unfolding(depth, Nparams, Nexponents):
    N = 21
    target = 7
    discretes = linear_algebra.LinearSpace.range(N)
    acyclic_graph, discretes, resolver = build_random_acyclic_graph(
        Nparams=Nparams, Nexponents=Nexponents, depth=depth, N=N
    )
    building_blocks, gradients, op_axes = parse(
        linear_algebra.resolve_parameters(acyclic_graph, resolver), discretes, dtype=np.complex128
    )
    supermatrices, _, superaxes = preprocess(
        building_blocks, gradients, op_axes, N, max_discrete_support=target
    )

    def forward_backward(building_blocks, axes, num_discretes):
        state = helpers.get_final_state(building_blocks, axes, num_discretes)
        assert state.perm == tuple(range(N))
        reversed_axes = reversed(axes)
        reversed_building_blocks = [g.T.conj() for g in reversed(building_blocks)]
        return helpers.apply_building_blocks(state, reversed_building_blocks, reversed_axes).align_axes()

    actual = jax.pmap(
        lambda x: forward_backward(supermatrices, superaxes, N), axis_name=AXIS_NAME
    )(np.arange(jax.device_count()))
    assert actual.perm == tuple(range(N))
    state = np.zeros(2 ** N)
    state[0] = 1.0
    state /= np.linalg.norm(state)
    eps = np.finfo(np.float64).eps * 100
    np.testing.assert_allclose(
        to_array(actual.concrete_tensor).ravel(), state.ravel(), atol=eps, rtol=eps
    )


@pytest.mark.parametrize("Nparams", [10])
@pytest.mark.parametrize("depth", [20])
@pytest.mark.parametrize("subdomain_length", [21])
@pytest.mark.parametrize("num_pbaxistrings", [4])
@pytest.mark.parametrize("num_pbaxisums", [1])
@pytest.mark.parametrize("string_length", [4])
@pytest.mark.parametrize("seed", [0])
def test_distributed_compute_gradients(
    Nparams,
    depth,
    subdomain_length,
    num_pbaxisums,
    num_pbaxistrings,
    string_length,
    seed,
):
    np.random.seed(seed)
    N = 21
    target = 7
    discretes = linear_algebra.LinearSpace.range(N)
    subdomain = np.sort(np.random.choice(np.arange(N), subdomain_length, replace=False))
    acyclic_graph, discretes, resolver = build_random_acyclic_graph(
        Nparams=Nparams, Nexponents=0, depth=depth, N=N, subdomain=subdomain
    )
    building_blocks, gradients, op_axes = parse(acyclic_graph, discretes, resolver, dtype=np.complex128)
    supermatrices, supergradients, superaxes = preprocess(
        building_blocks, gradients, op_axes, N, max_discrete_support=target
    )

    # canonicalize data
    canonical_superaxes = utils.canonicalize_ints(superaxes)
    canon_grads, smap = canonicalize_gradients(
        supergradients, broadcasted_shape=jax.device_count()
    )
    canon_supermats = canonicalize_building_blocks(
        supermatrices, broadcasted_shape=jax.device_count()
    )

    op_discretes = []
    for op in acyclic_graph.all_operations():
        op_discretes.extend(list(op.discretes))
    op_discretes = sorted(list(set(op_discretes)))

    pbaxisums = []
    for _ in range(num_pbaxisums):
        pbaxisums.append(generate_pbaxisum(num_pbaxistrings, op_discretes, string_length))

    prob_basis_axis_building_blocks, prob_basis_axis_coeffs, prob_basis_axis_opaxes = parse_pbaxisums(pbaxisums, discretes)
    superpaulimats, superpauliaxes = preprocess_pbaxisums(
        prob_basis_axis_building_blocks, prob_basis_axis_opaxes, num_discretes=N, max_discrete_support=target
    )

    canonical_superpauliaxes = utils.canonicalize_ints(superpauliaxes)
    canon_superpaulimats = canonicalize_building_blocks(
        superpaulimats, broadcasted_shape=jax.device_count()
    )
    canonical_prob_basis_axis_coeffs = canonicalize_building_blocks(
        prob_basis_axis_coeffs, broadcasted_shape=jax.device_count()
    )

    (
        actual_gradients,
        actual_expectations,
    ) = helpers_experimental.distributed_compute_gradients(
        canon_supermats,
        canon_grads,
        canonical_superaxes,
        canon_superpaulimats,
        canonical_superpauliaxes,
        canonical_prob_basis_axis_coeffs,
        N,
        len(smap),
    )
    simulator = linear_algebra.Simulator(dtype=np.complex128)
    linear_algebra_result = simulator.simulate(acyclic_graph, resolver)
    params = linear_algebra.parameter_symbols(acyclic_graph)
    exp_acyclic_graphs = [None] * num_pbaxisums
    g1 = []
    for m, pbaxisum in enumerate(pbaxisums):
        exp_acyclic_graphs[m] = [linear_algebra.Graph() for _ in range(num_pbaxistrings)]
        accumulator = np.zeros_like(linear_algebra_result.final_state_vector)
        for n, pbaxistring in enumerate(pbaxisum):
            exp_acyclic_graphs[m][n] += [p(q) for q, p in pbaxistring.items()]
            obs_result = simulator.simulate(
                exp_acyclic_graphs[m][n],
                discrete_order=op_discretes,
                initial_state=linear_algebra_result.final_state_vector.ravel(),
            )
            accumulator += obs_result.final_state_vector * prob_basis_axis_coeffs[m][n]
            expected_expectation = np.dot(
                linear_algebra_result.final_state_vector.conj(), accumulator
            )
        g1.append(expected_expectation)
    eps = jax.numpy.finfo(actual_expectations.dtype).eps * 100
    np.testing.assert_allclose(np.array(g1), actual_expectations[0], atol=eps, rtol=eps)
    delta = 1e-8
    g2 = {}
    for param in params:
        g2[param] = []
        shifted_dict = {k: v for k, v in resolver.param_dict.items()}
        shifted_dict[param.name] = resolver.param_dict[param.name] + delta
        shifted_resolver = linear_algebra.ParamResolver(shifted_dict)
        linear_algebra_result_shifted = simulator.simulate(acyclic_graph, shifted_resolver)
        for m, pbaxisum in enumerate(pbaxisums):
            accumulator = np.zeros_like(linear_algebra_result_shifted.final_state_vector)
            for n, pbaxistring in enumerate(pbaxisum):
                obs_result = simulator.simulate(
                    exp_acyclic_graphs[m][n],
                    discrete_order=op_discretes,
                    initial_state=linear_algebra_result_shifted.final_state_vector.ravel(),
                )
                accumulator += obs_result.final_state_vector * prob_basis_axis_coeffs[m][n]
            g2[param].append(
                np.dot(linear_algebra_result_shifted.final_state_vector.conj(), accumulator)
            )

    for s, idx in smap.items():
        for m, val in enumerate(g2[s]):
            expected = np.real((val - g1[m]) / delta)
            np.testing.assert_allclose(
                actual_gradients[idx, m], expected, atol=1e-5, rtol=1e-5
            )


def test_distributed_scalar_product():
    N = 21
    shape = (
        (jax.device_count(),)
        + (2,) * (N - int(math.log2(jax.device_count())) - 10)
        + (8, 128)
    )
    a = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    b = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    def scalar_prod(x, y):
        state1 = ShardedDiscretedProbabilityFunction(x, list(range(N)))
        state2 = ShardedDiscretedProbabilityFunction(y, list(range(N)))
        return helpers.scalar_product_real(state1, state2)

    actual = jax.pmap(scalar_prod, axis_name=AXIS_NAME)(a, b)
    expected = np.real(np.dot(a.ravel().conj(), b.ravel()))
    np.testing.assert_allclose(actual[0], expected)


def test_distributed_scalar_product_raises():
    N = 21
    shape = (
        (jax.device_count(),)
        + (2,) * (N - int(math.log2(jax.device_count())) - 10)
        + (8, 128)
    )
    a = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    b = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    def scalar_prod(x, y):
        state1 = ShardedDiscretedProbabilityFunction(x, list(range(N)))
        state2 = ShardedDiscretedProbabilityFunction(y, list(range(N - 1, -1, -1)))
        return helpers.scalar_product_real(state1, state2)

    with pytest.raises(AssertionError, match="state1 and state2"):
        actual = jax.pmap(scalar_prod, axis_name=AXIS_NAME)(a, b)
