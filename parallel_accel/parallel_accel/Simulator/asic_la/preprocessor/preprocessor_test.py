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
import pytest
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import sympy
import linear_algebra
import graph_helper_tool as tn
from linear_algebra.study import ParamResolver
from asic_la.preprocessor.preprocessor import preprocess, preprocess_pbaxisums
from asic_la.testutils import (
    apply_supermatrices,
    build_random_acyclic_graph,
    compute_gradients,
    finite_diff_gradients,
    get_full_matrices_from_supergradient,
    get_full_matrix_from_supermatrix,
)
from asic_la.parser import parse, parse_pbaxisums
from asic_la.sharded_probability_function.utils import invert_permutation

@pytest.mark.parametrize('Nparams, Nexponents', [(10, 10), (10, 0)])
@pytest.mark.parametrize('N, max_discrete_support, depth', ([(4, 2, 20), (4, 4, 20),
                                                       (4, 2, 30), (4, 4, 30),
                                                       (7, 4, 20), (7, 7, 20),
                                                       (7, 4, 30), (7, 7, 30)]))
@pytest.mark.parametrize('seed', np.random.randint(0, 100000, 10))
def test_preprocess_state_unfolding(Nparams, Nexponents, depth, N,
                                    max_discrete_support, seed):
  print(Nparams, Nexponents, depth, N, max_discrete_support, seed)

  acyclic_graph, discretes, resolver = build_random_acyclic_graph(Nparams, Nexponents, depth,
                                                   N, seed)
  resolved_acyclic_graph = linear_algebra.resolve_parameters(acyclic_graph, resolver)
  parsed_building_blocks, parsed_gradients, inds_per_building_block = parse(
      acyclic_graph, discretes, resolver)
  supermatrices, _, superlabels = preprocess(
      parsed_building_blocks,
      parsed_gradients,
      inds_per_building_block,
      N,
      max_discrete_support=max_discrete_support)

  state = np.random.rand(*(2,) * N)
  state /= np.linalg.norm(state)
  actual_final_state = apply_supermatrices(state, tuple(range(N)),
                                                    supermatrices, superlabels)

  simulator = linear_algebra.Simulator(dtype=np.complex128)
  linear_algebra_result = simulator.simulate(resolved_acyclic_graph,
                                   discrete_order=discretes,
                                   initial_state=state.ravel())
  expected_final_state = linear_algebra_result.final_state_vector

  np.testing.assert_allclose(actual_final_state.ravel(),
                             expected_final_state,
                             atol=1E-10,
                             rtol=1)


@pytest.mark.parametrize('Nparams, Nexponents', [(10, 10), (10, 0)])
@pytest.mark.parametrize('N, depth', [(4, 20), (4, 30), (7, 20), (7, 30)])
@pytest.mark.parametrize('seed', np.random.randint(0, 100000, 10))
def test_preprocess_unitary_matrix(Nparams, Nexponents, depth, N, seed):
  acyclic_graph, discretes, resolver = build_random_acyclic_graph(Nparams, Nexponents, depth,
                                                   N, seed)
  resolved_acyclic_graph = linear_algebra.resolve_parameters(acyclic_graph, resolver)
  parsed_building_blocks, parsed_gradients, inds_per_building_block = parse(
      acyclic_graph, discretes, resolver)

  supermatrices, supergradients, axes = preprocess(
      parsed_building_blocks, parsed_gradients, inds_per_building_block, N, max_discrete_support=N)
  actual = get_full_matrix_from_supermatrix(supermatrices[0], axes[0])
  expected = linear_algebra.unitary(resolved_acyclic_graph)
  np.testing.assert_allclose(actual, expected, atol=1E-10, rtol=1)

  eps = 1E-8
  expected_gradients = finite_diff_gradients(acyclic_graph, resolver, eps)
  actual_gradients = get_full_matrices_from_supergradient(
      supergradients[0], axes[0])
  for name, actual_gradient in actual_gradients.items():
    np.testing.assert_allclose(expected_gradients[name],
                               actual_gradient,
                               atol=1E-8,
                               rtol=1)


@pytest.mark.parametrize('Nparams, Nexponents', [(10, 10), (10, 0)])
@pytest.mark.parametrize('N, max_discrete_support, depth', ([(4, 2, 20), (4, 4, 20),
                                                       (4, 2, 30), (4, 4, 30),
                                                       (7, 4, 20), (7, 7, 20),
                                                       (7, 4, 30), (7, 7, 30)]))
@pytest.mark.parametrize('seed', np.arange(10))
def test_forward_backward(N, max_discrete_support, depth, Nparams, Nexponents, seed):
  np.random.seed(seed)

  def run_forward_backward(state, supermatrices, super_opaxes):
    psi = apply_supermatrices(state, tuple(range(state.ndim)),
                                       supermatrices, super_opaxes)
    reversed_super_opaxes = list(reversed(super_opaxes))
    reversed_supermatrices = [s.T.conj() for s in reversed(supermatrices)]
    final = apply_supermatrices(psi, tuple(range(state.ndim)),
                                         reversed_supermatrices,
                                         reversed_super_opaxes)
    return final

  rtol = 1e-5
  atol = 1e-5
  state = np.random.rand(*(2,) * N)
  state /= np.linalg.norm(state)
  acyclic_graph, discretes, resolver = build_random_acyclic_graph(Nparams=Nparams,
                                                   Nexponents=Nexponents,
                                                   depth=depth,
                                                   N=N)
  building_blocks, gradients, op_axes = parse(linear_algebra.resolve_parameters(acyclic_graph, resolver),
                                    discretes,
                                    dtype=np.complex128)
  supermatrices, _, super_mataxes = preprocess(
      building_blocks, gradients, op_axes, N, max_discrete_support=max_discrete_support)
  state2 = run_forward_backward(state, supermatrices, super_mataxes)
  np.testing.assert_allclose(state, state2, atol=atol, rtol=rtol)


def compute_numerical_gradients(state,
                                acyclic_graph,
                                discretes,
                                resolver,
                                observables,
                                observables_axes,
                                eps=1E-8):
  num_discretes = len(discretes)
  grads = finite_diff_gradients(acyclic_graph, resolver, eps)
  unitary = linear_algebra.unitary(linear_algebra.resolve_parameters(acyclic_graph, resolver))
  final_state = (unitary @ state.ravel()).reshape((2,) * num_discretes)

  psi = np.zeros(state.shape, final_state.dtype)
  state_labels = tuple(range(num_discretes))
  for ob, ob_labels in list(zip(observables, observables_axes)):
    inds = [state_labels.index(l) for l in ob_labels]
    cont_state_labels = list(range(-1, -len(state_labels) - 1, -1))
    cont_ob_labels = []
    for n, i in enumerate(inds):
      cont_ob_labels.append(cont_state_labels[i])
      cont_state_labels[i] = ob_labels[n] + 1
    shape = (2,) * (2 * len(ob_labels))
    psi += tn.ncon([final_state, ob.reshape(shape)], [
        tuple(cont_state_labels),
        tuple([o + 1 for o in ob_labels]) + tuple(cont_ob_labels)
    ])
  psi = psi.ravel()
  state = state.ravel()
  return {s: np.dot(psi.conj(), g @ state) for s, g in grads.items()}


H0 = np.random.rand(4, 4).astype(np.complex128)
H0 += H0.T
H0 = H0.reshape(2, 2, 2, 2)

H1 = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
H1 += H1.T.conj()
H1 = H1.reshape(2, 2, 2, 2)


@pytest.mark.parametrize('Nparams, Nexponents', [(10, 10), (10, 0)])
@pytest.mark.parametrize('N, max_discrete_support, depth', ([(4, 2, 20), (4, 4, 20),
                                                       (4, 2, 30), (4, 4, 30),
                                                       (7, 4, 20), (7, 7, 20),
                                                       (7, 4, 30), (7, 7, 30)]))
@pytest.mark.parametrize('seed', np.arange(10))
@pytest.mark.parametrize('H, axes',
                         [([np.eye(4).reshape(2, 2, 2, 2)], [(0, 1)]),
                          ([H0], [(0, 1)]), ([H1], [(0, 1)]),
                          ([H0, H1], [(0, 1), (1, 2)])])
def test_gradients(H, axes, N, max_discrete_support, depth, Nparams, Nexponents,
                   seed):
  np.random.seed(seed)
  rtol = 1e-5
  atol = 1e-5
  state = np.random.rand(*(2,) * N)
  state /= np.linalg.norm(state)
  acyclic_graph, discretes, resolver = build_random_acyclic_graph(Nparams=Nparams,
                                                   Nexponents=Nexponents,
                                                   depth=depth,
                                                   N=N)
  building_blocks, gradients, op_axes = parse(acyclic_graph, discretes, resolver, dtype=np.complex128)
  tmp = preprocess(building_blocks,
                   gradients,
                   op_axes,
                   N,
                   max_discrete_support=max_discrete_support)
  supermatrices, supergradients, super_mataxes = tmp
  accumulated_gradients, state2 = compute_gradients(
      state, supermatrices, supergradients, super_mataxes, H, axes, N)

  np.testing.assert_allclose(state, state2, atol=atol, rtol=rtol)
  expected_gradients = compute_numerical_gradients(state,
                                                   acyclic_graph,
                                                   discretes,
                                                   resolver,
                                                   H,
                                                   axes,
                                                   eps=1E-8)

  assert set(expected_gradients.keys()) == set(accumulated_gradients.keys())
  for k, v in expected_gradients.items():
    assert np.abs(v - accumulated_gradients[k]) < 1E-5


@pytest.mark.parametrize('N', [4, 6, 8, 10])
@pytest.mark.parametrize('seed', np.arange(10))
def test_preprocess_pbaxisums_unitaries_single_large_block_labels(N, seed):
  np.random.seed(seed)
  discretes = linear_algebra.LinearSpace.range(N)
  pbaxis = [linear_algebra.flip_x_axis, linear_algebra.flip_y_axis, linear_algebra.flip_z_axis]
  rawstring = np.random.choice(pbaxis, N)
  coeff = np.random.rand(1)
  string = linear_algebra.ProbBasisAxisString(coeff, [s(a) for s, a in zip(rawstring, discretes)])
  prob_basis_axis_sums = [sum([string])]
  prob_basis_axis_building_blocks, prob_basis_axis_coeffs, prob_basis_axis_opaxes = parse_pbaxisums(prob_basis_axis_sums, discretes)
  supermats, superaxes = preprocess_pbaxisums(
      prob_basis_axis_building_blocks, prob_basis_axis_opaxes, num_discretes=N, max_discrete_support=N)
  res = get_full_matrix_from_supermatrix(supermats[0][0][0], superaxes[0][0][0])
  np.testing.assert_allclose(string.matrix(), res * prob_basis_axis_coeffs[0][0])
  np.testing.assert_allclose(coeff, prob_basis_axis_coeffs[0][0])
