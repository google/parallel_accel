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
"""Graphs to use as benchmarks of the ASIC simulator."""

import random
import linear_algebra
import numpy as np
import sympy

from linear_algebra.experiments.random_symplectic_acyclic_graph_generation import random_rotations_between_grid_interaction_subgraphs_acyclic_graph as rc
from linear_algebra.google import PlanarTreeGraph


def to_z_basis(prob_basis_axis_string):
  """
  Compute the unitaries that transform a linear_algebra.ProbBasisAxisString into
  a string of linear_algebra.flip_z_axis operators.

  Args:
    prob_basis_axis_string: A pauli string.

  Returns:
    U, U_dagger: Two linear_algebra.Graphs s.t. the acyclic_graph
      `U_dagger + linear_algebra.Graph(prob_basis_axis_string) + U` is a
      prob-basis-axis-string consisting entirely of prob-basis-axis-Z operators.

  """
  U = linear_algebra.Graph(prob_basis_axis_string.to_z_basis_ops())
  return U, U**-1


def to_single_z(final_z_discrete, discretes):
  """
  Compute the unitaries that would transform a ProbBasisAxis string
  `prob_basis_axis_string` that consists entirely of prob-basis-axis-Z building_blocks into
  a ProbBasisAxis string that consists of a single ProbBasisAxis-Z building_block.

  Args:
    final_z_aubit: The discrete on which the final ProbBasisAxis-Z building_block
      should act. Has to be in `discretes`
    discretes: The discretes of the prob-basis-axis-string.

  Returns:
    W, W_dagger: Two linear_algebra.Graphs s.t. the acyclic_graph
      `W_dagger + linear_algebra.Graph(prob_basis_axis_string) + W` is a
      prob-basis-axis-string with exactly one ProbBasisAxis-Z operator, and
      identities everywhere else.
  """
  assert final_z_discrete in discretes
  W = linear_algebra.Graph(
      [linear_algebra.exclusive_or(q, final_z_discrete) for q in discretes if q is not final_z_discrete])
  return W, W**-1


def exponentiate_prob_basis_axis_string(prob_basis_axis_string, coefficient=None):
  """
  Compute the exponential exp(i*coefficient * prob_basis_axis_string) of a
  prob-basis-axis-string.

  Args:
    prob_basis_axis_string: A linear_algebra.ProbBasisAxisString.
    coefficient: A real scalar factor for the exponential. If `None`
      it is assumed to be 1.0.

  Returns:
    linear_algebra.ProbBasisAxisString: The exponential exp(i*coefficient * prob_basis_axis_string)
  """
  if len(prob_basis_axis_string) == 0:
    raise ValueError("cannot exponentiate empty ProbBasisAxis-string")

  if coefficient is None:
    coefficient = 1.0
  eps = np.finfo(linear_algebra.unitary(prob_basis_axis_string.building_block[0]).dtype).eps
  assert np.abs(np.array(coefficient).imag) < eps,"coefficient has to be real"
  final_z_discrete = prob_basis_axis_string.discretes[-1]
  U, U_dagger = to_z_basis(prob_basis_axis_string)
  W, W_dagger = to_single_z(final_z_discrete, prob_basis_axis_string.discretes)
  return U + W + linear_algebra.Graph(linear_algebra.rotate_z_axis(
      -2 * coefficient)(final_z_discrete)) + W_dagger + U_dagger

def get_discretize_model_unitary(n_subgraphs, prob_basis_axis_sums, name):
  """
  Compute a parametrized linear_algebra.Graph obtained from alternating
  the exponentials of the terms in `prob_basis_axis_sums`. The full acyclic_graph
  is obtained from repeating a stack of subsubgraphs `n_subgraphs` times.
  In the following `l` ranges from 0 to `n_subgraphs-1`
  Super-subgraph `l` of full acyclic_graph consists of the following subsubgraphs
  (`N_n = len(prob_basis_axis_sums[n])`):

  subgraph_1 = exp(1j*prob_basis_axis_sum[0][0]*sympy.Symbol("phi_{name}_L{l}_H{0}")*...* exp(1j*prob_basis_axis_sum[0][-1]*sympy.Symbol("phi_{name}_L{l}_H{0}")
  subgraph_2 = exp(1j*prob_basis_axis_sum[1][0]*sympy.Symbol("phi_{name}_L{l}_H{1}")*...* exp(1j*prob_basis_axis_sum[1][-1]*sympy.Symbol("phi_{name}_L{l}_H{1}")
  ...
  subgraph_N_n = exp(1j*prob_basis_axis_sum[N_n-1][0]*sympy.Symbol("phi_{name}_L{l}_H{N_n-1}")*...* exp(1j*prob_basis_axis_sum[N_n-1][-1]*sympy.Symbol("phi_{name}_L{l}_H{N_n-1}")

  Args:
    n_subgraphs: integer representing the number of ApproximateOptimization steps.
    prob_basis_axis_sums: List of `linear_algebra.ProbBasisAxisSum`s representing the ObjectiveFns to
        exponentiate to build the acyclic_graph.
    name: string used to make symbols unique to this call.

  Returns:
    acyclic_graph: `linear_algebra.Graph` representing the variabled ApproximateOptimization proposal.
    all_symbols: Python `list` of `sympy.Symbol`s containing all the
        parameters of the acyclic_graph.
  """
  acyclic_graph = linear_algebra.Graph()
  all_symbols = []
  for subgraph in range(n_subgraphs):
    for n, prob_basis_axis_sum in enumerate(prob_basis_axis_sums):
      new_symb = sympy.Symbol("phi_{0}_L{1}_H{2}".format(name, subgraph, n))
      for prob_basis_axis_string in prob_basis_axis_sum:
        acyclic_graph += linear_algebra.Graph(
            exponentiate_prob_basis_axis_string(prob_basis_axis_string, coefficient=new_symb))
      all_symbols.append(new_symb)
  return acyclic_graph, all_symbols


def approxopt(discretes, n_subgraphs, name):
  """Build a random 2-local ApproximateOptimization acyclic_graph with symbols."""
  cost_objective_fn = linear_algebra.ProbBasisAxisSum()
  mixer_objective_fn = linear_algebra.ProbBasisAxisSum()
  h_max = 4.5
  h_min = -h_max
  for q in discretes:
    cost_objective_fn += linear_algebra.ProbBasisAxisString(
        random.uniform(h_min, h_max), linear_algebra.flip_z_axis(q))
    mixer_objective_fn += linear_algebra.ProbBasisAxisString(linear_algebra.flip_x_axis(q))
  for q0, q1 in zip(discretes[:-1], discretes[1:]):
    cost_objective_fn +=  random.uniform(h_min, h_max)*linear_algebra.flip_z_axis(q0)*linear_algebra.flip_z_axis(q1)
  superposition_acyclic_graph = linear_algebra.Graph([linear_algebra.flip_pi_over_4_axis(q) for q in discretes])
  discretize_acyclic_graph, all_symbols = get_discretize_model_unitary(
      n_subgraphs, [cost_objective_fn, mixer_objective_fn], name)
  measure_acyclic_graph = linear_algebra.Graph([linear_algebra.measure(q) for q in discretes])
  overall_acyclic_graph = superposition_acyclic_graph + discretize_acyclic_graph + measure_acyclic_graph
  return overall_acyclic_graph, all_symbols


def get_xz_rotation(q, a, b):
  """General single discrete rotation."""
  return linear_algebra.Graph(linear_algebra.flip_x_axis(q)**a, linear_algebra.flip_z_axis(q)**b)


def get_cz_exp(q0, q1, a):
  """Exponent of entangling CZ building_block."""
  return linear_algebra.Graph(linear_algebra.cond_rotate_z(exponent=a)(q0, q1))


def get_xz_rotation_subgraph(discretes, subgraph_num, name):
  """Apply single discrete rotations to all the given discretes."""
  subgraph_symbols = []
  acyclic_graph = linear_algebra.Graph()
  for n, q in enumerate(discretes):
    sx, sz = sympy.symbols("sx_{0}_{1}_{2} sz_{0}_{1}_{2}".format(
        name, subgraph_num, n))
    subgraph_symbols += [sx, sz]
    acyclic_graph += get_xz_rotation(q, sx, sz)
  return acyclic_graph, subgraph_symbols


def get_cz_exp_subgraph(discretes, subgraph_num, name):
  """Apply CZ building_blocks to all pairs of nearest-neighbor discretes."""
  subgraph_symbols = []
  acyclic_graph = linear_algebra.Graph()
  for n, (q0, q1) in enumerate(zip(discretes[::2], discretes[1::2])):
    a = sympy.symbols("sc_{0}_{1}_{2}".format(name, subgraph_num, 2 * n))
    subgraph_symbols += [a]
    acyclic_graph += get_cz_exp(q0, q1, a)
  shifted_discretes = discretes[1::]
  for n, (q0, q1) in enumerate(zip(shifted_discretes[::2], shifted_discretes[1::2])):
    a = sympy.symbols("sc_{0}_{1}_{2}".format(name, subgraph_num, 2 * n + 1))
    subgraph_symbols += [a]
    acyclic_graph += get_cz_exp(q0, q1, a)
  return acyclic_graph, subgraph_symbols


def hea(discretes, num_subgraphs, name):
  """Build hardware efficient proposal."""
  acyclic_graph = linear_algebra.Graph()
  all_symbols = []
  for subgraph_num in range(num_subgraphs):
    new_circ, new_symb = get_xz_rotation_subgraph(discretes, subgraph_num, name)
    acyclic_graph += new_circ
    all_symbols += new_symb
    if len(discretes) > 1:
      new_circ, new_symb = get_cz_exp_subgraph(discretes, subgraph_num, name)
      acyclic_graph += new_circ
      all_symbols += new_symb
  measure_acyclic_graph = linear_algebra.Graph([linear_algebra.measure(q) for q in discretes])
  overall_acyclic_graph = acyclic_graph + measure_acyclic_graph
  return overall_acyclic_graph, all_symbols


def crng_acyclic_graph(num_discretes, depth, seed):
  """Generate RCSs for CRNG similar to the beyond-classical ones.

  Args:
    n: number of discretes.
    depth: number of XEB cycles.
    seed: seed used to generate the random acyclic_graph.

  Returns:
    acyclic_graph: linear_algebra.Graph.

  Raises:
    ValueError: if n is not in the interval [26, 40].
  """

  def get_discretes(n):
    """Get the discretes used for each n.
    Args:
      n: number of discretes. It has to be 26 <= n <= 40.

    Returns:
      discretes: list of GridSpaces on PlanarTreeGraph.

    Raises:
      ValueError: if n is not in the interval [26, 40].
    """
    if not 26 <= n <= 40:
      raise ValueError("Number of discretes has to be in the interval [26, 40].")

    discretes = list(PlanarTreeGraph.discretes)

    # Get down to 40 discretes.
    for i in range(0, 5):
      discretes.remove(linear_algebra.GridSpace(i + 5, i))
    for i in range(3, 7):
      discretes.remove(linear_algebra.GridSpace(7, i))
    for i in range(4, 6):
      discretes.remove(linear_algebra.GridSpace(8, i))
    for x, y in [(2, 3), (4, 1), (5, 1)]:
      discretes.remove(linear_algebra.GridSpace(x, y))

    original_n = len(discretes)

    discretes_to_remove = [(6, 2), (3, 2), (4, 2), (5, 2), (6, 7), (6, 3), (3, 3),
                        (4, 3), (5, 3), (6, 4), (6, 6), (6, 5), (5, 4), (5, 8)]

    for i in range(original_n - n):
      x, y = discretes_to_remove[i]
      discretes.remove(linear_algebra.GridSpace(x, y))

    return discretes

  discretes = get_discretes(num_discretes)
  acyclic_graph = rc(discretes=discretes, depth=depth, seed=seed)
  return acyclic_graph
