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
"""Benchmarks for state unfolding on a variety of standard acyclic_graphs."""
import gc
from sys import stdout

import linear_algebra
from linear_algebra.testing import random_acyclic_graph
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.utils import mpl_style
from benchmarks.utils.benchmark import BenchmarkSuite
from benchmarks.utils.jax_benchmark import JaxBenchmark
from asic_la.sharded_probability_function import ShardedDiscretedProbabilityFunction
from asic_la import utils, parser
from asic_la.preprocessor import preprocessor
import asic_la.asic_simulator_helpers as helpers


class CustomGate(linear_algebra.Gate):

  def __init__(self, unitary):
    super().__init__()
    self.my_unitary = unitary

  def _num_discretes_(self):
    return int(np.log2(self.my_unitary.shape[0]))

  def _unitary_(self):
    return self.my_unitary

  def _acyclic_graph_diagram_info_(self, args):
    return "O", "O"


def to_array(arr):
  return np.array(arr.real) + 1j * np.array(arr.imag)


def add_random_exponents(acyclic_graph, seed=0):
  np.random.seed(seed)
  all_ops = list(acyclic_graph.all_operations())
  params = (np.random.rand(len(all_ops)) - 0.5) * 4
  return linear_algebra.Graph([o**p for o, p in zip(all_ops, params)])


def amplitudes_from_acyclic_graphs(acyclic_graphs, num_amplitudes, dtype=np.complex128):
  """
  Sample bitstrings and their amplitudes of a acyclic_graph consisting entirely of
  single-discrete building_blocks.

  Args:
    acyclic_graphs: List of single-discrete acyclic_graphs
    num_amplitudes: The number of amplitudes to compute.

  Returns:
    np.ndarray: The amplitudes.
    np.ndarray: The bitstrings of each amplitude.
  """
  final_states = []
  for acyclic_graph in acyclic_graphs:
    sim = linear_algebra.Simulator(dtype)
    final_states.append(
        sim.simulate(acyclic_graph,
                     discrete_order=linear_algebra.DiscretedOrder.DEFAULT).state_vector())

  # now for the bitstrings
  samples = np.stack([
      np.random.choice(
          np.arange(len(state)), size=num_amplitudes, p=np.abs(state)**2)
      for state in final_states
  ],
                     axis=0)

  # linear_algebra uses big-endian ordering to map integers to discretes, i.e.
  # the most significant bit of the integers corresponds to discrete 0 in
  # default ordering. Hence we need to jump through some hoops here to
  # get this right.
  num_sub_discretes = [len(c.all_discretes()) for c in acyclic_graphs]
  shifts = list(reversed(np.cumsum(list(reversed(num_sub_discretes[1:] + [0])))))
  bitstrings = np.sum(np.stack(
      [samples[n, :] << shift for n, shift in enumerate(shifts)], axis=0),
                      axis=0)
  return np.prod(np.stack(
      [vec[samples[n]] for n, vec in enumerate(final_states)], axis=0),
                 axis=0), bitstrings


def amplitudes_from_states(states, num_amplitudes):
  """
  Sample bitstrings and their amplitudes of a probabilityfunction given by
  the outer product of N smaller probabilityfunctions.

  Args:
    states: List of states.
    num_amplitudes: The number of amplitudes to compute.

  Returns:
    np.ndarray: The amplitudes.
    np.ndarray: The bitstrings of each amplitude.
  """
  # now for the bitstrings
  samples = np.stack([
      np.random.choice(
          np.arange(len(state)), size=num_amplitudes, p=np.abs(state)**2)
      for state in states
  ],
                     axis=0)
  # linear_algebra uses big-endian ordering to map integers to discretes, i.e.
  # the most significant bit of the integers corresponds to discrete 0 in
  # default ordering. Hence we need to jump through some hoops here to
  # get this right.
  num_sub_discretes = [int(round(np.log2(np.prod(s.shape)))) for s in states]
  shifts = list(reversed(np.cumsum(list(reversed(num_sub_discretes[1:] + [0])))))
  bitstrings = np.sum(np.stack(
      [samples[n, :] << shift for n, shift in enumerate(shifts)], axis=0),
                      axis=0)
  return np.prod(np.stack([vec[samples[n]] for n, vec in enumerate(states)],
                          axis=0),
                 axis=0), bitstrings


def add_even_odd_subgraphs(discretes, seed=0):
  np.random.seed(seed)
  acyclic_graph = linear_algebra.Graph()
  for n in range(0, len(discretes) - 1, 2):
    q1 = discretes[n]
    q2 = discretes[n + 1]
    mat = np.random.rand(4, 4) - 0.5 + 1j * (np.random.rand(4, 4) - 0.5)
    building_block = CustomGate(np.linalg.qr(mat)[0])
    acyclic_graph.append(building_block(q1, q2))
  for n in range(1, len(discretes) - 1, 2):
    q1 = discretes[n]
    q2 = discretes[n + 1]
    mat = np.random.rand(4, 4) - 0.5 + 1j * (np.random.rand(4, 4) - 0.5)
    building_block = CustomGate(np.linalg.qr(mat)[0])
    acyclic_graph.append(building_block(q1, q2))
  return acyclic_graph


def _context_helper(subgraph):
  """
  Helper function which produces pmapped functions.
  Args:
    subgraph: A linear_algebra.Graph.
  """
  discretes = linear_algebra.DiscretedOrder.DEFAULT.order_for(subgraph.all_discretes())
  building_blocks, gradients, op_axes = parser.parse(subgraph, discretes)
  supermatrices, _, superaxes = preprocessor.preprocess(building_blocks, gradients,
                                                        op_axes, len(discretes))
  canonical_supermatrices = preprocessor.canonicalize_building_blocks(
      supermatrices, broadcasted_shape=jax.local_device_count())
  canonical_superaxes = utils.canonicalize_ints(superaxes)

  @jax.partial(jax.pmap, axis_name=helpers.AXIS_NAME)
  def align_axes(state):
    return state.align_axes()

  @jax.partial(jax.pmap,
               static_broadcasted_argnums=2,
               axis_name=helpers.AXIS_NAME)
  def pmapped_apply_building_blocks_align_axes(state, building_blocks, axes):
    return helpers.apply_building_blocks(state, building_blocks, axes).align_axes()

  @jax.partial(jax.pmap,
               static_broadcasted_argnums=1,
               axis_name=helpers.AXIS_NAME)
  def initialize_asic_state(_, num_discretes):
    return ShardedDiscretedProbabilityFunction.zero_state(num_discretes,
                                               num_global_discretes=int(
                                                   np.log2(jax.device_count())))

  return (canonical_supermatrices, canonical_superaxes,
          align_axes) + (pmapped_apply_building_blocks_align_axes, initialize_asic_state)


@jax.partial(jax.pmap,
             static_broadcasted_argnums=1,
             axis_name=helpers.AXIS_NAME)
def dist_from_zero_state(state, num_discretes):
  zero_state = ShardedDiscretedProbabilityFunction.zero_state(num_discretes)
  diff = zero_state.concrete_tensor - state.concrete_tensor
  maxdiff = jnp.sqrt(jnp.max(diff.real**2 + diff.imag**2))
  return maxdiff


class ReverseEvolutionTest(JaxBenchmark):
  name = 'reverse_unfolding_test'

  @staticmethod
  def context_fn(num_discretes, n_moments, op_density, seed):
    acyclic_graph = random_acyclic_graph(num_discretes,
                             n_moments,
                             op_density,
                             random_state=seed)
    acyclic_graph = add_random_exponents(acyclic_graph, seed + 100000)
    resolver = linear_algebra.ParamResolver({})
    return acyclic_graph, resolver

  @staticmethod
  def benchmark_fn(context):
    acyclic_graph, resolver = context
    inverse_acyclic_graph = acyclic_graph**-1
    discretes = tuple(acyclic_graph.all_discretes())
    num_discretes = len(discretes)
    num_local_cores = jax.local_device_count()
    building_blocks, gradients, operating_axes = parser.parse(acyclic_graph, discretes, resolver)
    inverse_building_blocks, inverse_gradients, inverse_operating_axes = parser.parse(
        inverse_acyclic_graph, discretes, resolver)

    supermatrices, _, superaxes = preprocessor.preprocess(building_blocks,
                                                          gradients,
                                                          operating_axes,
                                                          num_discretes,
                                                          target_ndiscretes=7)
    inverse_supermatrices, _, inverse_superaxes = preprocessor.preprocess(
        inverse_building_blocks,
        inverse_gradients,
        inverse_operating_axes,
        num_discretes,
        target_ndiscretes=7)

    canonical_superaxes = utils.canonicalize_ints(superaxes)
    canonical_supermatrices = preprocessor.canonicalize_building_blocks(
        supermatrices, broadcasted_shape=num_local_cores)
    inverse_canonical_superaxes = utils.canonicalize_ints(inverse_superaxes)
    inverse_canonical_supermatrices = preprocessor.canonicalize_building_blocks(
        inverse_supermatrices, broadcasted_shape=num_local_cores)

    final_state = helpers.compute_final_state(canonical_supermatrices,
                                              canonical_superaxes, num_discretes)
    initial_state = helpers.pmapped_apply_building_blocks(
        final_state, inverse_canonical_supermatrices,
        inverse_canonical_superaxes)
    del final_state
    gc.collect()
    maxdiffs = dist_from_zero_state(initial_state, num_discretes)
    return {'max absolute difference': np.max(maxdiffs)}


class AmplitudeErrorAnalysis(JaxBenchmark):
  """
  A base class for performing amplitude error analysis for PARALLELACCEL.
  """

  @staticmethod
  def context_fn(*args, **kwargs):
    raise NotImplementedError()

  @staticmethod
  def benchmark_fn(context):
    subgraph, num_subgraphs, pace, compression_factor = context[0:4]
    threshold, canonical_supermatrices = context[4:6]
    canonical_superaxes, align_axes = context[6:8]
    pmapped_apply_building_blocks_align_axes, initialize_asic_state = context[8:10]
    discretes = linear_algebra.DiscretedOrder.DEFAULT.order_for(subgraph.all_discretes())
    num_discretes = len(discretes)

    dtype = np.complex128
    sim = linear_algebra.Simulator(dtype=dtype)
    linear_algebra_state = np.zeros(2**num_discretes, dtype)
    linear_algebra_state[0] = 1.0
    state = initialize_asic_state(np.arange(jax.local_device_count()),
                                 num_discretes)
    results = {
        'absolute error real': [],
        'absolute error imag': [],
        'relative error real': [],
        'relative error imag': [],
        'absolute error prob': [],
        'relative error prob': [],
        'amplitudes real': [],
        'amplitudes imag': [],
        'probs': []
    }

    for step in range(num_subgraphs):
      # get the acyclic_graph representation of the current moment
      stdout.write(f'\rsubgraph {step}/{num_subgraphs}')
      state = pmapped_apply_building_blocks_align_axes(state, canonical_supermatrices,
                                             canonical_superaxes)
      linear_algebra_state = sim.simulate(
          subgraph, initial_state=linear_algebra_state).state_vector().ravel()

      if (step + 1) % pace == 0:
        actual = to_array(align_axes(state).concrete_tensor).ravel()
        mask_real = np.abs(linear_algebra_state.real) > threshold
        mask_imag = np.abs(linear_algebra_state.imag) > threshold
        sortperm_real = np.argsort(
            linear_algebra_state.real[mask_real])[::compression_factor]
        sortperm_imag = np.argsort(
            linear_algebra_state.imag[mask_imag])[::compression_factor]
        results['amplitudes real'].append(actual.real[mask_real][sortperm_real])
        results['amplitudes imag'].append(actual.imag[mask_imag][sortperm_imag])
        results['absolute error real'].append(
            np.abs(actual.real[mask_real][sortperm_real] -
                   linear_algebra_state.real[mask_real][sortperm_real]))
        results['absolute error imag'].append(
            np.abs(actual.imag[mask_imag][sortperm_imag] -
                   linear_algebra_state.imag[mask_imag][sortperm_imag]))
        results['relative error real'].append(
            (np.abs(actual.real[mask_real][sortperm_real] -
                    linear_algebra_state.real[mask_real][sortperm_real]) /
             np.abs(linear_algebra_state.real[mask_real][sortperm_real])))
        results['relative error imag'].append(
            (np.abs(actual.imag[mask_imag][sortperm_imag] -
                    linear_algebra_state.imag[mask_imag][sortperm_imag]) /
             np.abs(linear_algebra_state.imag[mask_imag][sortperm_imag])))
        del actual
        del sortperm_real
        del sortperm_imag
        del mask_real
        del mask_imag
        gc.collect()

        actual = np.abs(to_array(align_axes(state).concrete_tensor).ravel())**2
        expected = np.abs(linear_algebra_state)**2
        mask = expected > threshold
        sortperm = np.argsort(expected[mask])[::compression_factor]
        results['probs'].append(actual[mask][sortperm])
        results['absolute error prob'].append(
            np.abs(actual[mask][sortperm] - expected[mask][sortperm]))
        results['relative error prob'].append(
            np.abs(actual[mask][sortperm] - expected[mask][sortperm]) /
            expected[mask][sortperm])
        del actual
        del sortperm
        del mask
        gc.collect()
    return results

  def _plot(self, dataframe, name, times=None):
    name_mapping = {
        'amplitudes real': ('absolute error real', 'relative error real'),
        'amplitudes imag': ('absolute error imag', 'relative error imag'),
        'probs': ('absolute error prob', 'relative error prob'),
    }
    abs_name, rel_name = name_mapping[name]
    plot_param_names = {'num_discretes'}
    for k, v in name_mapping.items():
      plot_param_names |= {k} | set(v)
    num_discretes = np.unique(dataframe["num_discretes"].values)

    def moving_average(x, w):
      return np.convolve(x, np.ones(w), 'valid') / w

    parameter_names = list(set(dataframe.columns.values) - plot_param_names)
    grouped = dataframe.groupby(parameter_names, as_index=False)
    for data in grouped:
      param_values = data[0]
      pace = param_values[parameter_names.index('pace')]
      subdf = data[1]
      title = ', '.join([
          f"{name} = {value}"
          for name, value in zip(parameter_names, param_values)
      ])

      fig, ax = plt.subplots(len(num_discretes), 3, figsize=(10, 20), sharex=True)
      fig.suptitle(title)
      for m, nq in enumerate(num_discretes):
        all_data = subdf[subdf["num_discretes"] == nq][name].values[0][0]
        all_rel_err = subdf[subdf["num_discretes"] == nq][rel_name].values[0][0]
        all_abs_err = subdf[subdf["num_discretes"] == nq][abs_name].values[0][0]
        if times is None:
          step = len(all_data) // 4
          times = list(range(len(all_data) - 1, -1, -step))
        if len(num_discretes) > 1:
          ax[m][0].text(0.3,
                        0.1,
                        f'# discretes = {nq}',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax[m][0].transAxes)
        else:
          ax[0].text(0.3,
                     0.1,
                     f'# discretes = {nq}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax[0].transAxes)
        time = 0
        for d, rel_err, abs_err in zip(all_data, all_rel_err, all_abs_err):
          if time in times:
            if len(num_discretes) > 1:
              ax[m][0].loglog(np.abs(d), label=f'subgraph {pace*(time+1)}')
              ax[m][1].loglog(moving_average(rel_err, 10))
              ax[m][2].loglog(moving_average(abs_err, 10))
            else:
              ax[0].loglog(np.abs(d), label=f'subgraph {pace*(time+1)}')
              ax[1].loglog(moving_average(rel_err, 10))
              ax[2].loglog(moving_average(abs_err, 10))

          time += 1
      if len(num_discretes) > 1:
        ax[-1][1].set_xlabel('index (small to large)')
        ax[len(ax) // 2][0].set_ylabel(name)
        ax[len(ax) // 2][1].set_ylabel('relative error')
        ax[len(ax) // 2][2].set_ylabel('absolute error')
        ax[0][0].legend(loc='upper left')

      else:
        ax[1].set_xlabel('index (small to large)')
        ax[0].set_ylabel(name)
        ax[1].set_ylabel('relative error')
        ax[2].set_ylabel('absolute error')
        ax[0].legend(loc='upper left')

    return fig, ax

  def get_plot(self, times=None):
    dataframe = self.results_to_dataframe()
    plot_data = []
    for name in ['probs', 'amplitudes real', 'amplitudes imag']:
      plot_data.append(self._plot(dataframe, name, times))
    return plot_data


class StochasticAmplitudeErrorAnalysis(AmplitudeErrorAnalysis):
  """
  This benchmarks analyses the error introduced by finite precision arithmetic.
  It uses special acyclic_graphs for which the amplitudes can be obtained exaclty.
  The acyclic_graph it uses is build by repeating a single subgraph of building_blocks `num_subgraphs`
  To obtain the acyclic_graph for the single subgraph, we decompose the total number of discretes
  into disjoint sets each containing `num_sub_discretes` discretes. On each disjoint set
  we build a random acyclic_graph. The amplitudes of the full acyclic_graph, i.e. the acyclic_graph
  on all discretes, is obtained from the product of the amplitudes on the disjoint subsets.

  Since computing all amplitudes is not feasible in general, we only compute amplitudes of
  the most important bitstrings by sampling `num_amplitudes` bitstrings from the exact
  distribution.
  """
  name = 'stochastic_amplitude_error_analysis'

  def results_to_dataframe(self):
    df = super().results_to_dataframe()
    df.insert(0, 'num_discretes', df['discrete_partitioning'].apply(sum))
    return df

  @staticmethod
  def context_fn(*args, **kwargs):
    raise NotImplementedError()

  @staticmethod
  def benchmark_fn(context):
    subgraph, joined_subgraph, num_subgraphs, pace = context[0:4]
    num_amplitudes, canonical_supermatrices = context[4:6]
    canonical_superaxes, _, pmapped_apply_building_blocks_align_axes = context[6:9]
    initialize_asic_state = context[9]

    pmapped_get_amplitudes = jax.pmap(helpers.get_amplitudes_from_state,
                                      axis_name=helpers.AXIS_NAME,
                                      in_axes=(0, None, None),
                                      out_axes=(None, None))

    def unfold_linear_algebra(sim, subgraph, states):
      return [
          sim.simulate(l, discrete_order=linear_algebra.DiscretedOrder.DEFAULT,
                       initial_state=s).state_vector()
          for l, s in zip(subgraph, states)
      ]

    results = {
        'absolute error real': [],
        'absolute error imag': [],
        'relative error real': [],
        'relative error imag': [],
        'absolute error prob': [],
        'relative error prob': [],
        'amplitudes real': [],
        'amplitudes imag': [],
        'probs': []
    }
    discretes = linear_algebra.DiscretedOrder.DEFAULT.order_for(joined_subgraph.all_discretes())
    num_discretes = len(discretes)

    state = initialize_asic_state(np.arange(jax.local_device_count()),
                                 num_discretes)
    dtype = np.complex128
    linear_algebra_states = [np.zeros(2**len(l.all_discretes()), dtype) for l in subgraph]
    for c in linear_algebra_states:
      c[0] = 1.0

    sim = linear_algebra.Simulator(dtype=dtype)
    for step in range(num_subgraphs):
      stdout.write(f'\rsubgraph {step}/{num_subgraphs}')
      state = pmapped_apply_building_blocks_align_axes(state, canonical_supermatrices,
                                             canonical_superaxes)
      linear_algebra_states = unfold_linear_algebra(sim, subgraph, linear_algebra_states)
      expected, bitstrings = amplitudes_from_states(linear_algebra_states, num_amplitudes)

      bitstrings = np.array(bitstrings)
      num_global = int(np.round(np.log2(jax.device_count())))
      num_local = num_discretes - num_global
      # Global discretes are always the first N discretes after axes alignment,
      # and the given bitstrings are in big endian, so we shift the bits
      # down to grab them.
      global_bitstrings = bitstrings >> num_local
      local_bitstrings = bitstrings & (2**num_local - 1)

      real, imag = pmapped_get_amplitudes(state, global_bitstrings,
                                          local_bitstrings)
      actual = np.array(real) + 1j * np.array(imag)
      actual_prob = np.abs(actual)**2
      expected_prob = np.abs(expected)**2
      if (step + 1) % pace == 0:
        sortperm_real = np.argsort(expected.real)
        sortperm_imag = np.argsort(expected.imag)
        results['amplitudes real'].append(actual.real[sortperm_real])
        results['amplitudes imag'].append(actual.imag[sortperm_imag])

        results['absolute error real'].append(
            np.abs(actual.real[sortperm_real] - expected.real[sortperm_real]))
        results['absolute error imag'].append(
            np.abs(actual.imag[sortperm_imag] - expected.imag[sortperm_imag]))

        results['relative error real'].append(
            (np.abs(actual.real[sortperm_real] - expected.real[sortperm_real]) /
             np.abs(expected.real[sortperm_real])))
        results['relative error imag'].append(
            (np.abs(actual.imag[sortperm_imag] - expected.imag[sortperm_imag]) /
             np.abs(expected.imag[sortperm_imag])))

        sortperm = np.argsort(expected_prob)
        results['probs'].append(actual_prob[sortperm])
        results['absolute error prob'].append(
            np.abs(actual_prob[sortperm] - expected_prob[sortperm]))
        results['relative error prob'].append(
            np.abs(actual_prob[sortperm] - expected_prob[sortperm]) /
            expected_prob[sortperm])
    return results

  def get_plot(self, times=None):
    dataframe = self.results_to_dataframe()
    dataframe = dataframe.drop(columns=['discrete_partitioning'])
    plot_data = []
    for name in ['probs', 'amplitudes real', 'amplitudes imag']:
      plot_data.append(self._plot(dataframe, name, times))
    return plot_data


class StochasticRandomBrickworkAmplitudeErrorAnalysis(
    StochasticAmplitudeErrorAnalysis):
  """
  This benchmarks analyses the error introduced by finite precision arithmetic.
  It uses special acyclic_graphs for which the amplitudes can be obtained exaclty.
  The acyclic_graph it uses is build by repeating a single subgraph of building_blocks `num_subgraphs`
  To obtain the acyclic_graph for the single subgraph, we decompose the total number of discretes
  into disjoint sets each containing `num_sub_discretes` discretes. On each disjoint set
  we build a random acyclic_graph build from one even and one odd subgraph of complex two-discrete
  unitaries. The amplitudes of the full acyclic_graph, i.e. the acyclic_graph
  on all discretes, is obtained from the product of the amplitudes on the disjoint subsets.

  Since computing all amplitudes is not feasible in general, we only compute amplitudes of
  the most important bitstrings by sampling `num_amplitudes` bitstrings from the exact
  distribution.
  """
  name = 'stochastic_random_brickwork_amplitude_error_analysis'

  @staticmethod
  def context_fn(num_subgraphs, discrete_partitioning, pace, num_amplitudes, seed=0):
    """
    Args:
      num_subgraphs: The number of subgraphs.
      discrete_partitioning: Tuple of int describing a partitioning of the total number
        of discretes into subsets of discretes, e.g. (8,8,7) for a total of 23 discretes
        partitioned into sets of 8, 8 and 7 discretes.
      pace: An integer denoting the "observation" pace. A pace of `1` corresponds to
        measuring the amplitudes after every subgraph.
      num_amplitudes: The number of amplitudes to compare with the exact solution.
      seed: An integer seed for acyclic_graph generation.
    """
    discrete_partitioning = list(discrete_partitioning)
    num_discretes = sum(discrete_partitioning)
    cumsum = np.cumsum([0] + discrete_partitioning)
    discretes = linear_algebra.LinearSpace.range(num_discretes)
    subgraph = [
        add_even_odd_subgraphs(discretes[cumsum[n]:cumsum[n + 1]], seed + 10000 * n)
        for n in range(len(discrete_partitioning))
    ]
    joined_subgraph = sum(subgraph, linear_algebra.Graph())
    out = _context_helper(joined_subgraph)
    return (subgraph, joined_subgraph, num_subgraphs, pace, num_amplitudes) + out


class StochasticRandomGraphAmplitudeErrorAnalysis(
    StochasticAmplitudeErrorAnalysis):
  """
  This benchmarks analyses the error introduced by finite precision arithmetic.
  It uses special acyclic_graphs for which the amplitudes can be obtained exaclty.
  The acyclic_graph it uses is build by repeating a single subgraph of building_blocks `num_subgraphs`
  To obtain the acyclic_graph for the single subgraph, we decompose the total number of discretes
  into disjoint sets each containing `num_sub_discretes` discretes. On each disjoint set
  we build a random acyclic_graph build from one even and one odd subgraph of complex two-discrete
  unitaries. The amplitudes of the full acyclic_graph, i.e. the acyclic_graph
  on all discretes, is obtained from the product of the amplitudes on the disjoint subsets.

  Since computing all amplitudes is not feasible in general, we only compute amplitudes of
  the most important bitstrings by sampling `num_amplitudes` bitstrings from the exact
  distribution.
  """
  name = 'stochastic_random_acyclic_graph_amplitude_error_analysis'

  @staticmethod
  def context_fn(n_moments_per_subgraph,
                 num_subgraphs,
                 discrete_partitioning,
                 pace,
                 num_amplitudes,
                 op_density=1.0,
                 seed=0):
    """
    Args:
      num_subgraphs: The number of subgraphs.
      discrete_partitioning: Tuple of int describing a partitioning of the total number
        of discretes into subsets of discretes, e.g. (8,8,7) for a total of 23 discretes
        partitioned into sets of 8, 8 and 7 discretes.
      pace: An integer denoting the "observation" pace. A pace of `1` corresponds to
        measuring the amplitudes after every subgraph.
      num_amplitudes: The number of amplitudes to compare with the exact solution.
      seed: An integer seed for acyclic_graph generation.
    """
    discrete_partitioning = list(discrete_partitioning)
    num_discretes = sum(discrete_partitioning)
    cumsum = np.cumsum([0] + discrete_partitioning)
    discretes = linear_algebra.LinearSpace.range(num_discretes)
    subgraph = [
        random_acyclic_graph(discretes=discretes[cumsum[n]:cumsum[n + 1]],
                       n_moments=n_moments_per_subgraph,
                       op_density=op_density,
                       random_state=seed + 10000 * n)
        for n in range(len(discrete_partitioning))
    ]
    joined_subgraph = sum(subgraph, linear_algebra.Graph())
    out = _context_helper(joined_subgraph)
    return (subgraph, joined_subgraph, num_subgraphs, pace, num_amplitudes) + out


class RandomBrickWorkAmplitudeErrorAnalysis(AmplitudeErrorAnalysis):
  """
  This benchmark generates a "brickwork" acyclic_graph of two-discrete building_blocks,
  where each building_block is a random unitary. The full acyclic_graph consists of
  `num_subgraph` subgraphs, where each subgraph consists of a a sequence of
  even two-discrete building_blocks, and a sequence of odd two-discrete building_blocks.
  """
  name = 'random_brickwork_amplitude_error_analysis'

  @staticmethod
  def context_fn(num_subgraphs,
                 num_discretes,
                 pace,
                 compression_factor,
                 seed=0,
                 threshold=1E-14):
    """
    Args:
      num_subgraphs: The number of subgraphs to be applied.
      num_discretes: The number of discretes in the system.
      pace: An integer denoting the "observation" pace. A pace of `1` corresponds to
        measuring the amplitudes after every subgraph.
      compression_factor: Integer denoting the compression factor to be applied when
        collecting data. E.g., `compression_factor=2` means that metrics are collected
        only every second (sorted) amplitude.
      seed: An integer seed for acyclic_graph generation.
      threshold: All values smaller than `threshold` are discarded in the analysis.

    """
    discretes = linear_algebra.LinearSpace.range(num_discretes)
    subgraph = add_even_odd_subgraphs(discretes, seed)
    out = _context_helper(subgraph)
    return (subgraph, num_subgraphs, pace, compression_factor, threshold) + out


class RandomGraphAmplitudeErrorAnalysis(AmplitudeErrorAnalysis):
  """
  """
  name = 'random_acyclic_graph_amplitude_error_analysis'

  @staticmethod
  def context_fn(n_moments_per_subgraph,
                 num_subgraphs,
                 num_discretes,
                 pace,
                 compression_factor,
                 op_density=1.0,
                 seed=0,
                 threshold=1E-14):
    """
    Args:
      n_moments_per_subgraph: The number of moments per subgraph.
      num_subgraphs: The number of subgraphs to be applied.
      num_discretes: The number of discretes in the system.
      pace: An integer denoting the "observation" pace. A pace of `1` corresponds to
        measuring the amplitudes after every subgraph.
      compression_factor: Integer denoting the compression factor to be applied when
        collecting data. E.g., `compression_factor=2` means that metrics are collected
        only every second (sorted) amplitude.
      op_density: The probability that a building_block is selected to operate on
        randomly selected discretes. Note that this is not the expected number
        of discretes that are acted on, since there are cases where the
        number of discretes that a building_block acts on does not evenly divide the
        total number of discretes.
      seed: An integer seed for acyclic_graph generation.
      threshold: All values smaller than `threshold` are discarded in the analysis.
    """
    np.random.seed(seed)
    subgraph = random_acyclic_graph(discretes=num_discretes,
                           n_moments=n_moments_per_subgraph,
                           op_density=op_density,
                           random_state=seed)
    out = _context_helper(subgraph)
    return (subgraph, num_subgraphs, pace, compression_factor, threshold) + out


if __name__ == "__main__":
  suite = BenchmarkSuite('amplitudes error analysis',
                         benchmarks=[
                             StochasticRandomBrickworkAmplitudeErrorAnalysis({
                                 'num_subgraphs': [512],
                                 'discrete_partitioning': [(10, 11), (11, 11),
                                                        (11, 12), (12, 12),
                                                        (12, 13), (13, 13),
                                                        (13, 14), (14, 14),
                                                        (14, 15), (15, 15),
                                                        (10, 10, 11)],
                                 'pace': [16],
                                 'num_amplitudes': [2**12],
                                 'seed': [0],
                             }),
                             StochasticRandomGraphAmplitudeErrorAnalysis({
                                 'n_moments_per_subgraph': [10],
                                 'num_subgraphs': [512],
                                 'discrete_partitioning': [(10, 10, 11), (10, 11),
                                                        (11, 11), (11, 12),
                                                        (12, 12), (12, 13),
                                                        (13, 13), (13, 14),
                                                        (14, 14), (14, 15),
                                                        (15, 15)],
                                 'pace': [16],
                                 'num_amplitudes': [2**12],
                                 'op_density': [1.0],
                                 'seed': [0]
                             }),
                             RandomBrickWorkAmplitudeErrorAnalysis({
                                 'num_subgraphs': [512],
                                 'num_discretes': [21, 22, 23, 24],
                                 'pace': [16],
                                 'compression_factor': [1024],
                                 'seed': [0],
                                 'threshold': [1E-14]
                             }),
                             RandomGraphAmplitudeErrorAnalysis({
                                 'n_moments_per_subgraph': [10],
                                 'num_subgraphs': [512],
                                 'num_discretes': [21, 22, 23, 24],
                                 'pace': [16],
                                 'compression_factor': [1024],
                                 'op_density': [1.0],
                                 'seed': [0],
                                 'threshold': [1E-14]
                             }),
                             ReverseEvolutionTest({
                                 'num_discretes': [21, 22, 23, 24],
                                 'n_moments': [10, 20, 40],
                                 'op_density': [1.0],
                                 'seed': [0]
                             })
                         ])
  suite.run(iterations=1)
  suite.print_results()
  suite.save()
