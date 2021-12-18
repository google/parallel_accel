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
"""Benchmarks for endpoint functions including sampling, expectation computation
and expectation w/ gradient computation on a variety of standard acyclic_graphs."""

import time
import itertools

import linear_algebra
import numpy as np
from matplotlib.ticker import MaxNLocator

import asic_la.asic_simulator as asic_simulator
from benchmarks.acyclic_graphs import benchmark_acyclic_graphs
from benchmarks.acyclic_graphs import pbaxisum
from benchmarks.utils import benchmark
from benchmarks.utils.benchmark import BenchmarkSuite
from benchmarks.utils.jax_benchmark import JaxBenchmark
from benchmarks.utils import mpl_style


def label_axes(ax):
  ax.set_xlabel('num_discretes')
  ax.set_ylabel('walltime (s)')
  ax.set_yscale('log')
  ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))  ## Set major locators to integer values


def get_acyclic_graph(num_discretes, num_subgraphs, acyclic_graph_type, seed):
  np.random.seed(seed)
  discretes = linear_algebra.LinearSpace.range(num_discretes)
  if acyclic_graph_type == 'hea':
    acyclic_graph, symbols = benchmark_acyclic_graphs.hea(discretes, num_subgraphs,
                                              'benchmark_subgraph')
  elif acyclic_graph_type == 'approxopt':
    acyclic_graph, symbols = benchmark_acyclic_graphs.approxopt(discretes, num_subgraphs,
                                               'benchmark_subgraph')
  else:
    raise ValueError(f'Invalid acyclic_graph type: {acyclic_graph_type}')

  resolver = linear_algebra.ParamResolver(
      {s.name: float(np.random.randn(1)) for s in symbols})
  return discretes, acyclic_graph, symbols, resolver


def get_walltime(fn):
  def wrapper(context):
    t1 = time.time()
    fn(context)
    walltime = time.time() - t1
    return {'walltime': walltime}
  return wrapper

class WalltimeBenchmark(JaxBenchmark):

  def get_plot(self,
               marker='o',
               consolidation_params=['num_subgraphs'],
               average_over=None,
               xscale='linear',
               yscale='log'):
    _,_, figs_and_axes_1 = self.plot_helper(
        'num_discretes',
        'walltime',
        reduction_function=lambda x: x[0],
        consolidation_params=consolidation_params,
        linestyles=['--'],
        markers=[marker],
        xscale=xscale,
        yscale=yscale,
        label='first run',
        average_over=average_over)
    _,_, figs_and_axes_2 = self.plot_helper(
        'num_discretes',
        'walltime',
        reduction_function=lambda x: np.median(x[1:]),
        consolidation_params=consolidation_params,
        linestyles=['solid'],
        markers=[marker],
        xscale=xscale,
        yscale=yscale,
        label='subsequent runs (median)',
        average_over=average_over)
    figs_and_axes={}
    for param in figs_and_axes_1:
      l = figs_and_axes.get(param,[])
      for n, f_and_a in enumerate(figs_and_axes_1[param]):
        axes = [f_and_a[1], figs_and_axes_2[param][n][1]]
        print(axes)
        l.append(benchmark.merge_plot_axes(axes,
                                           xlabel='num_discretes', ylabel='walltime',
                                           title = axes[0].get_title(),
                                           xscale=xscale, yscale=yscale))
      figs_and_axes[param] = l

    fig, axes = benchmark.to_subplots(figs_and_axes, 'num_discretes', 'walltime')
    if isinstance(axes, np.ndarray):
      for a in axes:
        a.set_xscale(xscale)
        a.set_yscale(yscale)
    else:
      axes.set_xscale(xscale)
      axes.set_yscale(yscale)

    return fig

class SamplesBenchmark(WalltimeBenchmark):
  name = 'samples_benchmark'

  @staticmethod
  def context_fn(num_subgraphs, num_discretes, acyclic_graph_type, num_samples, seed):
    discretes, acyclic_graph, symbols, resolver = get_acyclic_graph(num_discretes, num_subgraphs,
                                                     acyclic_graph_type, seed)
    sim = asic_simulator.ASICSimulator()
    return acyclic_graph, symbols, resolver, sim, num_samples

  @staticmethod
  @get_walltime
  def benchmark_fn(context):
    acyclic_graph, symbols, resolver, sim, num_samples = context
    resolved_acyclic_graph = linear_algebra.resolve_parameters(acyclic_graph, resolver)
    sim._sample_observation_ops(resolved_acyclic_graph, num_samples)

class ExpBenchmark(WalltimeBenchmark):
  name = 'exp_benchmark'

  @staticmethod
  def context_fn(num_subgraphs,
                 num_discretes,
                 acyclic_graph_type,
                 seed,
                 num_prob_basis_axis_strings=None,
                 num_prob_basis_axis_factors=1):
    discretes, acyclic_graph, symbols, resolver = get_acyclic_graph(num_discretes, num_subgraphs,
                                                     acyclic_graph_type, seed)
    sim = asic_simulator.ASICSimulator()
    ps = pbaxisum.get_random_prob_basis_axis_sum(
        discretes, num_prob_basis_axis_strings=num_prob_basis_axis_strings,
        num_prob_basis_axis_factors=num_prob_basis_axis_factors, seed=seed+1000)
    return acyclic_graph, symbols, resolver, ps, sim

  @staticmethod
  @get_walltime
  def benchmark_fn(context):
    acyclic_graph, symbols, resolver, ps, sim = context
    result = sim.compute_expectations(acyclic_graph, [ps], resolver)



class ExpGradBenchmark(ExpBenchmark):
  name = 'exp_grad_benchmark'

  @staticmethod
  @get_walltime
  def benchmark_fn(context):
    acyclic_graph, symbols, resolver, ps, sim = context
    result = sim.compute_gradients(acyclic_graph, ps, resolver)


if __name__ == '__main__':
  iterations = 5

  suite = BenchmarkSuite( 'Method Benchmarks', benchmarks=(
    SamplesBenchmark({
        'num_subgraphs': [10, 20],
        'num_discretes': [26, 28, 30, 32],
        'acyclic_graph_type': ['hea','approxopt'],
        'num_samples': [100],
        'seed': [0]
      }),
    ExpBenchmark({
        'num_subgraphs': [10, 20],
        'num_discretes': [26, 28, 30],
        'acyclic_graph_type': ['hea','approxopt'],
        'num_prob_basis_axis_strings': [None], #use num_discretes prob-basis-axis-strings
        'num_prob_basis_axis_factors': [1], #each prob-basis-axis-string consists of a single prob-basis-axis-factor
        'seed': [0]
    }),
    ExpGradBenchmark({
        'num_subgraphs': [2, 4],
        'num_discretes': [26, 28],
        'acyclic_graph_type': ['hea','approxopt'],
        'num_prob_basis_axis_strings': [4], #use 4 prob-basis-axis-strings
        'num_prob_basis_axis_factors': [1],#each prob-basis-axis-string consists of a single prob-basis-axis-factor
        'seed': [0]
    })
  ))

  suite.run(iterations)
  #loaded_suite = suite.load('some_previous_results')

  suite.print_results()
  suite.save_plots('method_benchmark.pdf')
  suite.save()
