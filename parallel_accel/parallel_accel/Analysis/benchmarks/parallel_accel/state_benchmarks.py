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
import time
import itertools

import linear_algebra
import numpy as np
from matplotlib.ticker import MaxNLocator

import asic_la.asic_simulator as asic_simulator
from benchmarks.utils.benchmark import BenchmarkSuite
from benchmarks.parallel_accel.method_benchmarks import WalltimeBenchmark
from benchmarks.acyclic_graphs import benchmark_acyclic_graphs
from benchmarks.acyclic_graphs.quick_sim_acyclic_graphs import (
    acyclic_graph_q28c0d14, acyclic_graph_q30c0d14, acyclic_graph_q32c0d14, acyclic_graph_q34c0d14,
    acyclic_graph_q36c0d14, acyclic_graph_q40c0d14)
from benchmarks.utils import mpl_style


def label_axes(ax):
  ax.set_xlabel('num_discretes')
  ax.set_ylabel('walltime (s)')
  ax.set_yscale('log')
  # Set major locators to integer values
  ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))


def state_benchmark_fn(context):
  acyclic_graph, resolver, sim = context
  t1 = time.time()
  result = sim.compute_final_state_vector(acyclic_graph, resolver)
  result.real.block_until_ready()
  walltime = time.time() - t1
  del result
  gc.collect()
  return {'walltime': walltime}


class StateBenchmark(WalltimeBenchmark):
  benchmark_fn = staticmethod(state_benchmark_fn)
  def get_plot(self,
               marker='o',
               consolidation_params=['depth'],
               average_over=None,
               xscale='linear',
               yscale='log'):
    return super().get_plot(marker=marker, consolidation_params=consolidation_params,
                            average_over=average_over, xscale=xscale, yscale=yscale)


class CRNGParallelAccelBenchmark(StateBenchmark):
  name = 'crng_acyclic_graph_state'

  @staticmethod
  def context_fn(num_discretes, depth, seed):
    sim = asic_simulator.ASICSimulator()
    acyclic_graph = benchmark_acyclic_graphs.crng_acyclic_graph(num_discretes, depth, seed)
    resolver = linear_algebra.ParamResolver({})
    return acyclic_graph, resolver, sim


class RandomGraphBenchmark(StateBenchmark):
  name = 'random_acyclic_graph_state'

  @staticmethod
  def context_fn(num_discretes, depth, op_density, seed):
    sim = asic_simulator.ASICSimulator()
    acyclic_graph = linear_algebra.testing.random_acyclic_graph(
        num_discretes, depth, op_density, random_state=seed)
    resolver = linear_algebra.ParamResolver({})
    return acyclic_graph, resolver, sim


class QuickSimGraphParallelAccelBenchmark(StateBenchmark):
  name = 'quick_sim_acyclic_graph_state'

  @staticmethod
  def context_fn(num_discretes):
    sim = asic_simulator.ASICSimulator()
    acyclic_graphs = {
        28: acyclic_graph_q28c0d14,
        30: acyclic_graph_q30c0d14,
        32: acyclic_graph_q32c0d14,
        34: acyclic_graph_q34c0d14,
        36: acyclic_graph_q36c0d14,
        40: acyclic_graph_q40c0d14
    }
    resolver = linear_algebra.ParamResolver({})
    return acyclic_graphs[num_discretes], resolver, sim

  def get_plot(self,
               marker='o',
               average_over=None,
               xscale='linear',
               yscale='log'):
    return super().get_plot(marker=marker, consolidation_params=[],
                            average_over=average_over, xscale=xscale, yscale=yscale)


class HEABenchmark(StateBenchmark):
  name = 'hea_acyclic_graph_state'

  @staticmethod
  def context_fn(depth, num_discretes, seed):
    np.random.seed(seed)
    discretes = linear_algebra.LinearSpace.range(num_discretes)
    acyclic_graph, symbols = benchmark_acyclic_graphs.hea(discretes, depth,
                                              'benchmark_subgraph')
    resolver = linear_algebra.ParamResolver(
        {s.name: float(np.random.randn(1)) for s in symbols})
    sim = asic_simulator.ASICSimulator()
    return acyclic_graph, resolver, sim


class ApproximateOptimizationBenchmark(StateBenchmark):
  name = 'approxopt_acyclic_graph_state'

  @staticmethod
  def context_fn(depth, num_discretes, seed):
    np.random.seed(seed)
    discretes = linear_algebra.LinearSpace.range(num_discretes)
    acyclic_graph, symbols = benchmark_acyclic_graphs.approxopt(discretes, depth,
                                               'benchmark_subgraph')
    resolver = linear_algebra.ParamResolver(
        {s.name: float(np.random.randn(1)) for s in symbols})
    sim = asic_simulator.ASICSimulator()
    return acyclic_graph, resolver, sim

if __name__ == "__main__":
  iterations = 5

  suite = BenchmarkSuite( 'Basic benchmarks', benchmarks=(
    CRNGParallelAccelBenchmark({
        'num_discretes': [26, 28, 30, 32],
        'depth': [20, 30],
        'seed': [0]
      }),
    RandomGraphBenchmark({
        'num_discretes': [26, 28, 30, 32],
        'depth': [20, 30],
        'op_density': [1.0],
        'seed': [0]
    }),
    ApproximateOptimizationBenchmark({
        'num_discretes': [26, 28, 30, 32],
        'depth': [20, 30],
        'seed': [0]
    }),
    QuickSimGraphParallelAccelBenchmark({'num_discretes': [28, 30, 32]}),
    HEABenchmark({
        'num_discretes': [26, 28, 30, 32],
        'depth': [15, 20],
        'seed': [0]})
    ))


  suite.run(iterations)
  #loaded_suite = suite.load('some_previous_results')

  suite.print_results()
  suite.save()
  suite.save_plots('state_benchmark.pdf')
