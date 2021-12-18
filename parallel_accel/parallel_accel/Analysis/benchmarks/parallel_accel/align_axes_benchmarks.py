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
"""Benchmarks aligning axes"""
import argparse
import gc
import random
import time

import jax
import jax.numpy as jnp
from matplotlib.ticker import MaxNLocator
import numpy as np

from benchmarks.parallel_accel.method_benchmarks import ExpGradBenchmark
from benchmarks.parallel_accel.state_benchmarks import RandomGraphBenchmark
from benchmarks.utils import mpl_style
from benchmarks.utils.benchmark import BenchmarkSuite
from benchmarks.utils.jax_benchmark import JaxBenchmark
import asic_la
from asic_la.sharded_probability_function import ShardedDiscretedProbabilityFunction

AXIS_NAME = asic_la.sharded_probability_function.jax_wrappers.AXIS_NAME

@jax.partial(jax.pmap, static_broadcasted_argnums=(1,2), axis_name=AXIS_NAME)
def initialize_probability_function(x, num_discretes, perm):
  num_global_discretes = int(np.log2(jax.device_count()))
  shape = (2,) * (num_discretes - 10 - num_global_discretes) + (8, 128)
  state = ShardedDiscretedProbabilityFunction(jnp.ones(shape), perm)
  return state

@jax.partial(jax.pmap, axis_name=AXIS_NAME)
def align_axes(wf):
  return wf.align_axes()

def label_axes(ax, repetitions):
  ax.set_xlabel('seeds')
  ax.set_ylabel(f'median walltime (s) ({repetitions} reps)')
  ax.set_yscale('log')
  # Set major locators to integer values
  ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))

def benchmark_fn(context):
  t1 = time.time()
  result = align_axes(context)
  result.block_until_ready()
  t2 = time.time()
  del result
  gc.collect()
  return {'walltime': t2-t1}

class AlignAxesBenchmark(JaxBenchmark):
  """
  Benchmark for AbstractShardedProbabilityFunction.align_axes().
  This benchmark creates a `ShardedDiscretedProbabilityFunction` with a random `perm`
  attribute and aligns its axis using `ShardedDiscretedProbabilityFunction.align_axes()`.
  """
  name = 'align_axes_benchmark'

  @staticmethod
  def context_fn(seed, num_discretes):
    random.seed(seed)
    perm = tuple(random.sample(range(num_discretes),num_discretes))
    wf = initialize_probability_function(jnp.arange(jax.local_device_count()), num_discretes, perm)
    return wf

  @staticmethod
  def benchmark_fn(context):
    return benchmark_fn(context)

  def get_plot(self, label='', marker='o',
               linestyle='solid', consolidation_parameters=['seed'],
               average_over=None, reduction_function=lambda x: np.median(x[1:]),
               xscale='linear', yscale='linear'):
    f, _, _ = self.plot_helper('num_discretes',
                               'walltime',
                               reduction_function=reduction_function,
                               consolidation_params=consolidation_parameters,
                               linestyles=[linestyle],
                               markers=[marker],
                               xscale=xscale,
                               yscale=yscale,
                               label=label,
                               average_over=average_over)
    return f

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('filename', type=str,
                      help='filename for storing benchmarks')
  parser.add_argument('--loadfile', type=str,
                      help='filename of a stored benchmark')

  args = parser.parse_args()
  iterations = 5
  if not args.loadfile:
    suite = BenchmarkSuite( 'Align Axes Benchmark Suite ' + args.filename, benchmarks=(
      AlignAxesBenchmark({
        'num_discretes': [26, 28, 30, 32],
        'seed': list(range(10))
      }),
      RandomGraphBenchmark({
        'num_discretes': [26, 28, 30, 32],
        'depth': [30],
        'op_density': [1.0],
        'seed': list(range(10))
      }),
      ExpGradBenchmark({
          'num_subgraphs': [10],
          'num_discretes': [26, 28],
          'num_prob_basis_axis_strings': [2], #use num_discretes prob-basis-axis-strings
          'num_prob_basis_axis_factors': [1], #each prob-basis-axis-string consists of a single prob-basis-axis-factor
          'acyclic_graph_type': ['hea'],
          'seed': list(range(5))
      })
    ))
  else:
    suite = BenchmarkSuite.load(args.loadfile)
  suite.run(iterations)
  suite.print_results()
  suite.save(args.filename)
  suite.save_plots('align_axes_benchmark.pdf')
