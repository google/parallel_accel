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
from benchmarks.utils.benchmark import BenchmarkSuite
from benchmarks.parallel_accel.method_benchmarks import (SamplesBenchmark, ExpBenchmark,
                                               ExpGradBenchmark)
from benchmarks.parallel_accel.state_benchmarks import (
    CRNGParallelAccelBenchmark, RandomGraphBenchmark, ApproximateOptimizationBenchmark, HEABenchmark,
    QuickSimGraphParallelAccelBenchmark)

suite = BenchmarkSuite( 'Canonical Benchmark Suite', benchmarks=(
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
    'seed': [0]}),
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
      'num_prob_basis_axis_strings':[2, 4], #use num_discretes prob-basis-axis-strings
      'num_prob_basis_axis_factors':[1], #each prob-basis-axis-string consists of a single prob-basis-axis-factor
      'acyclic_graph_type': ['hea','approxopt'],
      'seed': [0]
  }),
  ExpGradBenchmark({
      'num_subgraphs': [2, 3],
      'num_discretes': [26, 28],
      'num_prob_basis_axis_strings':[2], #use num_discretes prob-basis-axis-strings
      'num_prob_basis_axis_factors':[1], #each prob-basis-axis-string consists of a single prob-basis-axis-factor
      'acyclic_graph_type': ['approxopt'],
      'seed': [0]
  })
))

if __name__ == '__main__':
  iterations = 5
  suite.run(iterations)
  suite.print_results()
  suite.save()
  suite.save_plots('canonical_benchmark.pdf')
