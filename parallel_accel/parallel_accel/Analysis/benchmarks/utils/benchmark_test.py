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
import unittest

from benchmarks.utils import benchmark
class TestBenchmark(benchmark.Benchmark):
  name = 'test_benchmark'

  @staticmethod
  def context_fn(a, b):
    return a, b

  @staticmethod
  def benchmark_fn(context):
    a, b = context
    return {'val': a * b}

class FailingBenchmark(TestBenchmark):
  @staticmethod
  def benchmark_fn(context):
    raise Exception
  
class BenchmarkTest(unittest.TestCase):

  def test_context_arg_check(self):
    with self.assertRaises(ValueError):
      b = TestBenchmark({})
    with self.assertRaises(ValueError):
      b = TestBenchmark({'a': [1], 'c': [1]})
    b = TestBenchmark({'a': [1], 'b': [1]})

  def test_run_output(self):
    b = TestBenchmark({'a': [1], 'b': [2]})
    output = b.run(5)
    # check name
    self.assertEqual(output['benchmark_name'], 'TestBenchmark.test_benchmark')

    # check vals
    expected = [{'context_params': {'a': 1, 'b': 2},
                 'results': {'val': [2] * 5}}]
    self.assertEqual(output['run_configs'], expected)

    # check num run_contexts
    b = TestBenchmark({'a': [1, 2, 3], 'b': [1, 2, 3]})
    output = b.run(1)
    self.assertEqual(len(output['run_configs']), 9)

  def test_failed_benchmark(self):

    b = FailingBenchmark({'a': [1], 'b': [1]})
    output = b.run(1)
    self.assertIn('fail', output['run_configs'][0])

  def test_save_load_suite(self):
    name = 'test-suite'
    suite = benchmark.BenchmarkSuite(name,
                                     benchmarks=[TestBenchmark({'a': [1,2], 'b': [3,4]})])
    suite.save(name)
    loaded_suite = suite.load(name)
    assert loaded_suite.name==suite.name
    assert not loaded_suite.completed
    loaded_suite.run(1)
    suite.run(1)
    val1 = suite._benchmarks[0].results['run_configs'][0]['results']['val']
    val2 = loaded_suite._benchmarks[0].results['run_configs'][0]['results']['val']
    assert val1 == val2

    suite.save(name)
    loaded_suite = suite.load(name)
    assert loaded_suite.completed
    val2 = loaded_suite._benchmarks[0].results['run_configs'][0]['results']['val']
    assert val1 == val2
