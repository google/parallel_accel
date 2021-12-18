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
"""
Test the speed of GRALTool on standard benchmark acyclic_graphs.

This is deprecated code and is included for reference. New benchmarks should use the
Benchmark and BenchmarkSuite models.
"""

import json
import os
import time

import benchmarks.acyclic_graphs.benchmark_acyclic_graphs as acyclic_graphs
from benchmarks.acyclic_graphs import pbaxisum
import benchmarks.gralt.settings as settings

import linear_algebra
import tensorflow as tf
import grapal_tool as gralt

sample_subgraph = gralt.subgraphs.Sample()
expectation_subgraph = gralt.subgraphs.Expectation()
state_subgraph = gralt.subgraphs.State()


def exp_and_grad_call(
    acyclic_graph_t, symbol_names_t, symbol_values_t, ops_t, num_samples_t):
  with tf.GradientTape() as g:
    g.watch(symbol_values_t)
    exp = expectation_subgraph(
        acyclic_graph_t, symbol_names=symbol_names_t, symbol_values=symbol_values_t,
        operators=ops_t)
  grad = g.gradient(exp, symbol_values_t)
  return exp, grad


call_dict = {
    "samples": lambda acyclic_graph_t, symbol_names_t, symbol_values_t, ops_t,
    num_samples_t: sample_subgraph(
        acyclic_graph_t, symbol_names=symbol_names_t, symbol_values=symbol_values_t,
        repetitions=num_samples_t),
    "exp": lambda acyclic_graph_t, symbol_names_t, symbol_values_t, ops_t,
    num_samples_t: expectation_subgraph(
        acyclic_graph_t, symbol_names=symbol_names_t, symbol_values=symbol_values_t,
        operators=ops_t),
    "exp_and_grad": exp_and_grad_call,
    "state": lambda acyclic_graph_t, symbol_names_t, symbol_values_t, ops_t,
    num_samples_t: state_subgraph(
        acyclic_graph_t, symbol_names=symbol_names_t, symbol_values=symbol_values_t),
}


get_num_samples_dict = {
    "samples": lambda settings_dict:
        tf.constant([settings_dict["num_samples"]]),
    "exp": lambda settings_dict: tf.constant([0]),
    "exp_and_grad": lambda settings_dict: tf.constant([0]),
    "state": lambda settings_dict: tf.constant([0]),
}


get_ops_dict = {
    "samples": lambda discretes: tf.constant(""),
    "exp": lambda discretes:
        gralt.convert_to_tensor([[pbaxisum.get_random_prob_basis_axis_sum(discretes)]]),
    "exp_and_grad": lambda discretes:
        gralt.convert_to_tensor([[pbaxisum.get_random_prob_basis_axis_sum(discretes)]]),
    "state": lambda discretes: tf.constant(""),
}


def run_gralt_benchmarks(
    min_subgraphs, max_subgraphs, skip_subgraphs, min_discretes, max_discretes, iterations,
    num_samples, rounding_digits, acyclic_graph_type, sim_type, rel_save_dir,
    save_dir_prefix=os.getcwd()):

  if acyclic_graph_type == "approxopt":
    acyclic_graph_builder = acyclic_graphs.approxopt
  elif acyclic_graph_type == "hea":
    acyclic_graph_builder = acyclic_graphs.hea
  else:
    raise ValueError(acyclic_graph_type + " is not a valid type of test acyclic_graph.")

  if sim_type in {"samples", "exp", "exp_and_grad", "state"}:
    call_subgraph = call_dict[sim_type]
    get_num_samples = get_num_samples_dict[sim_type]
    get_ops = get_ops_dict[sim_type]
  else:
    raise ValueError(sim_type + " is not a valid simulation types.")

  # Save settings.
  full_save_dir = os.path.join(save_dir_prefix, rel_save_dir)
  settings.set_settings(
      min_subgraphs=min_subgraphs,
      max_subgraphs=max_subgraphs,
      skip_subgraphs=skip_subgraphs,
      min_discretes=min_discretes,
      max_discretes=max_discretes,
      iterations=iterations,
      num_samples=num_samples,
      rounding_digits=rounding_digits,
      acyclic_graph_type=acyclic_graph_type,
      sim_type=sim_type,
      full_save_dir=full_save_dir
  )
  settings_dict = settings.load_settings(full_save_dir)

  # Run benchmarks.
  num_samples_t = get_num_samples(settings_dict)
  for q in range(settings_dict["min_discretes"], settings_dict["max_discretes"] + 1):
    print(f"Current discrete size: {q}")
    benchmarks_dict = dict()
    discretes = linear_algebra.GridSpace.rect(1, q)
    ops_t = get_ops(discretes)
    for l in range(
        settings_dict["min_subgraphs"], settings_dict["max_subgraphs"] + 1,
        settings_dict["skip_subgraphs"]):
      print(f"Current number of subgraphs: {l}")
      benchmarks_dict[l] = {}
      acyclic_graph, symbols = acyclic_graph_builder(discretes, l, acyclic_graph_type)
      is_acyclic_graph_compiled = False
      symbol_names_t = tf.constant([str(s) for s in symbols])
      for r in range(settings_dict["iterations"]):
        symbol_values_t = tf.random.uniform(
            [1, len(symbols)], minval=-2.0, maxval=2.0)
        start = time.time()
        if not is_acyclic_graph_compiled:
          compiled_acyclic_graph = gralt.convert_to_tensor([acyclic_graph])
          is_acyclic_graph_compiled = True
        result = call_subgraph(
            compiled_acyclic_graph, symbol_names_t, symbol_values_t,
            ops_t, num_samples_t)
        stop = time.time()
        this_runtime = round(stop - start, rounding_digits)
        if r == 0:
          # First run is special because it considers the compilation time
          benchmarks_dict[l]["initial"] = this_runtime
          benchmarks_dict[l]["remaining"] = []
          print("initial runtime of {} seconds".format(this_runtime))
        else:
          print("subsequent runtime of {} seconds".format(this_runtime))
          benchmarks_dict[l]["remaining"].append(this_runtime)
        benchmarks_dict[l]["depth"] = len(acyclic_graph)

    # Checkpoint the benchmarks after each discrete number.
    benchmarks_filename = "benchmarks_dict_{}.json".format(q)
    benchmarks_data_file = os.path.join(full_save_dir, benchmarks_filename)
    with open(benchmarks_data_file, 'w') as datafile:
      json.dump(benchmarks_dict, datafile)
