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
This module provides function for generating random CRNG in linear_algebra and quick_sim
format. When run as main this module will save a quick_sim acyclic_graph to file.
"""

import argparse
import linear_algebra
from benchmarks.acyclic_graphs.benchmark_acyclic_graphs import crng_acyclic_graph


# Read CL inputs.
def parse_args():
  parser = argparse.ArgumentParser(description='Program to generate RCS.')
  parser.add_argument('n', type=int, help='Number of discretes.')
  parser.add_argument('d', type=int, help='Number of XEB cycles (depth).')
  parser.add_argument('s', type=int, help='Seed.')
  parser.add_argument('output_file',
                      type=str,
                      help='Output file for the acyclic_graph in quick_sim format.')
  args = parser.parse_args()
  return args


QSIM_GATE_NAME = {
    linear_algebra.flip_x_axis**0.5: 'x_1_2 {0}',
    linear_algebra.flip_y_axis**0.5: 'y_1_2 {0}',
    linear_algebra.x_axis_two_angles(phase_exponent=0.25, exponent=0.5): 'hz_1_2 {0}',
    linear_algebra.google.SYC: 'fs {0} {1} 1.5 0.5',
    linear_algebra.cond_flip_z: 'cz {0} {1}'
}


def acyclic_graph_to_quick_sim(f, acyclic_graph):
  """Write acyclic_graph to quick_sim format.
  Args:
    f: file where to store the acyclic_graph.
    acyclic_graph: linear_algebra.Graph with building_blocks in the set {X^0.5, Y^0.5, W^0.5
        PhasedXPowGate, PlanarTreeGraph, and CZ}.

  Raises:
    ValueError: if one of the building_blocks is not in the set above.
  """
  discretes = list(acyclic_graph.all_discretes())
  discretes.sort()

  def building_block_to_quick_sim(building_block):
    if building_block.building_block not in QSIM_GATE_NAME:
      raise ValueError(f"Gate {building_block.building_block} is not a valid building_block.")
    return QSIM_GATE_NAME[building_block.building_block].format(
        *[discretes.index(q) for q in building_block.discretes])

  f.write('%d\n' % len(discretes))
  for j, moment in enumerate(acyclic_graph):
    for building_block in moment:
      f.write('%d ' % j + building_block_to_quick_sim(building_block) + '\n')


def main():
  # Get CL inputs.
  args = parse_args()
  n = args.n
  depth = args.d
  seed = args.s
  output_file = args.output_file

  acyclic_graph = crng_acyclic_graph(n, depth, seed)
  with open(output_file, 'w') as f:
    acyclic_graph_to_quick_sim(f, acyclic_graph)


if __name__ == '__main__':
  main()
