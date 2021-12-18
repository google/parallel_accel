#! /bin/python3
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
# Sanity check when to run sim code.
from asic_la.asic_simulator import ASICSimulator
import linear_algebra

discretes = linear_algebra.LinearSpace.range(30)
acyclic_graph = linear_algebra.Graph([linear_algebra.flip_x_axis(q) for q in discretes] + [linear_algebra.measure(q) for q in discretes])

sim = ASICSimulator()
print('-- first run --')
res = sim.run(acyclic_graph)
print(res)

print('-- second run --')
res = sim.run(acyclic_graph)
print(res)