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
import linear_algebra
import jax
import pytest


jax.config.update('jax_enable_x64', True)
import numpy as np
import graph_helper_tool as tn
from asic_la import parser
from asic_la.preprocessor import network

def test_network_append():
  nodes = [0,1,2,3]
  axes = [(0,1), (1,2), (3,0),(3,2)]
  net = network.Network.from_axes(axes, 4)
  adj = {0:[1,2, -1],
         1:[0,3, -1],
         2:[0,3, -1],
         3:[1,2]}
  links = {0:{0:[-1, 2],1:[-1,1]},
           1:{1:[0,None],2:[-1, 3]},
           2:{0:[0, None], 3:[-1,3]},
           3:{2:[1,None],3:[2,None]}}
  for n in nodes:
    assert(set(net.adjacencies[n]) == set(adj[n]))
    for ax in net.links[n]:
      assert net.links[ax] == links[ax]

@pytest.mark.parametrize('N',[4,6,8])
@pytest.mark.parametrize('depth',[4,6,8])
@pytest.mark.parametrize('seed',list(range(10)))
def test_contraction(N, depth, seed):
  discretes = linear_algebra.LinearSpace.range(N)
  acyclic_graph = linear_algebra.testing.random_acyclic_graph(discretes, depth, 1.0, random_state=seed)
  building_blocks, _, axes  = parser.parse(acyclic_graph, discretes, linear_algebra.ParamResolver({}), dtype=np.complex128)
  expected = linear_algebra.unitary(acyclic_graph)

  net = network.Network.from_axes(axes, N)
  ncon_labels = network.compute_ncon_labels(list(range(len(building_blocks))), net)
  actual = tn.ncon(building_blocks, ncon_labels).reshape(2**N, 2**N)
  np.testing.assert_almost_equal(expected, actual)


@pytest.mark.parametrize('N',[4,6,8])
@pytest.mark.parametrize('depth',[40,60])
@pytest.mark.parametrize('seed',list(range(10)))
def test_fuse(N, depth, seed):
  discretes = linear_algebra.LinearSpace.range(N)
  acyclic_graph = linear_algebra.testing.random_acyclic_graph(discretes, depth, 1.0, random_state=seed)
  building_blocks, _, axes  = parser.parse(acyclic_graph, discretes, linear_algebra.ParamResolver({}), dtype=np.complex128)
  expected = linear_algebra.unitary(acyclic_graph)

  net = network.Network.from_axes(axes, N)
  supernodes, _, ncon_labels = net.fuse(max_discrete_support = N)
  assert len(supernodes) == 1
  actual = tn.ncon([building_blocks[n] for n in supernodes[0]], ncon_labels[0]).reshape(2**N, 2**N)
  np.testing.assert_almost_equal(expected, actual)

@pytest.mark.parametrize('N',[4,6,8])
@pytest.mark.parametrize('depth',[40,60])
@pytest.mark.parametrize('seed',list(range(10)))
def test_multiple_fuse(N, depth, seed):
  discretes = linear_algebra.LinearSpace.range(N)
  acyclic_graph = linear_algebra.testing.random_acyclic_graph(discretes, depth, 1.0, random_state=seed)
  building_blocks, _, axes  = parser.parse(acyclic_graph, discretes, linear_algebra.ParamResolver({}), dtype=np.complex128)
  expected = linear_algebra.unitary(acyclic_graph)

  net = network.Network.from_axes(axes, N)
  supernodes, superaxes, ncon_labels = net.fuse(max_discrete_support = N//2)
  supnet = network.Network.from_axes(superaxes, N)
  supsupernodes, _, supncon_labels = supnet.fuse(max_discrete_support = N)

  large_blocks = [tn.ncon([building_blocks[g] for g in sn], labels) for sn, labels in zip(supernodes, ncon_labels)]
  assert len(supsupernodes) == 1
  actual = tn.ncon([large_blocks[n] for n in supsupernodes[0]], supncon_labels[0]).reshape(2**N, 2**N)
  np.testing.assert_almost_equal(expected, actual)


@pytest.mark.parametrize('N',[4,6])
@pytest.mark.parametrize('depth',[10])
@pytest.mark.parametrize('seed',list(range(10)))
def test_multiple_fuse_with_maxlength(N, depth, seed):
  discretes = linear_algebra.LinearSpace.range(N)
  acyclic_graph = linear_algebra.testing.random_acyclic_graph(discretes, depth, 1.0, random_state=seed)
  building_blocks, _, axes  = parser.parse(acyclic_graph, discretes, linear_algebra.ParamResolver({}), dtype=np.complex128)
  expected = linear_algebra.unitary(acyclic_graph)

  net = network.Network.from_axes(axes, N)
  supernodes, superaxes, ncon_labels = net.fuse(max_discrete_support = N//2, maxlength=4)
  assert all([len(sn) <= 4 for sn in supernodes])
  supnet = network.Network.from_axes(superaxes, N)
  supsupernodes, _, supncon_labels = supnet.fuse(max_discrete_support = N)

  large_blocks = [tn.ncon([building_blocks[g] for g in sn], labels) for sn, labels in zip(supernodes, ncon_labels)]
  assert len(supsupernodes) == 1
  actual = tn.ncon([large_blocks[n] for n in supsupernodes[0]], supncon_labels[0]).reshape(2**N, 2**N)
  np.testing.assert_almost_equal(expected, actual)


