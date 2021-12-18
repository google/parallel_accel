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
import copy
import functools as fct
import numpy as np
from asic_la import utils
import opt_einsum
from asic_la import config

from typing import Any, List, Tuple, Dict, Optional, Sequence

MAX_CACHE_SIZE = config.MAX_CACHE_SIZE


class Network:
  """
  A class for managing symplectic acyclic_graphs in tensor networks format.
  This class is instantiated from a list `axes` of tensor-axes and
  an integer of the total number of discretes. Each element in the list corresponds
  to the discretes on which the corresponding node acts. The order of `axes`
  matters. It is assumed that the corresponding nodes are applied in the
  order in which they appear in the list corresponding to `axes`.
  Nodes are represented by non-negative integers. The order of these integers
  determines the order in which the nodes are added to the symplectic acyclic_graph,
  ie it represents time. The class contains a special node, self.PROBABILITYFUNCTION,
  with value -1, which represents the state-vector, or probabilityfunction, of the
  acyclic_graph.

  Attributes:
    * PROBABILITYFUNCTION: A global alias for the integer -1, which represents the
      state-vector within the network.
    * adjacencies: A dict mapping a node (int) to a list of its neighbors.
    * links: A dict of a dict. The outer dict keys are nodes (int). The
      inner dict maps the discrete-name/axes on which the node acts to a tuple
      [earlier_neighbor, later_neighbor] of earlier/later neighbors of node.
      earlier_neighbor can be any integer in [-1, len(axes) - 2]. -1 denotes
      a dangling top link. later_neighbor can be any of [0, len(axes) - 1]
      or None. The latter denotes a dangling bottom link. Top and bottom are
      identified with initial/final states of the corresponding simulation.
    * nodes: A list of integers representing the nodes in the acyclic_graph. Smaller
      values denote earlier times in the acyclic_graph.
    * axes: A dict mapping a node (int) to a tuple of ints representing the
      discretes/axes on which the node acts
    * num_discretes: The number of discretes in the acyclic_graph.
  """
  PROBABILITYFUNCTION = -1

  @classmethod
  def from_axes(cls, axes: List[Tuple[int]], num_discretes: int) -> "Network":
    """
    Initialize a Network object from a list of operating axes.

    Args:
      axes: A list of tuples of ints representing the
        discretes/axes on which the nodes act
      num_discretes: The total number of discretes.

    Returns:
      Network: An initialized Network object
    """
    network = cls(num_discretes)
    for a in axes:
      network.append(a)
    return network

  def __init__(self,
               num_discretes: int) -> None:
    """
    Initialize a Network object

    Args:
      num_discretes: The total number of discretes.
      adjacencies: A dict mapping integers (node) to its neighbors (ints)
      links: A dict of a dict. The outer dict keys are nodes (int). The
        inner dict maps the discrete-name/axes on which the node acts to a tuple
        [earlier_neighbor, later_neighbor] of earlier/later neighbors of node.
        earlier_neighbor can be any integer in [-1, len(axes) - 2]. -1 denotes
        a dangling top link. later_neighbor can be any of [0, len(axes) - 1]
        or None. The latter denotes a dangling bottom link. Top and bottom are
        identified with initial/final states of the corresponding simulation.
      axes: A dict mapping a node (int) to a tuple of ints representing the
        discretes/axes on which the node acts

    """
    self.adjacencies = {self.PROBABILITYFUNCTION: []}
    self.links = {
        self.PROBABILITYFUNCTION: {a: [None, None] for a in range(num_discretes)}
    }
    self.axes = {self.PROBABILITYFUNCTION: tuple(range(num_discretes))}
    self.last_node_on_discrete = [self.PROBABILITYFUNCTION] * num_discretes
    self.num_discretes = num_discretes
    self.nodes = [self.PROBABILITYFUNCTION]


  def __len__(self) -> int:
    return len(self.axes)

  def get_neighbors(self, node: int) -> List[int]:
    """
    Get the neighbors of `node`.

    Args:
      node: An integer representing a symplectic node.

    Returns:
      List[int]: The neighbors of `node`.
    """
    return self.adjacencies[node]

  def get_links(self, node) -> Dict[int, List[int]]:
    """
    Get the links of `node`, ie the neighbors on each
    of its discrete-axes.

    Args:
      node: An integer representing a symplectic node.

    Returns:
      Dict[int, Tuple[Union[None, int]]:
        A mapping of a discrete axis to the earlier and later
        neighbors on that discrete axis.

    """
    return self.links[node]

  def append(self, axes: Tuple[int]) -> None:
    """
    Append a new node to the network. The node has support
    on `axes`. The order of the elements in `axes` matters.
    A full network is constructed by successively appending new
    nodes acting on `axes`.

    Args:
      axes: The axes on which a node acts.

    Returns:
      None
    """
    node = self.nodes[-1] + 1 #nodes are just consecutive integers
    self.nodes.append(node)
    self.axes[node] = axes
    self.adjacencies[node] = []
    for ax in axes:
      if node not in self.adjacencies[self.last_node_on_discrete[ax]]:
        self.adjacencies[self.last_node_on_discrete[ax]].append(node)
      if self.last_node_on_discrete[ax] not in self.adjacencies[node]:
        self.adjacencies[node].append(self.last_node_on_discrete[ax])
    self.adjacencies[node] = sorted(self.adjacencies[node])

    # now compute the links
    # we start with the neighbor closest in time to node
    # and recursively reduce the links
    self.links[node] = {ax:[None,None] for ax in axes}
    earlier_node_neighbors = reversed(
        sorted([self.last_node_on_discrete[a] for a in axes]))
    axes_set = set(axes)
    for neighbor in earlier_node_neighbors:
      common_axes = axes_set.intersection(self.axes[neighbor])
      for a in common_axes:
        self.links[node][a][0] = neighbor
        self.links[neighbor][a][1] = node
      axes_set -= common_axes

    for ax in axes:
      self.last_node_on_discrete[ax] = node

  def contract(self, next_node: int) -> None:
    """
    Contract `next_node` into the probabilityfunction (given by node PROBABILITYFUNCTION).

    Args:
      next_node: A node that is fully connected to the probabilityfunction
        (node self.PROBABILITYFUNCTION).

    Returns:
      None
    """
    # next_node is absorbed into node self.PROBABILITYFUNCTION
    # we need to update adjacencies and links accordingly
    if next_node == self.PROBABILITYFUNCTION:
      # nothing to do
      return

    for neighbor in self.adjacencies[next_node]:
      self.adjacencies[neighbor].remove(next_node)
      # update the links on connected axes.
      connected_axes = [
          ax for ax, val in self.links[neighbor].items() if val[0] == next_node
      ]
      for ax in connected_axes:
        # the links of neighboring nodes that formerly pointed to next_node
        # are redirected to point to node self.PROBABILITYFUNCTION (ie the probabilityfunction).
        self.links[neighbor][ax][0] = self.PROBABILITYFUNCTION
      if neighbor != self.PROBABILITYFUNCTION:
        if self.PROBABILITYFUNCTION not in self.adjacencies[neighbor]:
          self.adjacencies[neighbor].insert(0, self.PROBABILITYFUNCTION)

    for ax, l in self.links[next_node].items():
      self.links[self.PROBABILITYFUNCTION][ax] = list(l)

    old_neighbors = self.adjacencies[self.PROBABILITYFUNCTION]
    node_neighbors = self.adjacencies[next_node]
    node_neighbors.remove(self.PROBABILITYFUNCTION)
    self.adjacencies[self.PROBABILITYFUNCTION] = sorted(
        set(old_neighbors) | set(node_neighbors))

    del self.adjacencies[next_node]
    del self.links[next_node]
    del self.axes[next_node]
    self.nodes.remove(next_node)

  def contractable_nodes(self) -> List[int]:
    """
    Return a list of nodes that are fully connected with the probabilityfunction
    (ie. with node self.PROBABILITYFUNCTION). Fully connected means that all top
    legs of a node are connected with the probabilityfunction.

    Returns:
      List[int]: A list of nodes fully connected with the probabilityfunction.
    """
    nodes = []
    for neighbor in self.adjacencies[self.PROBABILITYFUNCTION]:
      if all([c[0] == self.PROBABILITYFUNCTION for c in self.links[neighbor].values()
             ]):
        nodes.append(neighbor)
    return nodes

  def fuse(
      self,
      max_discrete_support=7,
      maxlength=None
  ) -> Tuple[Tuple[Tuple[int]], Tuple[Tuple[int]], List[List[List[int]]]]:
    """
    Fuse nodes in the network into supernodes, such
    that the network is represented as a network of supernodes.

    Args:
      max_discrete_support: The maximum support of the fused supernodes.
      maxlength: The maximum number of nodes per supernode

    Returns:
      Tuple[Tuple[int]]: The supernodes, given as a list of integers.
      Tuple[Tuple[int]]: The superaxes of each supernode.
      List[List[List[int]]]: Contraction labels of each supernode, in ncon-format.
        The labels can be used to contract the nodes of each supernode into
        an actual array.
    """
    adjacencies = copy.deepcopy(self.adjacencies)
    links = copy.deepcopy(self.links)
    nodes = list(self.nodes)
    axes = dict(self.axes)
    supernode = []
    nodesupport = set()
    supernodes = []
    def length_exceeded(x):
      if maxlength is not None:
        return len(x) >= maxlength
      return False

    while (contractable_nodes:=self.contractable_nodes()):
      possible_axes = [
          nodesupport | set(self.axes[n]) for n in contractable_nodes
      ]
      # enlarge by smallest possible support
      i = np.argmin([len(p) for p in possible_axes])
      nodesupport = possible_axes[i]
      next_node = contractable_nodes[i]
      if (len(nodesupport) >
          max_discrete_support) or length_exceeded(supernode):
        supernodes.append(tuple(supernode))
        supernode = [next_node]
        nodesupport = set(self.axes[next_node])
      else:
        supernode.append(next_node)
      self.contract(next_node) # contract the node into PROBABILITYFUNCTION

    if len(supernode) > 0:
      supernodes.append(tuple(supernode))

    # restore the dicts
    self.adjacencies = adjacencies
    self.links = links
    self.nodes = nodes
    self.axes = axes

    ncon_labels = [compute_ncon_labels(sg, self) for sg in supernodes]
    superaxes = []
    for nl in ncon_labels:
      labels = sorted(set(flatten(nl)))
      nz = np.flatnonzero(np.asarray(labels) > 0)
      if len(nz) > 0:
        index = np.min(nz)
      else:
        index = len(labels)
      free_labels = list(reversed(labels[:index]))
      free_labels = free_labels[:len(free_labels) // 2]
      superaxes.append(tuple([-l - 1 for l in free_labels]))
    return tuple(supernodes), tuple(superaxes), ncon_labels

def compute_ncon_labels(supernode: Sequence[int],
                        network: Network) -> List[List[int]]:
  """
  Compute the ncon labels of a `supernode` which is part
  of `network`.

  Args:
    supernode: A sequence of nodes (int).
    network: The network object to which `supernode` belongs.

  Returns:
    List[List[int]]: The ncon labels, to be used for contracting
      `supernode`. The contraction order is not optimized. The
      convention for the open (negative) labels is that any
      uncontracted leg at the bottom gets assigned the negative
      discrete number minus 1 on which it acts. The uncontracted
      top legs get the same numbers decreased by the support of
      the node. That is, the resulting building_block can be directly reshaped
      into a matrix by combining the first and second half of the
      tensor legs.
  """
  num_discretes = network.num_discretes
  contraction_labels = {}
  for node in supernode:
    contraction_labels[node] = {}
    for discrete in network.axes[node]:
      contraction_labels[node][discrete] = [
          -(discrete + 1) - num_discretes, -(discrete + 1)
      ]

  next_cont = 1
  for node in supernode:
    dictionary = contraction_labels[node]
    for discrete, neighbors in network.links[node].items():
      l = dictionary[discrete]
      for neighbor in neighbors:
        if neighbor in supernode:
          # only update it once
          if l[node < neighbor] < 0:
            l[node < neighbor] = next_cont
            cont = next_cont
            next_cont += 1
          l2 = contraction_labels[neighbor][discrete]
          # only update it once
          if l2[neighbor < node] < 0:
            l2[neighbor < node] = cont

  ncon_labels = []
  for node in supernode:
    n_discretes = len(network.axes[node])
    labels = [None] * 2 * n_discretes
    cont_labels = contraction_labels[node]
    for n, discrete in enumerate(cont_labels.keys()):
      labels[n] = cont_labels[discrete][1]
      labels[n + n_discretes] = cont_labels[discrete][0]
    ncon_labels.append(labels)
  return ncon_labels


@fct.lru_cache(maxsize=MAX_CACHE_SIZE)
def opt_einsum_to_ncon_path(opteinsum_path: Tuple[Tuple[int]],
                            ncon_labels: Tuple[Tuple[int]]) -> Tuple[int]:
  """
  Convert an opt-einsum path of a given tensor contraction
  into a ncon-path that can be used to specify the contraction
  path within the ncon API.

  Args:
    opteinsum_path: The contraction path in opt_einsum format.
    ncon_labels: The ncon-labels of the corresponding contraction.

  Returns:
    Tuple[int]: The ncon contraction path.
  """
  if len(opteinsum_path) > 1:
    path = []
    tmp = list(ncon_labels)

    for pair in opteinsum_path:
      path.extend(list(set(tmp[pair[0]]).intersection(tmp[pair[1]])))
      new = set(tmp[pair[0]]).union(tmp[pair[1]]) - set(
          tmp[pair[0]]).intersection(tmp[pair[1]])
      for p in reversed(sorted(pair)):
        tmp.pop(p)
      tmp.append(new)
    return tuple(path)
  return None


def compute_contraction_order(building_blocks,
                              ncon_labels,
                              algorithm='greedy') -> Tuple[int]:
  """
  Compute a contraction oder of a tensor network given by `building_blocks`
  and `ncon_labels`. The path is computed using the opt_einsum package.

  Args:
    building_blocks: The tensors to contract.
    ncon_labels: The ncon labels of the contraction.
    algorithm: The algorithm to be used to find the contraction path.
      Can be any of {'greedy','branch','optimal'}.

  Returns:
    Tuple[int]: The contraction path in ncon format.
  """

  algorithms = {
      'greedy': opt_einsum.paths.greedy,
      'optimal': opt_einsum.paths.dynamic_programming,
      'branch': opt_einsum.paths.branch
  }
  if algorithm not in algorithms:
    raise ValueError(f'algorithm {algorithm} is not supported')
  inputs = [set(labels) for labels in ncon_labels]
  output = [{l for l in labels if l < 0} for labels in ncon_labels]
  output = set(flatten([o for o in output if len(o) > 0]))

  shapes = [g.shape for g in building_blocks]
  size_dict = {}
  for shape, labels in zip(shapes, ncon_labels):
    size_dict.update(dict(zip(labels, shape)))
  path = algorithms[algorithm](inputs, output, size_dict)

  ncon_path = opt_einsum_to_ncon_path(tuple(path),
                                      utils.to_tuples_of_ints(ncon_labels))
  return ncon_path


def flatten(ll: List[List[Any]]) -> List[Any]:
  """
  Flatten a list of lists into a single list.
  """
  return [a for l in ll for a in l]
