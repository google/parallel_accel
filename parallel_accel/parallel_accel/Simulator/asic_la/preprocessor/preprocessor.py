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
import functools as fct
import time
from typing import (Any, List, Tuple, Dict, Optional, Sequence, Union)

import jax.numpy as jnp
import jax
import jaxlib
import numpy as np
import sympy
import graph_helper_tool as tn

from parallel_accel.shared import logger
from asic_la import parser
from asic_la import utils

import asic_la.config as config

import asic_la.sharded_probability_function.complex_workaround as cw
from asic_la.preprocessor import network

JAX_PREPRO_BACKEND = parser.JAX_PREPRO_BACKEND
MAX_CACHE_SIZE = config.MAX_CACHE_SIZE

Array = Any
log = logger.get_logger(__name__)
JAXARRAY = jaxlib.xla_extension.DeviceArray

@fct.lru_cache(maxsize=MAX_CACHE_SIZE)
def _preprocess(
    axes: Sequence[Sequence[int]],
    num_discretes: int,
    max_discrete_support: int = 7,
    maxlength: Optional[int] = None
) -> Tuple[Tuple[Tuple[int]], Tuple[Tuple[int]], List[List[List[int]]]]:
  """
  Preprocess the acyclic_graph by merging building_blocks into discrete building_blocks with
  support on `max_discrete_support` discretes. `axes` are the discretes on which
  each building_block in the acyclic_graph acts, in the order in which the building_blocks are applied
  to the probabilityfunction of the acyclic_graph. This routine does not perform any
  contractions. The returned values can be used to contract building_block-collections
  (representing a large_block) into an actual array, using the ncon
  API. That is,

  ```python
  import graph_helper_tool as tn # implements the ncon API

  building_blocks, gradients, axes = parser.parse(acyclic_graph,...) #parse a acyclic_graph

  large_block_ids, superaxes, ncon_labels = _perprocess(axes, num_discretes)

  #perform the contraction of each supernode into an actual array
  large_blocks = []
  for ids, labels in zip(supernode_ids, ncon_labels):
   large_block = tn.ncon([building_blocks[n] for n in ids], labels)
   large_blocks.append(large_block)

  # large_blocks contains actual numpy-arrays.
  ```

  Args:
      axes: A tuple of tuples of ints. Each inner
      tuple contains the linear locations of the discretes in the
      original sequence of discretes on which the building_block acts.
    num_discretes: number of discretes in the simulation.
    target_support: the targeted support-size of merged building_blocks.
    maxlength: The maximally allowed number of building_blocks per large_block.
      If `None`, no limit is applied.

  Returns:
    large_block_ids: Tuple of tuple of int of building_block ids representing each large_block.
    superaxes: A tuple of tuple of ints holding the discrete labels
      on which the corresponding large_block acts. Discreted labels here refers to a
      fixed labelling of the original discretes used in the preprocessor. That is,
      if the original discretes were given as a (not necessarily ordered) sequence
      `discretes`, the labels are given by
      ```
      labels = [str(q) for q in range(len(discretes))]
      ```
      i.e. each discrete in `discretes` gets tagged with a label according to its
      position in `discretes`. The returned acyclic_graph is defined on these auxilliary
      labels.
    ncon_labels: A list of list of list of int that can be used to contract the
      building_block-collections into large_blocks using the `ncon` API.
  """
  acyclic_graph_network = network.Network.from_axes(axes, num_discretes)
  return acyclic_graph_network.fuse(max_discrete_support, maxlength)


@jax.partial(jax.jit, static_argnums=(1, 2), backend=JAX_PREPRO_BACKEND)
@jax.partial(jax.vmap, in_axes=(0, None, None))
def vncon(building_blocks: Tuple[Tuple[np.ndarray]], ncon_labels: Tuple[Tuple[int]],
          path: Tuple[int]) -> Union[np.ndarray, JAXARRAY]:
  return tn.ncon(building_blocks, ncon_labels, con_order=path, backend='jax')


def _compute_building_blocks(
    building_blocklist: Sequence[Sequence[np.ndarray]],
    ncon_labels: Sequence[Sequence[int]], path: Sequence[int],
    symbols: List[str], symbol_map: Dict[str, sympy.Symbol],
    support: int) -> Union[np.ndarray, Dict[sympy.Symbol, np.ndarray]]:
  """
  Compute large_blocks and supergradients by contracting all building_blocks in
  building_blockslist. This function either uses a vmap-jit approach in jax, or
  plain numpy to perform contractions, depending on if a large_block has
  a non-trivial gradient.

  The rationale here is roughly as follows: assume that a given large_block
  consists of `N` building_blocks, out which `L` <= `N` building_blocks depend on one of
  `k` <= `L` parameters (parameters need not be different between different
  building_blocks). To compute the gradients of the large_block, we iterate through the
  `L` different variabled building_blocks `g_l`, `l=0,1,..,L-1`. In each iteration
  we replace `g_l` in the supernode network with its gradient, and contract
  the resulting network. This way we end up with `L` distinct contractions,
  each of which though has exactly the same topology. Out of these `L`
  contraction results we need to sum up all sub-groups belonging to the same
  parameter. After that we end up with `k` gradients.

  `building_blocklist` here contains the building_blocks of each of these contractions. The
  outer sequence iterates through different values of each building_block per contraction,
  and the inner sequence iterates through the different contractions (i.e. the
  loop over `l` above)

  The contraction problem described above is trivially parallelizable. Hence,
  to improve the preprocessing performance, we use jax.vmap to parallelize
  these contractions.


  Args:
    building_blocklist: A list of list numpy arrays.
    ncon_labels: The contraction labels of `building_blocklist` in ncon format.
    path: A contraction path in ncon format.
    symbols: A list of names of sympy symbols, one for each contraction.
    symbol_map: A map of symbol names to sympy.Symbol.
    support: The discrete-support of building_blocklist.

  Returns:
    np.ndarray: The large_block.
    Dict[sympy.Symbol, np.ndarray]: The gradient of `large_block` with
      respect to each sympy.Symbol on which `large_block` depends.
  """
  supergradient = {}
  if len(symbols) > 0:
    stacked_building_blocks = [np.stack(g) for g in building_blocklist]
    stackedresult = np.array(
        vncon(stacked_building_blocks, utils.to_tuples_of_ints(ncon_labels), path))
    large_block = stackedresult[0].reshape(2**support, 2**support)
    stackedresult = stackedresult[1:, ...]
    symbols = np.asarray(symbols)
    for u in np.unique(symbols):
      mask = symbols == u
      supergradient[symbol_map[u]] = np.sum(stackedresult[mask],
                                            axis=0).reshape(
                                                2**support, 2**support)
  else:
    tmpbuilding_blocks = [g[0] for g in building_blocklist]
    large_block = tn.ncon(tmpbuilding_blocks,
                        ncon_labels,
                        con_order=path,
                        backend='numpy').reshape(2**support, 2**support)
  return large_block, supergradient


def preprocess(
    building_blocks: Sequence[Sequence[np.ndarray]],
    gradients: Sequence[Dict[sympy.Symbol, np.ndarray]],
    axes: Sequence[Sequence[int]],
    num_discretes: int,
    max_discrete_support: int = 7,
    quiet: bool = False
) -> Tuple[Tuple[np.ndarray], Tuple[Dict[sympy.Symbol, np.ndarray]],
           Tuple[Tuple[int]]]:
  """
  Preprocess the acyclic_graph by merging building_blocks and their derivatives
  into discrete building_blocks with support on `max_discrete_support` discretes. Gates
  are represented as matrices of shape (2**max_discrete_support, 2**max_discrete_support)

  Args:
    building_blocks: Sequence of numpy.ndarrays, the symplectic building_blocks.
    gradients: Sequence of dict of symplecticbuilding_block derivatives.
      Each element in `gradients` is a dictionary mapping
      sympy.Symbols on which the specific building_block depends to
      the derivative of the building_block with respect to that symbol.
    axes: A tuple of tuples of ints. Represents the acyclic_graph
      building_block topology.
    num_discretes: Number of discretes in the simulation.
    max_discrete_support: the targeted support-size of merged building_blocks.
    quiet: Flag that suppresses logging when True. Default is False.

  Returns:
    large_blocks: The (2**dim, 2**dim) shaped result of merging several building_blocks,
     with `dim <= max_discrete_support`. The resulting matrix of building_blocks is returned
     in "natural" transposition, i.e. such that axis 1 of the matrix is supposed
     to be contracted with the state.
    derivatives: A `list` of `dict`. The dict at position `n` in the list
      maps the symbols on which the building_block at `building_blocks[n]` depends to the derivative
      of `building_blocks[n]` with respect to that symbol.
    operating_axes: A tuple of tuple of ints holding the discrete labels
      on which the corresponding large_block acts. Discreted labels here refers to a
      fixed labelling of the original discretes used in the preprocessor. That is,
      if the original discretes were given as a (not necessarily ordered) sequence
      `discretes`, the labels are given by
      ```
      labels = [str(q) for q in range(len(discretes))]
      ```
      i.e. each discrete in `discretes` gets tagged with a label according to its
      position in `discretes`. The returned acyclic_graph is defined on these auxilliary
      labels.

  """
  config.set_jax_config_to_cpu()
  t0 = time.time()
  if max_discrete_support < 2:
    raise ValueError(f"targetted building_block support has to be at least 2,"
                     f" got max_discrete_support={max_discrete_support}")
  large_block_ids, superaxes, ncon_labels = _preprocess(
      axes, num_discretes, max_discrete_support=max_discrete_support)
  large_blocks = []
  supergradients = []
  symbol_map = {}
  for ids, a, labels in zip(large_block_ids, superaxes, ncon_labels):
    building_blocklist = [[] for n in ids]
    _building_blocks = [building_blocks[node] for node in ids]
    [building_blocklist[m].append(g) for m, g in enumerate(_building_blocks)]
    path = network.compute_contraction_order(_building_blocks, labels)

    symbols = []

    for n, node in enumerate(ids):
      for symbol, gradbuilding_block in gradients[node].items():
        symbol_map[symbol.name] = symbol
        tmpbuilding_blocks = list(_building_blocks)
        tmpbuilding_blocks[n] = gradbuilding_block
        symbols.append(symbol.name)
        [building_blocklist[m].append(g) for m, g in enumerate(tmpbuilding_blocks)]

    supergradient = {}
    # if the number of symbols is small using plain numpy is faster
    support = len(a)
    large_block, supergradient = _compute_building_blocks(building_blocklist, labels, path,
                                              symbols, symbol_map,
                                              support)
    large_blocks.append(large_block)
    supergradients.append(supergradient)

  if not quiet:
    log.info('preprocessing finished',
             num_building_blocks=len(building_blocks),
             num_grad_building_blocks=len([g for g in gradients if len(g) > 0]),
             num_params=len({s for sg in supergradients for s in sg.keys()}),
             num_discretes=num_discretes,
             num_super_building_blocks=len(large_block_ids),
             num_super_gradients=sum([len(m) for m in supergradients]),
             preprocessing_time=time.time() - t0)
  config.reset_jax_to_former_config()
  return tuple(large_blocks), tuple(supergradients), superaxes


def preprocess_pbaxisums(
    prob_basis_axis_sums: Sequence[Sequence[Sequence[Array]]],
    opaxes: Sequence[Sequence[Sequence[Tuple[int]]]],
    num_discretes: int,
    max_discrete_support=7
) -> Tuple[Tuple[Tuple[Tuple[Array]]], Tuple[Tuple[Tuple[Tuple[int]]]]]:
  """
  Preprocess parsed prob-basis-axis-sums, i.e. the result of `parser.parse_pbaxisums`.

  Args:
    prob_basis_axis_sums: Sequence of sequence of sequence of single discrete matrix-building_blocks.
      A single pbaxistring is represented as the innermost sequence of
      matrix-building_blocks. The outermost sequence iterates through different
      prob-basis-axis-sums, the intermediate sequence iterates through pbaxistrings
      within a pbaxisum.
    opaxes: nested sequence of tuples of int. The innermost sequence collects
      the operating labels of a single pauli string. The outermost sequence
      iterates through individual pauli sums, the intermediate sequence
      iterates through the pauli strings of a particular pauli sum.

  Returns:
    Tuple[Tuple[Tuple[Array]]]: Superbuilding_block representation
      of the pauli sums in `prob_basis_axis_sums`. The nested list structure is the
      same as for `prob_basis_axis_sums`
    Tuple[Tuple[Tuple[Tuple[int]]]]: Discreted labels for the large_block
      representation of the pauli sums in `prob_basis_axis_sums`. The nested
      list structure is the same as for 'opaxes`

  """
  t0 = time.time()
  large_blocks = []
  operating_axes = []
  for prob_basis_axis_building_blocks, prob_basis_axis_axes in zip(prob_basis_axis_sums, opaxes):
    tmplarge_blocks = []
    tmpaxes = []
    for building_blocks, axes in zip(prob_basis_axis_building_blocks, prob_basis_axis_axes):
      _large_blocks, _, opaxes = preprocess(building_blocks=building_blocks,
                                        gradients=[dict()] * len(building_blocks),
                                        axes=axes,
                                        num_discretes=num_discretes,
                                        max_discrete_support=max_discrete_support,
                                        quiet=True)
      tmplarge_blocks.append(_large_blocks)
      tmpaxes.append(opaxes)
    large_blocks.append(tmplarge_blocks)
    operating_axes.append(tmpaxes)
    log.info('pbaxisum preprocessing finished',
             num_pbaxis=len(prob_basis_axis_sums),
             num_discretes=num_discretes,
             preprocessing_time=time.time() - t0)

  return tuple(large_blocks), tuple(operating_axes)


def canonicalize_gradients(
    gradients: Sequence[Dict[sympy.Symbol, np.ndarray]],
    broadcasted_shape: Optional[int] = None
) -> Tuple[Sequence[Dict[int, cw.ComplexDeviceArray]], Dict[sympy.Symbol, int]]:
  """
  Canonicalize `gradients` by mapping the sympy.Symbol
  keys of the dicts in `gradients` to integers
  and by packing complex arrays into a ComplexDeviceArray.
  Possibly also broadcast the arrays to a new dimension.

  Args:
    gradients: Sequence of dict of symplecticbuilding_block derivatives.
      Each element in `gradients` is a dictionary mapping
      sympy.Symbols on which  the specific building_block depends to
      the derivative of the building_block with respect to that symbol.
    broadcasted_shape: Optional integer for specifying
      a broadcast dimension. The returend arrays
      will be broadcast along their zeroth axis with
      the dimension of this axis given by this integer.

  Returns:
    Sequence[Dict[int, ComplexDeviceArray]]: The canonicalized gradient mapping.
    Dict[sympy.Symbol, int]: Mapping from sympy.Symbols `gradients`
      to integers used for canonicalization.
  """
  all_symbols = set()
  canonicalized_gradients = []
  for g in gradients:
    all_symbols |= set(g.keys())
  symbol_map = {s: n for n, s in enumerate(all_symbols)}
  if broadcasted_shape is None:
    for grad in gradients:
      packed_grad = [
          tuple([
              symbol_map[s],
              cw.ComplexDeviceArray(jnp.array(g.real), jnp.array(g.imag))
          ]) for s, g in grad.items()
      ]
      canonicalized_gradients.append(packed_grad)
  else:
    for grad in gradients:
      packed_grad = []
      for s, g in grad.items():
        shape = (broadcasted_shape,) + g.shape
        broadcasted = cw.ComplexDeviceArray(
            jnp.array(np.broadcast_to(g.real, shape)),
            jnp.array(np.broadcast_to(g.imag, shape)))
        encoded_symbols = jnp.full(shape=broadcasted_shape,
                                   fill_value=symbol_map[s],
                                   dtype=np.int32)
        packed_grad.append((encoded_symbols, broadcasted))
      canonicalized_gradients.append(packed_grad)

  return canonicalized_gradients, symbol_map


def canonicalize_building_blocks(building_blocks: Sequence[np.ndarray],
                       broadcasted_shape:Optional[int]=None) -> Tuple[cw.ComplexDeviceArray]:
  """
  Canonicalize `building_blocks` by mapping them from a sequence of
  complex arrays to a tuple of ComplexDeviceArray, possibly
  broadcasting the arrays to a new dimension.

  Args:
    building_blocks: Nested list or tuple of np.ndarrays
    broadcasted_shape: Optional integer for specifying
      a broadcast dimension. The returend arrays
      will be broadcast along their zeroth axis with
      the dimension of this axis given by this integer.

  Returns:
    Tuple[cw.ComplexDeviceArray]: The canonicalized building_blocks.

  """
  if isinstance(building_blocks, (list, tuple)):
    res = []
    for g in building_blocks:
      res.append(canonicalize_building_blocks(g, broadcasted_shape=broadcasted_shape))
    return tuple(res)

  if broadcasted_shape is None:
    return cw.ComplexDeviceArray(jnp.array(building_blocks.real), jnp.array(building_blocks.imag))

  return cw.ComplexDeviceArray(
      jnp.array(np.broadcast_to(building_blocks.real,
                                (broadcasted_shape,) + building_blocks.shape)),
      jnp.array(np.broadcast_to(building_blocks.imag,
                                (broadcasted_shape,) + building_blocks.shape)))
