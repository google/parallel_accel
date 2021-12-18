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
"""Util methods for the asic_simulator."""

import numpy as np
import graph_helper_tool as tn
import linear_algebra


def unpackbits(x, num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits).reshape([1, num_bits])
    return (
        (x & mask).astype(np.bool).astype(np.int32).reshape(xshape + [num_bits])
    )


def to_ints(values):
    "Returns a tuple of ints"
    return canonicalize_ints(values)


def to_tuples_of_ints(values):
    """
    Cast list of list of numbers or
    tuple of tuple of numbers
    to a tuple of tuples of ints.
    """
    return canonicalize_ints(values)


def canonicalize_ints(values):
    """
    Cast a nested sequence of numbers
    to a nested tuple of ints.
    """
    if isinstance(values, (list, tuple)):
        res = []
        for v in values:
            res.append(canonicalize_ints(v))
        return tuple(res)
    return int(values)


def canonicalize_values(vals, broadcasted_shape):
    """
    broadcast np.ndarrays at the leaves
    of a nested list or nested tuple to
    `broadcasted_shape`.
    """

    if isinstance(vals, (list, tuple)):
        res = []
        for val in vals:
            res.append(
                canonicalize_values(val, broadcasted_shape=broadcasted_shape)
            )
        return tuple(res)
    return np.broadcast_to(vals, (broadcasted_shape, len(vals)))


def invperm(axes, N):
    """
    Helper function. Compute the permutation that
    lead to `axes`, given `N` discretes.
    """
    axes = list(axes)
    perm = np.arange(N)
    building_block_labels = np.array(perm[-len(axes) :])
    other_labels = np.sort(list(set(list(range(N))) - set(building_block_labels)))
    other_axes = list(np.sort(list(set(list(range(N))) - set(axes))))
    perm[axes] = building_block_labels
    perm[other_axes] = other_labels
    return perm


class RemoveObservations(linear_algebra.PointOptimizer):
    """
    Helper class for removing all observations from a symplectic acyclic_graph.
    """

    def optimization_at(
        self, acyclic_graph: linear_algebra.Graph, index: int, op: linear_algebra.Operation
    ):
        if isinstance(op.building_block, linear_algebra.ObservationGate):
            return linear_algebra.PointOptimizationSummary(
                clear_span=1, new_operations=[], clear_discretes=op.discretes
            )
        return None


def remove_observations(acyclic_graph):
    """
    Remove all observations from a symplectic acyclic_graph.
    """
    result = acyclic_graph.copy()
    RemoveObservations().optimize_acyclic_graph(result)
    return result
