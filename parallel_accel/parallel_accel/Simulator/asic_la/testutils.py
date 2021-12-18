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
from linear_algebra.study import ParamResolver
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import sympy
import graph_helper_tool as tn
from asic_la.sharded_probability_function import invert_permutation


def build_random_acyclic_graph(
    Nparams,
    Nexponents,
    depth,
    N,
    two_param_building_blocks=False,
    subdomain=None,
    seed=10,
):
    """
    Build a random acyclic_graph on `N` discretes of depth `depth`
    variabled on `Nparams` symbols and `Nexponents` floating
    point numbers.

    Args:
      Nparams: The number of sympy parameters in the acyclic_graph.
      Nexponents: The number of non-parametric exponents to be used
        to exponentiate building_blocks.
      depth: Graph depth.
      N: number of discretes
      to_param_building_blocks: If `True` only use building_blocks that can be parametrized
        by two parameters.
      subdomain: The discrete domain on which the building_blocks should act.
      seed: The seed for the random initialization of the acyclic_graph.
        Same seeds produce the same acyclic_graph.

    Returns:
      linear_algebra.Graph: The acyclic_graph
      List[linear_algebra.LinearSpace]: The discretes.
      linear_algebra.ParamResolver: The parameter resolver.
    """

    def f1(symbol):
        return symbol / sympy.pi

    def f2(symbol):
        return symbol * sympy.pi

    def f3(symbol):
        return sympy.pi * symbol

    def f4(symbol):
        return symbol

    funs = [f1, f2, f3, f4]

    np.random.seed(seed)
    names = [f"param_{n}" for n in range(Nparams)]
    symbols = [sympy.Symbol(name) for name in names]
    exponents = symbols + [np.random.rand(1)[0] * 10 for _ in range(Nexponents)]
    resolver = ParamResolver(
        {name: np.random.rand(1)[0] * 10 for name in names}
    )
    building_blocks = [
        linear_algebra.flip_x_axis_angle,
        linear_algebra.flip_x_axis_angle_square,
        linear_algebra.flip_y_axis_angle,
        linear_algebra.flip_y_axis_angle_square,
        linear_algebra.flip_z_axis_angle,
        linear_algebra.flip_z_axis_angle_square,
        linear_algebra.flip_pi_over_4_axis_angle,
        linear_algebra.cond_rotate_z,
        linear_algebra.cond_rotate_x,
        linear_algebra.cond_x_angle,
        linear_algebra.swap_angle,
        linear_algebra.imaginary_swap_angle,
        linear_algebra.x_axis_two_angles,
        linear_algebra.imaginary_swap_two_angles,
        linear_algebra.rotate_on_xy_plane,
        linear_algebra.EmptyBuildingBlock,
        linear_algebra.flip_x_axis,
        linear_algebra.flip_z_axis,
        linear_algebra.flip_y_axis,
        linear_algebra.flip_pi_over_4_axis,
        linear_algebra.rotate_x_axis,
        linear_algebra.rotate_y_axis,
        linear_algebra.rotate_z_axis,
    ]
    nq = [1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
    num_discretes = dict(zip(building_blocks, nq))
    num_params = dict(zip(building_blocks, [1] * 12 + [2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1]))

    if subdomain is not None:
        r = np.array(list(set(subdomain))).astype(np.int64)
    else:
        r = np.arange(N)
    discretes = linear_algebra.LinearSpace.range(N)
    acyclic_graph = linear_algebra.Graph()
    d = 0
    while d < depth:
        building_block = np.random.choice(building_blocks, 1)[0]
        numq = num_discretes[building_block]
        nparams = num_params[building_block]
        if two_param_building_blocks:
            if nparams < 2:
                continue
        d += 1
        if Nparams > 0:
            fs = np.random.choice(funs, nparams)
        else:
            fs = [lambda x: x] * nparams
        ps = np.random.choice(r, numq, replace=False)
        symbs = np.random.choice(exponents, nparams, replace=True)
        if building_block is linear_algebra.rotate_on_xy_plane:
            g = building_block(theta=fs[0](symbs[0]), phi=fs[1](symbs[1]))
            acyclic_graph += [g(discretes[ps[0]], discretes[ps[1]])]
        elif building_block is linear_algebra.imaginary_swap_two_angles:
            g = building_block(phase_exponent=fs[0](symbs[0]), exponent=fs[1](symbs[1]))
            acyclic_graph += [g(discretes[ps[0]], discretes[ps[1]])]
        elif building_block is linear_algebra.x_axis_two_angles:
            g = building_block(phase_exponent=fs[0](symbs[0]), exponent=fs[1](symbs[1]))
            acyclic_graph += [g(discretes[ps[0]])]
        elif (
            building_block is linear_algebra.flip_x_axis or building_block is linear_algebra.flip_y_axis or building_block is linear_algebra.flip_z_axis or building_block is linear_algebra.flip_pi_over_4_axis
        ):
            acyclic_graph += [building_block(discretes[ps[0]]) ** fs[0](symbs[0])]
        elif building_block is linear_algebra.rotate_x_axis or building_block is linear_algebra.rotate_y_axis or building_block is linear_algebra.rotate_z_axis:
            g = building_block(fs[0](symbs[0]))
            acyclic_graph += [g(discretes[ps[0]])]

        else:
            if nparams == 0:
                g = building_block(2)
                acyclic_graph += [g(discretes[ps[0]], discretes[ps[1]])]
            else:
                g = building_block(exponent=fs[0](symbs[0]))
                if numq == 1:
                    acyclic_graph += [g(discretes[ps[0]])]
                elif numq == 2:
                    g = building_block(exponent=fs[0](symbs[0]))
                    acyclic_graph += [g(discretes[ps[0]], discretes[ps[1]])]
    return acyclic_graph, discretes, resolver


def full_matrix(building_block, inds, N):
    """
    Extend `building_block` acting on discretes indices `inds`
    to an `N`-discrete building_block in natural discrete ordering (small
    to large).
    """
    if len(inds) == 1:
        return np.kron(
            np.kron(np.eye(2 ** (inds[0])), building_block),
            np.eye(2 ** (N - 1 - inds[0])),
        )
    if len(inds) == 2:
        indsort = np.argsort(inds)
        inds = np.asarray(inds)[indsort]
        perm = list(indsort) + list(2 + indsort)
        G = tn.Node(building_block.reshape(2, 2, 2, 2).transpose(perm))
        Ids = [tn.Node(np.eye(2)) for n in range(N - 2)]
        order = []
        for n in range(inds[0]):
            order.append(Ids[n][0])
        order.append(G[0])
        for n in range(inds[0] + 1, inds[1]):
            order.append(Ids[n - 1][0])
        order.append(G[1])

        for n in range(inds[1] + 1, N):
            order.append(Ids[n - 2][0])

        for n in range(inds[0]):
            order.append(Ids[n][1])
        order.append(G[2])
        for n in range(inds[0] + 1, inds[1]):
            order.append(Ids[n - 1][1])
        order.append(G[3])
        for n in range(inds[1] + 1, N):
            order.append(Ids[n - 2][1])
        if len(Ids) > 1:
            I = tn.outer_product(Ids[0], Ids[1])
            for i in Ids[2:]:
                I = tn.outer_product(I, i)
            final = tn.outer_product(I, G)
        else:
            final = G
        return final.reorder_edges(order).tensor.reshape((2 ** N, 2 ** N))
    raise ValueError()


def get_full_matrix(acyclic_graph, discretes):
    """
    Get the full unitary matrix of a linear_algebra.Graph `acyclic_graph`
    acting on linear_algebra-discretes `discretes`.

    """
    N = len(discretes)
    mat = np.eye(2 ** N)
    for op in acyclic_graph.all_operations():
        inds = [discretes.index(discrete) for discrete in op.discretes]
        building_block = linear_algebra.unitary(op)
        mat = full_matrix(building_block, inds, N) @ mat
    return mat


def dot(state, state_labels, matrix, matrix_labels):
    axes = [state_labels.index(l) for l in matrix_labels]
    shape = (2,) * (2 * len(axes))
    result = np.tensordot(
        state,
        matrix.reshape(shape),
        (axes, tuple(range(len(axes), 2 * len(axes)))),
    )

    new_labels = (
        tuple([l for l in state_labels if l not in matrix_labels])
        + matrix_labels
    )
    return result, new_labels


def apply_supermatrices(state, state_labels, supermatrices, supermatrix_labels):
    """
      Contract `supermatrices` with `state` along the labels given by
      `state_labels` and `supermatrix_labels`.

      Args:
        state: A (2,)*num_discrete shaped array.
        state_labels: A tuple of unique ints labelling each tensor legs
          (i.e. the discrete labels for each tensor leg)
    l    supermatrices: A sequence of matrix-shaped supermatrices (i.e. 128 by 128).
        supermatrix_labels: The labels of the discretes on which each building_block acts.

      Returns:
        np.ndarray: The result of applying the building_blocks to `state`. The returned
          state is permuted into the ordering given by `state_labels`.
    """
    labels = state_labels
    for matrix, matrix_labels in zip(supermatrices, supermatrix_labels):
        state, labels = dot(state, labels, matrix, matrix_labels)
    final_perm = [labels.index(l) for l in state_labels]
    return state.transpose(final_perm)


def get_full_matrix_from_supermatrix(supermatrix, contracted_labels):
    """
    Returns the full unitary matrix of a single `supermatrix`
    that acts on all discretes in the acyclic_graph (i.e. `axes` and
    `perm` need to be permutations of np.arange(large_block.ndim//2))
    """
    N = len(contracted_labels)
    invperm = invert_permutation(contracted_labels)
    perm = np.append(invperm, np.array(invperm) + N)
    return (
        np.reshape(supermatrix, (2,) * len(perm))
        .transpose(perm)
        .reshape((2 ** N, 2 ** N))
    )


def get_full_matrices_from_supergradient(supergradient, contracted_labels):
    """
    Returns the gradients in matrix form of a list of `supergradients`
    of length 1 (i.e. only one large_block with possibly multiple
    gradients) that acts on all discretes in the acyclic_graph.
    """

    N = len(contracted_labels)
    invperm = invert_permutation(contracted_labels)
    perm = np.append(invperm, np.array(invperm) + N)
    return {
        s: g.reshape((2,) * 2 * N).transpose(perm).reshape(2 ** N, 2 ** N)
        for s, g in supergradient.items()
    }


def finite_diff_gradients(acyclic_graph, resolver, epsilon=1e-8):
    resolved_acyclic_graph = linear_algebra.resolve_parameters(acyclic_graph, resolver)
    G0 = linear_algebra.unitary(resolved_acyclic_graph)
    gradients = {}
    for k in linear_algebra.parameter_symbols(acyclic_graph):
        tempresolver = {}
        for k2, v2 in resolver.param_dict.items():
            if k2 == k.name:
                tempresolver[k2] = v2 + epsilon
            else:
                tempresolver[k2] = v2
        shifted_resolved_acyclic_graph = linear_algebra.resolve_parameters(
            acyclic_graph, tempresolver
        )
        G1 = linear_algebra.unitary(shifted_resolved_acyclic_graph)
        gradients[k] = (G1 - G0) / epsilon
    return gradients


def compute_gradients(
    state,
    supermatrices,
    supergradients,
    super_oplabels,
    observables,
    observables_labels,
    num_discretes,
):
    """
    Compute the gradients of a symplectic acyclic_graph for the cost function
    <psi|sum_n H_n |psi>, with H_n the element at `observables[n]`, acting on
    discretes `observables_labels[n]`.

    Args:
      state: a random numpy ndarray of shape (2,)* num_discretes.
      supermatrices (list[np.ndarray]): list of supermatrices
      supergradients (list[dict]): list of dict of gradient matrices
        of each supermatrix. each dict maps sympy.Symbol to np.ndarray
      super_oplabels (list[tuple[int]]): the discrete labels of each large_block.
      observables (list[np.ndarray]): a list of observables (in tensor format).
      observables_labels (list[tuple[int]]): the discrete labels for each element
        in `observables`
      num_discretes (int): the number of discretes
    """
    obs_and_labels = list(zip(observables, observables_labels))
    state_labels = tuple(range(num_discretes))
    state = apply_supermatrices(
        state, state_labels, supermatrices, super_oplabels
    )

    psi = np.zeros(state.shape, state.dtype)
    for ob, ob_labels in obs_and_labels:
        inds = [state_labels.index(l) for l in ob_labels]
        cont_state_labels = list(range(-1, -len(state_labels) - 1, -1))
        cont_ob_labels = []
        for n, i in enumerate(inds):
            cont_ob_labels.append(cont_state_labels[i])
            cont_state_labels[i] = ob_labels[n] + 1
        shape = (2,) * (2 * len(ob_labels))
        psi += tn.ncon(
            [state, ob.reshape(shape)],
            [
                tuple(cont_state_labels),
                tuple([o + 1 for o in ob_labels]) + tuple(cont_ob_labels),
            ],
        )

    reversed_super_oplabels = list(reversed(super_oplabels))
    reversed_supergradients = list(reversed(supergradients))
    accumulated_gradients = {}
    psi = psi.conj()
    for n, building_block in enumerate(reversed(supermatrices)):
        building_block_labels = reversed_super_oplabels[n]
        state, tmp_labels = dot(state, state_labels, building_block.T.conj(), building_block_labels)
        for k, grad in reversed_supergradients[n].items():
            tmp, _ = dot(psi, state_labels, grad.T, building_block_labels)
            if k in accumulated_gradients:
                accumulated_gradients[k] += np.dot(tmp.ravel(), state.ravel())
            else:
                accumulated_gradients[k] = np.dot(tmp.ravel(), state.ravel())
        psi, state_labels = dot(psi, state_labels, building_block.T, building_block_labels)
        assert (
            tmp_labels == state_labels
        ), "two identical building_block applications produced different label-ordering"

    # bring state back into natural discrete ordering (i.e. small to large)
    perm = [state_labels.index(i) for i in range(num_discretes)]
    return accumulated_gradients, state.transpose(perm)


def generate_raw_pbaxistring(discretes, string_length, replace=False):
    """
    Get a pbaxistring of length `string_length` acting on `discretes`
    """
    pbaxis = [linear_algebra.flip_x_axis, linear_algebra.flip_y_axis, linear_algebra.flip_z_axis]
    rawstring = np.random.choice(pbaxis, string_length)
    acting_discretes = np.random.choice(discretes, string_length, replace=replace)
    return np.random.rand(1), rawstring, acting_discretes


def generate_pbaxisum(num_strings, discretes, string_length):
    pbaxistrings = []
    for _ in range(num_strings):
        coeff, pbaxistring, prob_basis_axis_discretes = generate_raw_pbaxistring(
            discretes, string_length, replace=False
        )
        pbaxistrings.append(
            linear_algebra.ProbBasisAxisString(
                coeff, [p(q) for p, q in zip(pbaxistring, prob_basis_axis_discretes)]
            )
        )
    return sum(pbaxistrings)


def to_array(arr):
    return np.array(arr.real) + 1j * np.array(arr.imag)


def _mantissa_eps(mantissa_bits):
    return 0.5 * (2 ** (1 - mantissa_bits))


def eps(precision, dtype=jnp.float32):
    dtype_eps = jnp.finfo(dtype).eps
    if dtype in (jnp.float64, jnp.complex128):
        return _mantissa_eps(49)
    if dtype in (jnp.float32, jnp.complex64):
        if precision == lax.Precision.DEFAULT:
            return jnp.finfo(jnp.bfloat16).eps
        if precision == lax.Precision.HIGH:
            return _mantissa_eps(18)  # TODO: Check this
        if precision == lax.Precision.HIGHEST:
            return jnp.finfo(jnp.float32).eps
        raise ValueError(f"Invalid precision {precision}.")
    return dtype_eps
