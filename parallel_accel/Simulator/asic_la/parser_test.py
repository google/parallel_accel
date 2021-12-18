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
import pytest
import jax
import numpy as np
import sympy
import linear_algebra
from linear_algebra.study import ParamResolver
from asic_la.parser import (
    get_unitary,
    get_autodiff_gradient,
    get_finitediff_gradient,
    parse,
    assert_is_allowed_expression,
    SIMPLE_BUILDING_BLOCKS,
)
from asic_la.testutils import build_random_acyclic_graph


jax.config.update("jax_enable_x64", True)
eigen_building_blocks = sorted(SIMPLE_BUILDING_BLOCKS, key=repr)
prob_basis_axis_building_blocks = [linear_algebra.flip_x_axis, linear_algebra.flip_y_axis, linear_algebra.flip_z_axis, linear_algebra.flip_pi_over_4_axis]
rotation_building_blocks = [linear_algebra.rotate_x_axis, linear_algebra.rotate_y_axis, linear_algebra.rotate_z_axis]
phased_building_blocks = [linear_algebra.x_axis_two_angles, linear_algebra.imaginary_swap_two_angles]


def test_assert_is_allowed_expression():
    pi = sympy.pi
    a, b = sympy.symbols("a b")
    with pytest.raises(ValueError):
        assert_is_allowed_expression(a ** 2)
    with pytest.raises(ValueError):
        assert_is_allowed_expression(a + b)
    with pytest.raises(ValueError):
        assert_is_allowed_expression(a / b)
    with pytest.raises(ValueError):
        assert_is_allowed_expression(2 / b)
    with pytest.raises(ValueError):
        assert_is_allowed_expression(pi / b)

    assert_is_allowed_expression(a + a)
    assert_is_allowed_expression(a * 2)
    assert_is_allowed_expression(2 * a * 2)
    assert_is_allowed_expression(pi * a * pi)
    assert_is_allowed_expression(pi * a * pi ** 2)
    assert_is_allowed_expression(pi ** 2 * a * pi)
    assert_is_allowed_expression(2 * a)
    assert_is_allowed_expression(a / 2)
    assert_is_allowed_expression(a / pi)


@pytest.mark.parametrize("seed", np.arange(10))
@pytest.mark.parametrize("linear_algebra_building_block", rotation_building_blocks)
def test_get_unitary_rotation_regular(linear_algebra_building_block, seed):
    np.random.seed(seed)
    resolver = ParamResolver({})
    phase = (np.random.rand(1) - 0.5) * 100
    building_block = linear_algebra_building_block(phase)
    actual = get_unitary(building_block, resolver)
    expected = linear_algebra.unitary(building_block)
    eps = np.finfo(actual.dtype).eps * 500
    np.testing.assert_allclose(expected, actual, atol=eps, rtol=eps)


@pytest.mark.parametrize("seed", np.arange(10))
@pytest.mark.parametrize("linear_algebra_building_block", prob_basis_axis_building_blocks)
def test_get_unitary_prob_basis_axis_regular(linear_algebra_building_block, seed):
    np.random.seed(seed)
    resolver = ParamResolver({})
    building_block = linear_algebra_building_block
    actual = get_unitary(building_block, resolver)
    expected = linear_algebra.unitary(building_block)
    eps = np.finfo(actual.dtype).eps * 500
    np.testing.assert_allclose(expected, actual, atol=eps, rtol=eps)


@pytest.mark.parametrize("linear_algebra_building_block", eigen_building_blocks)
@pytest.mark.parametrize("seed", np.arange(10))
def test_get_unitary_eigen(linear_algebra_building_block, seed):
    t = sympy.Symbol("time")
    np.random.seed(seed)
    resolver = ParamResolver({"time": np.random.rand(1)[0] * 100.0})
    building_block = linear_algebra_building_block(exponent=t)
    actual = get_unitary(building_block, resolver)
    expected = linear_algebra.unitary(linear_algebra.resolve_parameters(building_block, resolver))
    eps = np.finfo(actual.dtype).eps * 500
    np.testing.assert_allclose(expected, actual, atol=eps, rtol=eps)


@pytest.mark.parametrize("linear_algebra_building_block", phased_building_blocks)
@pytest.mark.parametrize("seed", np.random.randint(0, 100000, 102))
def test_get_unitary_phased(linear_algebra_building_block, seed):
    t = sympy.Symbol("time")
    p = sympy.Symbol("phase")
    np.random.seed(seed)
    resolver = ParamResolver(
        {
            "time": np.random.rand(1)[0] * 100.0,
            "phase": np.random.rand(1)[0] * 100.0,
        }
    )
    building_block = linear_algebra_building_block(phase_exponent=p, exponent=t)
    actual = get_unitary(building_block, resolver)
    expected = linear_algebra.unitary(linear_algebra.resolve_parameters(building_block, resolver))
    eps = np.finfo(actual.dtype).eps * 500
    np.testing.assert_allclose(expected, actual, atol=eps, rtol=eps)


@pytest.mark.parametrize("seed", np.arange(10))
def test_get_unitary_fsimbuilding_block(seed):
    t = sympy.Symbol("theta")
    p = sympy.Symbol("phi")
    np.random.seed(seed)
    resolver = ParamResolver(
        {
            "theta": np.random.rand(1)[0] * 100.0,
            "phi": np.random.rand(1)[0] * 100.0,
        }
    )
    building_block = linear_algebra.rotate_on_xy_plane(theta=t, phi=p)
    actual = get_unitary(building_block, resolver)
    expected = linear_algebra.unitary(linear_algebra.resolve_parameters(building_block, resolver))
    eps = np.finfo(actual.dtype).eps * 500
    np.testing.assert_allclose(expected, actual, atol=eps, rtol=eps)


@pytest.mark.parametrize("linear_algebra_building_block", prob_basis_axis_building_blocks)
@pytest.mark.parametrize("seed", np.random.randint(0, 100000, 2))
def test_autodiff_gradient_pauli(linear_algebra_building_block, seed):
    t = sympy.Symbol("time")
    np.random.seed(seed)
    eps = 1e-7
    tol = 1e-6
    resolver = ParamResolver({"time": np.random.rand(1)[0] * 100.0})
    building_block = linear_algebra_building_block ** t
    finitediff = get_finitediff_gradient(building_block, resolver, eps)
    grads = get_autodiff_gradient(building_block, resolver)
    g = {k.name: v for k, v in grads.items()}
    for k, expected in finitediff.items():
        actual = g[k]
        np.testing.assert_allclose(expected, actual, atol=tol, rtol=tol)


@pytest.mark.parametrize("linear_algebra_building_block", prob_basis_axis_building_blocks)
@pytest.mark.parametrize("seed", np.arange(10))
def test_autodiff_gradient_prob_basis_axis_regular(linear_algebra_building_block, seed):
    np.random.seed(seed)
    t = (np.random.randint(1) - 0.5) * 100
    resolver = ParamResolver({"time": np.random.rand(1)[0] * 100.0})
    building_block = linear_algebra_building_block ** t
    grads = get_autodiff_gradient(building_block, resolver)
    assert len(grads) == 0


@pytest.mark.parametrize("linear_algebra_building_block", rotation_building_blocks)
@pytest.mark.parametrize("seed", np.arange(10))
def test_autodiff_gradient_rotation(linear_algebra_building_block, seed):
    t = sympy.Symbol("time")
    np.random.seed(seed)
    eps = 1e-7
    tol = 1e-6
    resolver = ParamResolver({"time": np.random.rand(1)[0] * 100.0})
    building_block = linear_algebra_building_block(t)
    finitediff = get_finitediff_gradient(building_block, resolver, eps)
    grads = get_autodiff_gradient(building_block, resolver)
    g = {k.name: v for k, v in grads.items()}
    for k, expected in finitediff.items():
        actual = g[k]
        np.testing.assert_allclose(expected, actual, atol=tol, rtol=tol)


@pytest.mark.parametrize("linear_algebra_building_block", rotation_building_blocks)
@pytest.mark.parametrize("seed", np.arange(10))
def test_autodiff_gradient_rotation_regular(linear_algebra_building_block, seed):
    np.random.seed(seed)
    t = (np.random.randint(1) - 0.5) * 100
    resolver = ParamResolver({"time": np.random.rand(1)[0] * 100.0})
    building_block = linear_algebra_building_block(t)
    grads = get_autodiff_gradient(building_block, resolver)
    assert len(grads) == 0


@pytest.mark.parametrize("linear_algebra_building_block", eigen_building_blocks)
@pytest.mark.parametrize("seed", np.arange(10))
def test_autodiff_gradient_eigen(linear_algebra_building_block, seed):
    t = sympy.Symbol("time")
    np.random.seed(seed)
    eps = 1e-7
    tol = 1e-6
    resolver = ParamResolver({"time": np.random.rand(1)[0] * 100.0})
    building_block = linear_algebra_building_block(exponent=t)
    finitediff = get_finitediff_gradient(building_block, resolver, eps)
    grads = get_autodiff_gradient(building_block, resolver)
    g = {k.name: v for k, v in grads.items()}
    for k, expected in finitediff.items():
        actual = g[k]
        np.testing.assert_allclose(expected, actual, atol=tol, rtol=tol)


@pytest.mark.parametrize("linear_algebra_building_block", phased_building_blocks)
@pytest.mark.parametrize("seed", np.arange(10))
def test_autodiff_gradient_phased(linear_algebra_building_block, seed):
    t = sympy.Symbol("time")
    p = sympy.Symbol("phase")
    np.random.seed(seed)
    eps = 1e-7
    tol = 1e-6
    resolver = ParamResolver(
        {
            "time": np.random.rand(1)[0] * 100.0,
            "phase": np.random.rand(1)[0] * 100.0,
        }
    )
    building_block = linear_algebra_building_block(phase_exponent=p, exponent=t)
    finitediff = get_finitediff_gradient(building_block, resolver, eps)
    grads = get_autodiff_gradient(building_block, resolver)
    g = {k.name: v for k, v in grads.items()}
    for k, expected in finitediff.items():
        actual = g[k]
        np.testing.assert_allclose(expected, actual, atol=tol, rtol=tol)


@pytest.mark.parametrize("seed", np.arange(10))
def test_autodiff_gradient_fsim(seed):
    t = sympy.Symbol("theta")
    p = sympy.Symbol("phi")
    np.random.seed(seed)
    eps = 1e-7
    tol = 1e-6
    resolver = ParamResolver(
        {
            "theta": np.random.rand(1)[0] * 100.0,
            "phi": np.random.rand(1)[0] * 100.0,
        }
    )
    building_block = linear_algebra.rotate_on_xy_plane(theta=t, phi=p)
    finitediff = get_finitediff_gradient(building_block, resolver, eps)
    grads = get_autodiff_gradient(building_block, resolver)
    g = {k.name: v for k, v in grads.items()}
    for k, expected in finitediff.items():
        actual = g[k]
        np.testing.assert_allclose(expected, actual, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "Nparams, Nexponents",
    [(10, 10), (10, 0), (50, 50), (50, 0), (100, 100), (100, 0)],
)
@pytest.mark.parametrize("depth", [10, 50, 100])
@pytest.mark.parametrize("N", [10, 15, 20])
def test_parser(Nparams, Nexponents, depth, N):
    seed = 0
    acyclic_graph, discretes, resolver = build_random_acyclic_graph(
        Nparams, Nexponents, depth, N, seed
    )
    building_blocks, gradients, _ = parse(acyclic_graph, discretes, resolver)
    for n, op in enumerate(acyclic_graph.all_operations()):
        resolved = linear_algebra.unitary(linear_algebra.resolve_parameters(op, resolver))
        np.testing.assert_allclose(resolved, building_blocks[n].reshape(resolved.shape))
        if linear_algebra.is_variabled(op):
            grads = get_autodiff_gradient(op.building_block, resolver)
            for s, actual in gradients[n].items():
                np.testing.assert_allclose(
                    actual, grads[s].reshape((2, 2) * len(op.discretes))
                )
