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
import sympy
import numpy as np
from asic_la.testutils import full_matrix
import pytest

sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])


@pytest.mark.parametrize(
    "i1, i2, N, m1, m2",
    [(0, 4, 5, sx, sx), (1, 2, 5, sx, sz), (2, 4, 5, sx, sy)],
)
def test_full_matrix(i1, i2, N, m1, m2):
    assert i2 > i1
    exp = np.eye(2) if i1 > 0 else m1
    for n in range(1, N):
        if n == i1:
            exp = np.kron(exp, m1)
        elif n == i2:
            exp = np.kron(exp, m2)
        else:
            exp = np.kron(exp, np.eye(2))
    actual = full_matrix(np.kron(m1, m2), (i1, i2), N)
    np.testing.assert_allclose(actual, exp)
