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
import numpy as np


def get_random_prob_basis_axis_sum(discretes,
                         num_prob_basis_axis_strings=None,
                         num_prob_basis_axis_factors=None,
                         seed=0):
  """Test fixture.

  Args:
    discretes: A list of `linear_algebra.GridSpace`s on which to build the pauli sum.
    num_prob_basis_axis_strings: The number of prob_basis_axis_strings to sum to generate the
      returned pauli sum.
    num_prob_basis_axis_factors: The number of factors (X, Y, Z) to compose for each
      pauli string.
    seed: A seed for randomization.

  Returns:
    prob_basis_axis_sum: A `linear_algebra.ProbBasisAxisSum` which is a linear combination of random pauli
    strings on `discretes`.
  """
  np.random.seed(seed)
  pbaxis = [linear_algebra.flip_x_axis, linear_algebra.flip_y_axis, linear_algebra.flip_z_axis]
  coeff_max = 1.5
  coeff_min = -1.5

  num_discretes = len(discretes)
  if num_prob_basis_axis_strings is None:
    num_prob_basis_axis_strings = num_discretes
  if num_prob_basis_axis_factors is None:
    num_prob_basis_axis_factors = num_discretes

  prob_basis_axis_sum = linear_algebra.ProbBasisAxisSum()
  for _ in range(num_prob_basis_axis_strings):
    pbaxis = np.random.choice(pbaxis, num_prob_basis_axis_factors)
    sub_discretes = np.random.choice(discretes, num_prob_basis_axis_factors, replace=False)
    coeff = np.random.rand(1) * (coeff_max - coeff_min) + coeff_min
    prob_basis_axis_sum += linear_algebra.ProbBasisAxisString(coeff,
                                  *[p(q) for p, q in zip(pbaxis, sub_discretes)])
  return prob_basis_axis_sum
