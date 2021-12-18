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
"""Context Validator Tests."""
import random
import unittest

import linear_algebra
import sympy

import context_validator
from context_validator import ValidationError
from parallel_accel.shared import schemas


def get_acyclic_graph(num_discretes, depth, num_params):
    """Returns a valid acyclic_graph for the given number of discretes."""
    if depth * num_discretes < num_params:
        raise Exception("Can only have as many parameters as building_blocks.")
    acyclic_graph = linear_algebra.Graph()
    discretes = linear_algebra.GridSpace.rect(1, num_discretes)
    if num_params < 1:
        for d in range(depth):
            for q in discretes:
                acyclic_graph += linear_algebra.flip_x_axis(q)
    else:
        params = [sympy.Symbol(f"s_{n}") for n in range(num_params)]
        params *= (depth * num_discretes + num_params) // num_params
        for d in range(depth):
            for q, s in zip(
                discretes, params[d * num_discretes : (d + 1) * num_discretes]
            ):
                acyclic_graph += linear_algebra.flip_x_axis(q) ** s
    return acyclic_graph


def get_operators(num_discretes, num_ops, num_terms):
    """Returns a valid list of operators for the given number of discretes."""
    if 2 ** (num_discretes + 1) <= num_terms:
        raise Exception("No more than 2**num_discretes terms are possible.")
    operators = []
    discretes = linear_algebra.GridSpace.rect(1, num_discretes)
    for _ in range(num_ops):
        this_op = linear_algebra.ProbBasisAxisSum()
        for term_num in range(num_terms):
            term = random.random() * linear_algebra.I(discretes[0])
            temp_term_num = int(term_num)
            if term_num <= 2 ** num_discretes:
                for i in range(num_discretes):
                    if temp_term_num % 2:
                        term *= linear_algebra.flip_x_axis(discretes[i])
                    temp_term_num //= 2
            else:
                temp_term_num //= 2
                for i in range(num_discretes):
                    if temp_term_num % 2:
                        term *= linear_algebra.flip_y_axis(discretes[i])
                    temp_term_num //= 2
            this_op += term
        operators.append(this_op)
    return operators


def get_param_resolver(num_params):
    """Returns a valid param_resolver for the given number of discretes."""
    params = {f"s_{n}": n for n in range(num_params)}
    return linear_algebra.ParamResolver(params)


class ValidatorTestCase(unittest.TestCase):
    def validate_pass(self):
        self.validator.validate()

    def validate_fail(self):
        with self.assertRaises(ValidationError):
            self.validator.validate()


class BaseValidatorTest(ValidatorTestCase):
    @classmethod
    def setUpClass(cls):
        if cls is BaseValidatorTest:
            raise unittest.SkipTest("Skip Base Tests")
        super(BaseValidatorTest, cls).setUpClass()

    def test_valid_context(self):
        for discretes in range(
            self.validator.min_num_discretes, self.validator.max_num_discretes + 1
        ):
            with self.subTest(discretes=discretes):
                self.context.acyclic_graph = get_acyclic_graph(discretes, 10, 10)
                self.validate_pass()

    def test_max_num_discretes(self):
        num_discretes = self.validator.max_num_discretes + 1
        self.context.acyclic_graph = get_acyclic_graph(num_discretes, 10, 10)
        self.validate_fail()

    def test_max_depth(self):
        self.context.acyclic_graph = get_acyclic_graph(
            10, self.validator.acyclic_graph_depth_limit + 1, 10
        )
        self.validate_fail()

    def test_terminal_observation(self):
        self.context.acyclic_graph.append(
            [linear_algebra.measure(q) for q in self.context.acyclic_graph.all_discretes()]
        )
        self.validate_pass()
        self.context.acyclic_graph.append(
            [linear_algebra.flip_x_axis(q) for q in self.context.acyclic_graph.all_discretes()]
        )
        self.validate_fail()

    def test_num_params(self):
        self.context.acyclic_graph = get_acyclic_graph(
            10, 4000, self.validator.param_resolver_limit + 1
        )
        self.context.param_resolver = get_param_resolver(
            self.validator.param_resolver_limit + 1
        )
        self.validate_fail()

    def test_non_matching_params(self):
        self.context.acyclic_graph = get_acyclic_graph(10, 10, 0)
        self.validate_fail()

    def test_bad_building_block(self):
        bad_op = linear_algebra.PhasedXZGate(
            x_exponent=1, z_exponent=1, axis_phase_exponent=1
        )
        self.context.acyclic_graph.append(
            [bad_op(q) for q in self.context.acyclic_graph.all_discretes()]
        )
        self.validate_fail()

    def test_bad_building_block_symbol_exp(self):
        bad_exp = sympy.Symbol("a") * sympy.Symbol("b")
        bad_op = linear_algebra.flip_x_axis ** bad_exp
        self.context.acyclic_graph.append(
            [bad_op(q) for q in self.context.acyclic_graph.all_discretes()]
        )
        self.validate_fail()

    def test_bad_building_block_num_params(self):
        bad_op = linear_algebra.rotate_on_xy_plane(theta=sympy.Symbol("a"), phi=sympy.Symbol("b"))
        discretes = list(self.context.acyclic_graph.all_discretes())
        self.context.acyclic_graph.append([bad_op(discretes[0], discretes[1])])
        self.validate_fail()


class SampleValidatorTest(BaseValidatorTest):
    context_schema = schemas.SampleJobContext

    def setUp(self) -> None:
        acyclic_graph = get_acyclic_graph(10, 10, 10)
        param_resolver = get_param_resolver(10)
        self.context = self.context_schema(
            acyclic_graph=acyclic_graph, param_resolver=param_resolver
        )
        self.validator = context_validator.SampleValidator(self.context)

    def test_bad_num_reps(self):
        self.context.repetitions = self.validator.repetition_limit + 1
        self.validate_fail()


class ExpectationValidatorTest(BaseValidatorTest):
    context_schema = schemas.ExpectationJobContext

    def setUp(self) -> None:
        acyclic_graph = get_acyclic_graph(21, 10, 10)
        param_resolver = get_param_resolver(10)
        operators = get_operators(21, 4, 10)
        self.context = self.context_schema(
            acyclic_graph=acyclic_graph, param_resolver=param_resolver, operators=operators
        )
        self.validator = context_validator.ExpectationValidator(self.context)

    def test_operator_num(self):
        self.context.operators = get_operators(21, 10, 1)
        self.validate_fail()

    def test_operator_terms(self):
        self.context.operators = get_operators(21, 1, 22)
        self.validate_fail()


class SampleSweepValidatorTest(SampleValidatorTest):
    context_schema = schemas.SampleSweepJobContext

    def setUp(self) -> None:
        acyclic_graph = get_acyclic_graph(10, 10, 10)
        params = [get_param_resolver(10)]
        self.context = self.context_schema(acyclic_graph=acyclic_graph, params=params)
        self.validator = context_validator.SampleSweepValidator(self.context)

    def test_num_params(self):
        self.context.acyclic_graph = get_acyclic_graph(
            10, 4000, self.validator.param_resolver_limit + 1
        )
        self.context.params = [
            get_param_resolver(self.validator.param_resolver_limit),
            get_param_resolver(self.validator.param_resolver_limit + 1),
        ]
        self.validate_fail()

    def test_non_matching_params(self):
        self.context.acyclic_graph = get_acyclic_graph(10, 10, 0)
        self.validate_fail()

    def test_num_sweepables(self):
        self.context.params = [get_param_resolver(10) for _ in range(11)]
        self.validate_fail()


class ExpectationSweepValidatorTest(ExpectationValidatorTest):
    context_schema = schemas.ExpectationSweepJobContext

    def setUp(self) -> None:
        acyclic_graph = get_acyclic_graph(21, 10, 10)
        params = [get_param_resolver(10)]
        operators = get_operators(21, 4, 10)
        self.context = self.context_schema(
            acyclic_graph=acyclic_graph, params=params, operators=operators
        )
        self.validator = context_validator.ExpectationSweepValidator(
            self.context
        )

    def test_num_params(self):
        self.context.acyclic_graph = get_acyclic_graph(
            10, 4000, self.validator.param_resolver_limit + 1
        )
        self.context.params = [
            get_param_resolver(self.validator.param_resolver_limit),
            get_param_resolver(self.validator.param_resolver_limit + 1),
        ]
        self.validate_fail()

    def test_non_matching_params(self):
        self.context.acyclic_graph = get_acyclic_graph(10, 10, 0)
        self.validate_fail()

    def test_num_sweepables(self):
        self.context.params = [get_param_resolver(10) for _ in range(11)]
        self.validate_fail()


class SweepBatchValidatorTest(ValidatorTestCase):
    context_schema = schemas.SampleBatchJobContext

    def setUp(self) -> None:
        batch_size = 10
        acyclic_graphs = [get_acyclic_graph(10, 10, 10) for _ in range(batch_size)]
        params = [
            [get_param_resolver(10) for _ in range(batch_size)]
            for _ in range(batch_size)
        ]
        repetitions = [1] * batch_size
        self.context = self.context_schema(
            acyclic_graphs=acyclic_graphs, params=params, repetitions=repetitions
        )
        self.validator = context_validator.SampleBatchValidator(self.context)

    def test_valid(self):
        self.validate_pass()

    def test_num_batches(self):
        batch_size = 11
        self.context.acyclic_graphs = [
            get_acyclic_graph(21, 10, 10) for _ in range(batch_size)
        ]
        self.context.params = [
            [get_param_resolver(10) for _ in range(batch_size)]
            for _ in range(batch_size)
        ]
        self.validate_fail()


class ExpectationBatchValidatorTest(ValidatorTestCase):
    context_schema = schemas.ExpectationBatchJobContext

    def setUp(self) -> None:
        batch_size = 10
        acyclic_graphs = [get_acyclic_graph(21, 10, 10) for _ in range(batch_size)]
        params = [
            [get_param_resolver(10) for _ in range(batch_size)]
            for _ in range(batch_size)
        ]
        operators = [get_operators(21, 4, 4) for _ in range(batch_size)]
        self.context = self.context_schema(
            acyclic_graphs=acyclic_graphs, params=params, operators=operators
        )
        self.validator = context_validator.ExpectationBatchValidator(
            self.context
        )

    def test_valid(self):
        self.validate_pass()

    def test_num_batches(self):
        batch_size = 11
        self.context.acyclic_graphs = [
            get_acyclic_graph(21, 10, 10) for _ in range(batch_size)
        ]
        self.context.params = [
            [get_param_resolver(10) for _ in range(batch_size)]
            for _ in range(batch_size)
        ]
        self.context.operators = [
            get_operators(21, 4, 4) for _ in range(batch_size)
        ]
        self.validate_fail()
