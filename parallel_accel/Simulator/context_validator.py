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
"""
Context Validator objects to ensure that all job contexts adhere to service
limits.
"""

from abc import ABC, abstractmethod
from functools import cached_property
import time
from typing import List, Set, Union, Any

import linear_algebra
import sympy

from parallel_accel.shared import logger
from parallel_accel.shared.schemas import (
    ExpectationBatchJobContext,
    ExpectationJobContext,
    ExpectationSweepJobContext,
    SampleBatchJobContext,
    SampleJobContext,
    SampleSweepJobContext,
    SampleSweepJobResult,
)

JobContextType = Union[
    ExpectationBatchJobContext,
    ExpectationJobContext,
    ExpectationSweepJobContext,
    SampleBatchJobContext,
    SampleJobContext,
    SampleSweepJobContext,
    SampleSweepJobResult,
]

log = logger.get_logger(__name__)


class ValidationError(Exception):
    """Error to send back the client about invalid Job request contexts."""


# ============================================================================ #
# Globals.
# ============================================================================ #

SIMPLE_BUILDING_BLOCKS = {
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
}
BUILDING_BLOCKS_WITH_ANGLES = {linear_algebra.x_axis_two_angles, linear_algebra.imaginary_swap_two_angles}

# ============================================================================ #
# Helper Functions.
# ============================================================================ #


def validate_get_symbols(val: sympy.Basic) -> Set[sympy.Symbol]:
    """Checks that `val` is an allowed sympy-expression.

    ***Adapted from parser***

    `val` can be either a sympy.Symbol, a numeric constant,
    an expression of the form symbol * constant,
    constant * symbol or symbol / constant.

    Args:
        val: sympy.Basic

    Returns:
        Set of symbols contained in the valid expression

    Raises:
        ValidationError: If `val` is not an allowed sympy-expression.
    """
    msg = (
        "Parameters can only be sympy.Symbols, not expressions. Got a "
        f"parameter = {val} of type(parameter) = {type(val)}"
    )
    val = sympy.sympify(val)
    if not len(val.free_symbols) < 2:
        raise ValidationError(
            "Only expressions with fewer than 2 free symbols are allowed."
        )
    if isinstance(val, sympy.Basic):
        if val.is_number:
            return set()
        if val.is_symbol:
            return {val}
    args = val.args
    if val.func is sympy.Mul:
        if args[0].is_symbol and args[1].is_number:
            return {args[0]}
        if args[1].is_symbol and args[0].is_number:
            return {args[1]}
        if args[0].is_number and args[1].is_number:
            return set()
        if args[0].is_symbol and args[1].func is sympy.Pow:
            if args[1].args[0].is_number and args[1].args[1].is_number:
                return set()
    raise ValidationError(msg)


def get_symbol_expressions(linear_algebra_op: linear_algebra.Operation) -> Set[sympy.Basic]:
    """
    ***Adapted from parser***

    Obtain a set of parameter-symbols of `linear_algebra_op`.
    As opposed to `linear_algebra.parameter_symbols`, this function
    also returns expressions of symbols.

    Args:
      linear_algebra_op: One of the supported linear_algebra building_blocks.
    Returns:
      set[sympy.Basic]: Set of sympy symbols or expressions.
    """
    symbols = set()
    if linear_algebra.is_observation(linear_algebra_op) or isinstance(linear_algebra_op, linear_algebra.EmptyBuildingBlock):
        return symbols  # Observations and identity are not variabled.
    if isinstance(linear_algebra_op, tuple(SIMPLE_BUILDING_BLOCKS)):
        # pylint: disable=protected-access
        if isinstance(linear_algebra_op._exponent, sympy.Basic):
            symbols |= {linear_algebra_op._exponent}
        # pylint: enable=protected-access
    elif isinstance(linear_algebra_op, tuple(BUILDING_BLOCKS_WITH_ANGLES)):
        # pylint: disable=protected-access
        for param in [linear_algebra_op._exponent, linear_algebra_op._phase_exponent]:
            # pylint: enable=protected-access
            if isinstance(param, sympy.Basic):
                symbols |= {param}
    elif isinstance(linear_algebra_op, linear_algebra.rotate_on_xy_plane):
        for param in [linear_algebra_op.theta, linear_algebra_op.phi]:
            if isinstance(param, sympy.Basic):
                symbols |= {param}
    return symbols


# ============================================================================ #
# Validator Classes.
# ============================================================================ #


class ContextValidator:
    """
    Class defining base validator methods and structure.

    Validators should be subclassed from this and define the following:
      * Limits - properties which are referenced by validators to determine
        whether a context is valid.
      * Helper properties - Additional properties derived from the context that
        are useful for performing validation, like cached calculations.
      * Validators - methods beginning with _validate that take no arguments and
        return None. These methods should access the context, context derived
        helper properties and limits via self and raise a ValidationError if
        the context does not meet the requirements of the validator.
    """

    def __init__(self, context: Any) -> None:
        self._context = context
        self.validator_fns = [
            getattr(self, x)
            for x in dir(self)
            if x.startswith("_validate") and callable(getattr(self, x))
        ]

    def validate(self) -> None:
        for fn in self.validator_fns:
            fn()

    @property
    def context(self) -> Any:
        return self._context


class BaseValidator(ContextValidator, ABC):
    """Base Validator shared by Sample/Expectation Validators."""

    # Limits
    acyclic_graph_depth_limit = 4000
    param_resolver_limit = 4000
    supported_parameterizable_building_block_types = (
        SIMPLE_BUILDING_BLOCKS | BUILDING_BLOCKS_WITH_ANGLES | {linear_algebra.EmptyBuildingBlock, linear_algebra.rotate_on_xy_plane}
    )

    @property
    @abstractmethod
    def min_num_discretes(self) -> int:
        """Defines the minimum number of discretes limit."""

    @property
    @abstractmethod
    def max_num_discretes(self) -> int:
        """Defines the maximum number of discretes limit."""

    ## Helper properties
    @cached_property
    def acyclic_graph_building_blocks(self) -> List[linear_algebra.Gate]:
        building_blocks = [op.building_block for op in self._context.acyclic_graph.all_operations()]
        return building_blocks

    @property
    def param_resolvers(self) -> List[linear_algebra.ParamResolver]:
        return [self._context.param_resolver]

    ## Validators
    def _validate_acyclic_graph_basic(self) -> None:
        num_discretes = len(self.context.acyclic_graph.all_discretes())
        if not self.min_num_discretes <= num_discretes <= self.max_num_discretes:
            raise ValidationError(
                "Graph is outside the allowed num_discrete range. Received "
                f"{num_discretes}, allowed range is {self.min_num_discretes} - "
                f"{self.max_num_discretes}."
            )
        if len(self.context.acyclic_graph) > self.acyclic_graph_depth_limit:
            raise ValidationError(
                "Graph exceeds the maximum allowed depth. Received "
                f"{len(self.context.acyclic_graph)}, max allowed depth is "
                f"{self.acyclic_graph_depth_limit}."
            )
        if not self.context.acyclic_graph.are_all_observations_terminal():
            raise ValidationError("All observations must be terminal.")

    def _validate_ops(self) -> None:
        for building_block in self.acyclic_graph_building_blocks:
            if not isinstance(
                building_block, tuple(self.supported_parameterizable_building_block_types)
            ) and not linear_algebra.is_observation(building_block):
                raise ValidationError(
                    f"Given building_block type not supported, got {type(building_block)}. Allowed "
                    "building_block types are observations or one of "
                    f"{self.supported_parameterizable_building_block_types}."
                )

    def _validate_params_resolvers(self) -> None:
        for i, param_resolver in enumerate(self.param_resolvers):
            if not isinstance(param_resolver, linear_algebra.ParamResolver):
                raise ValidationError(
                    "Invalid param_resolver object type, expected "
                    f"linear_algebra.ParamResolver, got {type(param_resolver)}."
                )
            param_vals = set(param_resolver.param_dict.keys())
            num_params = len(param_vals)
            num_params_limit = self.param_resolver_limit
            if num_params > num_params_limit:
                raise ValidationError(
                    f"Too many parameters in resolver {i}. Number of "
                    f"parameters is {num_params}, limit is {num_params_limit}."
                )

    def _validate_symbols(self) -> None:
        all_symbol_exprs = set()
        for building_block in self.acyclic_graph_building_blocks:
            symbol_exprs = get_symbol_expressions(building_block)
            if len(symbol_exprs) > 1:
                raise ValidationError(
                    f"Only one argument of the op can be variabled, got "
                    f"{len(symbol_exprs)}"
                )
            all_symbol_exprs |= symbol_exprs

        symbols = set()
        for symbol_expr in all_symbol_exprs:
            symbols |= validate_get_symbols(symbol_expr)

        symbol_strs = {str(sym) for sym in symbols}

        for i, param_resolver in enumerate(self.param_resolvers):
            resolver_params = set(param_resolver.param_dict.keys())
            if symbol_strs != resolver_params:
                raise ValidationError(
                    "The set of symbols in the acyclic_graph does not equal "
                    f"the set of symbols in param_resolver at index {i}."
                )


class SampleValidator(BaseValidator):
    """Validates SampleJobContext Objects."""

    repetition_limit = 1e6

    @property
    def min_num_discretes(self) -> int:
        return 1

    @property
    def max_num_discretes(self) -> int:
        return 32

    def _validate_reps(self) -> None:
        if not 1 <= self.context.repetitions <= self.repetition_limit:
            raise ValidationError(
                f"Requested {self.context.repetitions} samples, maximum "
                f"allowed is {self.repetition_limit}."
            )


class ExpectationValidator(BaseValidator):
    """Validates ExpectationJobContext Objects."""

    operators_len_limit = 4

    @property
    def min_num_discretes(self) -> int:
        return 21

    @property
    def max_num_discretes(self) -> int:
        return 32

    @cached_property
    def operators_terms_limit(self) -> int:
        return len(self._context.acyclic_graph.all_discretes())

    def _validate_operators(self) -> None:
        operators = self.context.operators
        if isinstance(operators, linear_algebra.ProbBasisAxisSum):
            operators = [operators]
        operators_len = len(operators)
        if operators_len > self.operators_len_limit:
            raise ValidationError(
                "Too many operators. Number of operators given is "
                f"{operators_len}, limit is {self.operators_len_limit}.\n"
                "Consider using multiple expectation calls, each with fewer "
                "operators.",
            )
        for op_num, op in enumerate(operators):
            operators_terms = len(op)
            if operators_terms > self.operators_terms_limit:
                raise ValidationError(
                    f"Operator {op_num} has too many terms. Number of terms is "
                    f"{operators_terms}, limit is {self.operators_terms_limit}."
                )


class SweepMixin:
    """Mixin to validate Sweep context types."""

    num_sweepables_limit = 10

    @property
    def param_resolvers(self) -> List[linear_algebra.ParamResolver]:
        return list(self._context.params)

    def _validate_params(self) -> None:
        num_resolvers = len(self.context.params)
        if num_resolvers > self.num_sweepables_limit:
            raise ValidationError(
                "Maximum number of resolvers exceeded in sweepable. Max is "
                f"{self.num_sweepables_limit}, got {num_resolvers}."
            )


class SampleSweepValidator(SweepMixin, SampleValidator):
    """Validates SampleSweepJobContext Objects."""


class ExpectationSweepValidator(SweepMixin, ExpectationValidator):
    """Validates ExpectationSweepJobContext Objects."""


class SampleBatchValidator(ContextValidator):
    """Validates SampleBatchJobContext Objects."""

    num_batches_limit = 10

    def _validate_batches_limit(self) -> None:
        num_batches = len(self.context.acyclic_graphs)
        if num_batches > self.num_batches_limit:
            raise ValidationError(
                "Maximum number of batches exceeded. Max is "
                f"{self.num_batches_limit}, got {num_batches}."
            )

    def _validate_sub_contexts(self) -> None:
        if isinstance(self.context.repetitions, list):
            repetitions_list = self.context.repetitions
        else:
            repetitions_list = [self.context.repetitions] * len(
                self.context.acyclic_graphs
            )
        for acyclic_graph, param, repetitions in zip(
            self.context.acyclic_graphs, self.context.params, repetitions_list
        ):
            sub_context = SampleSweepJobContext(acyclic_graph, param, repetitions)
            SampleSweepValidator(sub_context).validate()


class ExpectationBatchValidator(SampleBatchValidator):
    """Validates ExpectationBatchJobContext Objects."""

    def _validate_sub_contexts(self) -> None:
        for acyclic_graph, param, operators in zip(
            self.context.acyclic_graphs, self.context.params, self.context.operators
        ):
            sub_context = ExpectationSweepJobContext(acyclic_graph, param, operators)
            ExpectationSweepValidator(sub_context).validate()


def validate(context: JobContextType) -> None:
    context_map = {
        SampleJobContext: SampleValidator,
        SampleSweepJobContext: SampleSweepValidator,
        SampleBatchJobContext: SampleBatchValidator,
        ExpectationJobContext: ExpectationValidator,
        ExpectationSweepJobContext: ExpectationSweepValidator,
        ExpectationBatchJobContext: ExpectationBatchValidator,
    }
    t0 = time.time()
    log.info("Begin context validation", context_type=type(context).__name__)
    validator = context_map[type(context)](context)
    validator.validate()
    log.info("Context validation finished", time=time.time() - t0)
