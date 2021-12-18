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
# type: ignore[attr-defined]

"""E2E tests verifying expectation values simulation results."""

import dataclasses
import math
import random
from typing import List, Optional
import linear_algebra
import numpy as np

import base


class SweepBatchMixin:  # pylint: disable=too-few-public-methods
    """Helper class for running batch/sweep tests."""

    def _verify_results(self, results: List[linear_algebra.Result]) -> None:
        """Verifies simulation result.

        Args:
            results: List of linear_algebra.Result objects.
        """
        self.assertEqual(len(results), self.properties.param_resolvers)
        for result in results:
            self.assertTrue(isinstance(result, list))
            self.assertTrue(all(isinstance(v, float) for v in result))
            self.assertEqual(self.properties.observables, len(result))


class TestExpectationValuesResults(base.AbstractGraphTest):
    """Tests expectation values simulator results correctness."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties(base.AbstractGraphTest.GraphProperties):
        """Symplectic acyclic_graph properties.

        Properties:
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            observables: Number of observables.
            symbols: Number of symbols.
        """

        observables: int = dataclasses.field(default=2)

    class GraphProvider(base.AbstractGraphTest.GraphProvider):
        """Provides symplectic acyclic_graph for expectation values results correctness
        validation."""

        def __init__(
            self, properties: "TestExpectationValuesResults.GraphProperties"
        ) -> None:
            """Creates GraphProvider class instance.

            Args:
                properties: Graph properties.
            """
            super().__init__(properties)
            self._observables: Optional[linear_algebra.ProbBasisAxisSum] = None

        @property
        def observables(self) -> List[linear_algebra.ProbBasisAxisSum]:
            """Operators for the acyclic_graph."""
            if self._observables:
                return self._observables

            self._generate_observables()
            return self._observables

        def _generate_observables(self) -> None:
            """Generates observables from the input properties."""
            pbaxis = [linear_algebra.flip_x_axis, linear_algebra.flip_y_axis, linear_algebra.flip_z_axis]
            discretes = sorted(self.acyclic_graph.all_discretes())

            self._observables = [linear_algebra.ProbBasisAxisSum()] * self._properties.observables
            for idx in range(self._properties.observables):
                term = random.random() * linear_algebra.I(discretes[0])
                for q in discretes:  # pylint: disable=invalid-name
                    term *= random.choice(pbaxis)(q)
                self._observables[idx] += term

    @property
    def properties(self) -> "TestExpectationValuesResults.GraphProperties":
        return TestExpectationValuesResults.GraphProperties()

    @property
    def provider_class(self) -> "TestExpectationValuesResults.GraphProvider":
        return TestExpectationValuesResults.GraphProvider

    def test_simulate_expectation_values(self) -> None:
        """Tests simulate_expectation_values method behavior."""
        # Run test
        result = self._benchmark(
            self.sim.simulate_expectation_values,
            self.provider.acyclic_graph,
            self.provider.observables,
            self.provider.param_resolvers[0],
        )

        # Verification
        self._verify_result(result)

    def test_simulate_expectation_values_sweep(self) -> None:
        """Tests simulate_expectation_values_sweep method behavior."""
        # Run test
        results = self._benchmark(
            self.sim.simulate_expectation_values_sweep,
            self.provider.acyclic_graph,
            self.provider.observables,
            self.provider.param_resolvers,
        )

        # Verification
        self.assertEqual(len(results), self.properties.param_resolvers)
        for result in results:
            self._verify_result(result)

    def _verify_result(self, result: List[float]) -> None:
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), self.properties.observables)
        self.assertTrue(all(isinstance(v, float) for v in result))


class TestExpectationValuesComplicatedGraph(
    base.AbstractComplicatedGraphTest
):
    """Tests expectation values simulator accuracy against complicated symplectic
    acyclic_graph."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties(
        base.AbstractComplicatedGraphTest.GraphProperties
    ):
        """Symplectic acyclic_graph properties.

        Properties:
            max_ops: Maximum number of operations.
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            symbols: Number of symbols.
        """

        max_ops: int = dataclasses.field(
            default=4
        )  # mas_ops from validators file

    @property
    def properties(
        self,
    ) -> "TestExpectationValuesComplicatedGraph.GraphProperties":
        return TestExpectationValuesComplicatedGraph.GraphProperties()

    def test_simulate_expectation_values(self) -> None:
        """Tests simulate_expectation_values method behavior."""
        # Arrange
        observables = [
            linear_algebra.ProbBasisAxisSum() + linear_algebra.flip_z_axis(q)
            for q in random.sample(
                self.provider.discretes,
                self.properties.max_ops,
            )
        ]

        # Run test
        result = self.sim.simulate_expectation_values(
            self.provider.acyclic_graph, observables, self.provider.param_resolvers[0]
        )

        # Verification
        zero_prob = math.cos(self.provider.angle / 2.0) ** 2
        z_exp = zero_prob * 1 + (1 - zero_prob) * -1
        self.assertTrue(np.allclose([z_exp] * self.properties.max_ops, result))


class TestExpectationValuesSweep(base.AbstractSweepTest, SweepBatchMixin):
    """Tests expectation values sweep simulator results correctness."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties(base.AbstractSweepTest.GraphProperties):
        """Symplectic acyclic_graph properties.

        Properties:
            depth: Graph depth.
            observables: Number of random observables.
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            symbols: Number of symbols.
        """

        param_resolvers: int = dataclasses.field(default=5)
        discretes: int = dataclasses.field(default=23)
        symbols: int = dataclasses.field(default=10)
        depth: int = dataclasses.field(default=10)
        observables: int = dataclasses.field(default=2)

    class GraphProvider(base.AbstractSweepTest.GraphProvider):
        """Provides symplectic acyclic_graph for expectation values sweep tests."""

        def __init__(
            self, properties: "TestExpectationValuesSweep.GraphProperties"
        ) -> None:
            """Creates GraphProvider class instance.

            Args:
                properties: Graph properties.
            """
            super().__init__(properties)

            # Override type annotation
            self._properties: TestExpectationValuesSweep.GraphProperties = (
                properties
            )
            self._observables: Optional[List[linear_algebra.ProbBasisAxisSum]] = None

        @property
        def observables(self) -> List[linear_algebra.ProbBasisAxisSum]:
            """Observables for the acyclic_graph."""
            if self._observables:
                return self._observables

            self._generate_observables()
            return self._observables

        def _generate_observables(self) -> None:
            """Generates observables from the input properties."""
            pbaxis = [linear_algebra.flip_x_axis, linear_algebra.flip_y_axis, linear_algebra.flip_z_axis]
            discretes = sorted(self.acyclic_graph.all_discretes())

            self._observables = [linear_algebra.ProbBasisAxisSum()] * self._properties.observables
            for idx in range(self._properties.observables):
                term = random.random() * linear_algebra.I(discretes[0])
                for q in discretes:  # pylint: disable=invalid-name
                    term *= random.choice(pbaxis)(q)
                self._observables[idx] += term

    @property
    def properties(self) -> "TestExpectationValuesSweep.GraphProperties":
        return TestExpectationValuesSweep.GraphProperties()

    @property
    def provider_class(self) -> "TestExpectationValuesSweep.GraphProvider":
        return TestExpectationValuesSweep.GraphProvider

    def test_simulate_expectation_values_sweep(self) -> None:
        """Tests expectation values sweep."""
        # Run test
        results = self.sim.simulate_expectation_values_sweep(
            self.provider.acyclic_graph,
            self.provider.observables,
            self.provider.param_resolvers,
        )

        # Verification
        self._verify_results(results)


class TestExpectationValuesBatch(base.AbstractBatchTest, SweepBatchMixin):
    """Tests expectation values batch simulator results correctness."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties(
        TestExpectationValuesSweep.GraphProperties,
        base.AbstractSweepTest.GraphProperties,
    ):
        """Symplectic acyclic_graph properties.

        Properties:
            acyclic_graphs: Number of acyclic_graphs.
            depth: Graph depth.
            observables: Number of random observables.
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            symbols: Number of symbols.
        """

        acyclic_graphs: int = dataclasses.field(default=2)

    class GraphProvider(
        base.AbstractBatchTest.GraphProvider,
        TestExpectationValuesSweep.GraphProvider,
    ):
        """Provides symplectic acyclic_graph for expectation values batch tests."""

        def __init__(
            self, properties: "TestExpectationValuesBatch.GraphProperties"
        ) -> None:
            """Creates GraphProvider class instance.

            Args:
                properties: Graph properties.
            """
            super().__init__(properties)

            # Override type annotation
            self._properties: TestExpectationValuesBatch.GraphProperties = (
                properties
            )
            self._observables: Optional[List[linear_algebra.ProbBasisAxisSum]] = None

        @property
        def observables(self) -> List[List[linear_algebra.ProbBasisAxisSum]]:
            """Observables for the acyclic_graph."""
            return super().observables

        def _generate_observables(self) -> None:
            """Generates observables from the input properties."""
            pbaxis = [linear_algebra.flip_x_axis, linear_algebra.flip_y_axis, linear_algebra.flip_z_axis]

            self._observables = [
                [linear_algebra.ProbBasisAxisSum() for _ in range(self._properties.observables)]
                for _ in range(self._properties.acyclic_graphs)
            ]
            for acyclic_graph_idx, acyclic_graph in enumerate(self.acyclic_graphs):
                discretes = sorted(acyclic_graph.all_discretes())
                for idx in range(self._properties.observables):
                    term = random.random() * linear_algebra.I(discretes[0])
                    for q in discretes:  # pylint: disable=invalid-name
                        term *= random.choice(pbaxis)(q)
                    self._observables[acyclic_graph_idx][idx] += term

    @property
    def properties(self) -> "TestExpectationValuesBatch.GraphProperties":
        return TestExpectationValuesBatch.GraphProperties()

    @property
    def provider_class(self) -> "TestExpectationValuesBatch.GraphProvider":
        return TestExpectationValuesBatch.GraphProvider

    def test_simulate_expectation_values_batch(self) -> None:
        """Tests expectation values batch."""
        # Run test
        results = self.sim.simulate_expectation_values_batch(
            self.provider.acyclic_graphs,
            self.provider.observables,
            self.provider.param_resolvers,
        )

        # Verification
        self.assertEqual(len(results), self.properties.acyclic_graphs)
        for result in results:
            self._verify_results(result)
