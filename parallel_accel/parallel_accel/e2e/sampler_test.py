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

"""E2E tests verifying sampling simulation results."""

import dataclasses
import math
import random
from typing import Dict, List
import unittest
import linear_algebra
import numpy as np

import base


class SweepBatchMixin(unittest.TestCase):
    """Helper class for running batch/sweep tests.

    Properties:
        properties_context: Dictionary object that allows to override default
            GraphProperties dataclass values.
    """

    properties_context: Dict[str, int] = {}

    def tearDown(self) -> None:
        super().tearDown()
        self.properties_context = {}

    def _verify_results(
        self, results: List[linear_algebra.Result], parameters: List[linear_algebra.ParamResolver]
    ) -> None:
        """Verifies simulation result.

        Args:
            results: List of linear_algebra.Result objects.
            parameters: List of linear_algebra.ParamResolver submitted together with the
                input acyclic_graph.
        """
        self.assertEqual(len(results), self.properties.param_resolvers)
        for result, params in zip(results, parameters):
            self.assertTrue(isinstance(result, linear_algebra.Result))
            self.assertEqual(result.params, params)
            self.assertEqual(len(result.observations), self.properties.discretes)
            for value in result.observations.values():
                self.assertEqual(len(value), self.properties.repetitions)
                self.assertTrue(all(len(v) == 1 for v in value))


class TestSamplerResults(base.AbstractGraphTest):
    """Base class for testing sampling simulations."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties(base.AbstractGraphTest.GraphProperties):
        """Symplectic acyclic_graph properties.

        Properties:
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            repetitions: Number of input repetitions.
            symbols: Number of symbols.
        """

        repetitions: int = dataclasses.field(default=1000)

    @property
    def properties(self) -> "TestSamplerResults.GraphProperties":
        return TestSamplerResults.GraphProperties()

    def test_run(self) -> None:
        """Tests run function behavior."""
        # Run test
        result = self._benchmark(
            self.sim.run,
            self.provider.acyclic_graph,
            self.provider.param_resolvers[0],
            self.properties.repetitions,
        )

        # Verification
        self._verify_result(
            result,
            self.provider.param_resolvers[0],
            self.properties.repetitions,
        )

    def test_run_batch(self) -> None:
        """Tests run_batch method behavior."""
        # Arrange
        batches = 2
        acyclic_graph = [self.provider.acyclic_graph] * batches
        param_resolvers = [self.provider.param_resolvers] * batches
        repetitions = [
            random.randint(
                1,
                self.properties.repetitions,
            )
            for _ in range(batches)
        ]

        # Run test
        results = self._benchmark(
            self.sim.run_batch,
            acyclic_graph,
            param_resolvers,
            repetitions,
        )

        # Verification
        self.assertEqual(len(results), batches)
        for result, repetition in zip(results, repetitions):
            self.assertTrue(len(result), self.properties.param_resolvers)
            for r, p in zip(  # pylint: disable=invalid-name
                result, self.provider.param_resolvers
            ):  # pylint: disable=invalid-name
                self._verify_result(r, p, repetition)

    def test_run_sweep(self) -> None:
        """Tests run_sweep method behavior."""
        # Run test
        results = self._benchmark(
            self.sim.run_sweep,
            self.provider.acyclic_graph,
            self.provider.param_resolvers,
            self.properties.repetitions,
        )

        # Verification
        self.assertEqual(len(results), self.properties.param_resolvers)
        for result, provider in zip(results, self.provider.param_resolvers):
            self._verify_result(result, provider, self.properties.repetitions)

    def test_sample(self) -> None:
        """Tests sample method behavior."""
        # Run test
        result = self._benchmark(
            self.sim.sample,
            self.provider.acyclic_graph,
            params=self.provider.param_resolvers[0],
            repetitions=self.properties.repetitions,
        )

        # Verification
        self.assertEqual(len(result[:]), self.properties.repetitions)

    def test_sample_default_repetitions(self) -> None:
        """Tests sample method behavior: default number of
        repetitions."""
        # Run test
        result = self._benchmark(
            self.sim.sample,
            self.provider.acyclic_graph,
            params=self.provider.param_resolvers[0],
        )

        # Verification
        self.assertEqual(len(result[:]), 1)

    def test_sample_multiple_resolvers(self) -> None:
        """Tests sample method behavior: multiple resolvers."""
        # Run test
        self._benchmark(
            self.sim.sample,
            self.provider.acyclic_graph,
            params=self.provider.param_resolvers,
            repetitions=self.properties.repetitions,
        )

    def _verify_result(  # pylint: disable=arguments-differ
        self,
        result: linear_algebra.Result,
        param_resolver: linear_algebra.ParamResolver,
        repetitions: int,
    ) -> None:
        self.assertTrue(isinstance(result, linear_algebra.Result))
        self.assertEqual(result.params, param_resolver)
        self.assertEqual(len(result.observations), self.properties.discretes)
        for value in result.observations.values():
            self.assertEqual(len(value), repetitions)
            self.assertTrue(all(len(r) == 1 for r in value))


class TestSamplerComplicatedGraph(base.AbstractComplicatedGraphTest):
    """Tests sampling simulator accuracy against complicated symplectic acyclic_graph."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties(
        base.AbstractComplicatedGraphTest.GraphProperties
    ):
        """Symplectic acyclic_graph properties.

        Properties:
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            repetitions: Number of input repetitions.
            symbols: Number of symbols.
        """

        repetitions: int = dataclasses.field(default=10000)

    @property
    def properties(self) -> "TestSamplerComplicatedGraph.GraphProperties":
        return TestSamplerComplicatedGraph.GraphProperties()

    def test_run(self) -> None:
        """Tests run method behavior."""
        # Run test
        result = self.sim.run(
            self.provider.acyclic_graph,
            self.provider.param_resolvers[0],
            self.properties.repetitions,
        )

        # Verification
        # Check that repetitions are only all zeros or all ones.
        repetitions_matrix = [
            [m[0] for m in result.observations[str(q)]]
            for q in self.provider.discretes
        ]
        repetitions_ndarray = np.asarray(repetitions_matrix).T

        counters = [0, 0]
        for sample in repetitions_ndarray:
            is_one = sample[0] == 1
            counters[is_one] += 1
            self.assertTrue(all(x == int(is_one) for x in sample))

        self.assertEqual(sum(counters), self.properties.repetitions)

        # Test for correct sample probabilities.
        # Expected relative frequency of zero repetitions is cos(angle/2)**2
        # Standard deviation of the mean is then
        # sqrt(cos(angle/2)**2 - cos(angle/2)**4)) / sqrt(num_repetitions)
        # Finally, the relative frequency should be within a few standard deviations
        # of the mean.
        zero_rel_freq = counters[0] / self.properties.repetitions
        zero_prob = math.cos(self.provider.angle / 2.0) ** 2
        zero_prob_std = math.sqrt(
            math.cos(self.provider.angle / 2.0) ** 2
            - math.cos(self.provider.angle / 2.0) ** 4
        ) / math.sqrt(self.properties.repetitions)

        self.assertLess(abs(zero_prob - zero_rel_freq), 5 * zero_prob_std)


class TestSamplerSweep(base.AbstractSweepTest, SweepBatchMixin):
    """Tests sampling sweep results correctness."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties(base.AbstractSweepTest.GraphProperties):
        """Symplectic acyclic_graph properties.

        Properties:
            depth: Graph depth.
            repetitions: Number of repetitions.
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            symbols: Number of symbols.
        """

        param_resolvers: int = dataclasses.field(default=5)
        discretes: int = dataclasses.field(default=23)
        symbols: int = dataclasses.field(default=10)
        depth: int = dataclasses.field(default=10)
        repetitions: int = dataclasses.field(default=10)

    @property
    def properties(self) -> "TestSamplerSweep.GraphProperties":
        return TestSamplerSweep.GraphProperties(**self.properties_context)

    def test_sampler_sweep_small(self) -> None:
        """Tests sampler sweep: small acyclic_graph."""
        # Arrange
        self.properties_context = {"discretes": 8}
        self.provider = TestSamplerSweep.GraphProvider(self.properties)

        # Run test
        self._run_test()

    def test_sampler_sweep(self) -> None:
        """Tests sampler sweep."""
        # Arrange
        # Use default GraphProperties values

        # Run test
        self._run_test()

    def _run_test(self) -> None:
        """Runs test."""
        # Run test
        results = self.sim.run_sweep(
            self.provider.acyclic_graph,
            self.provider.param_resolvers,
            self.properties.repetitions,
        )

        # Verification
        self._verify_results(results, self.provider.param_resolvers)


class TestSamplerBatch(base.AbstractBatchTest, SweepBatchMixin):
    """Tests sampling batch results correctness."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties(
        TestSamplerSweep.GraphProperties,
        base.AbstractBatchTest.GraphProperties,
    ):
        """Symplectic acyclic_graph properties.

        Properties:
            acyclic_graphs: Number of acyclic_graphs.
            depth: Graph depth.
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            repetitions: Number of repetitions.
            symbols: Number of symbols.
        """

        acyclic_graphs: int = dataclasses.field(default=2)

    @property
    def properties(self) -> "TestSamplerBatch.GraphProperties":
        return TestSamplerBatch.GraphProperties(**self.properties_context)

    def test_sampler_batch_small(self) -> None:
        """Tests sampler batch: small acyclic_graph."""
        # Arrange
        self.properties_context = {"discretes": 8}
        self.provider = TestSamplerBatch.GraphProvider(self.properties)

        # Run test
        self._run_test()

    def test_sampler_batch(self) -> None:
        """Tests sampler batch."""
        # Arrange
        # Use default GraphProperties values

        # Run test
        self._run_test()

    def _run_test(self) -> None:
        """Runs test."""
        # Run test
        results = self.sim.run_batch(
            self.provider.acyclic_graphs,
            self.provider.param_resolvers,
            self.properties.repetitions,
        )

        # Verification
        self.assertTrue(len(results), self.properties.acyclic_graphs)
        for result, params in zip(results, self.provider.param_resolvers):
            self._verify_results(result, params)
