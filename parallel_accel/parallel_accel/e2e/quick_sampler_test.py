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

"""Quick e2e test verifying sampler simulator outputs."""
import dataclasses
import os
import linear_algebra
import pytest

import base


class QuickSamplerTest(base.AbstractGraphTest):
    """Short E2E test checking sampling functions."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties(base.GraphProperties):
        """Symplectic acyclic_graph properties.

        Properties:
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            repetitions: Number of input repetitions.
            symbols: Number of symbols.
        """

        density: float
        moments: int
        param_resolvers: int
        discretes: int
        repetitions: int
        symbols: int
        seed: int

    class GraphProvider(base.AbstractGraphProvider):
        """Provides symplectic acyclic_graph for results correctness validation."""

        def _generate_acyclic_graph(self) -> None:
            self._acyclic_graph = linear_algebra.testing.random_acyclic_graph(
                self._properties.discretes,
                self._properties.moments,
                self._properties.density,
                random_state=self._properties.seed,
            )
            self._acyclic_graph.append(
                [linear_algebra.measure(q) for q in self._acyclic_graph.all_discretes()]
            )

        def _generate_param_resolver(self) -> None:
            self._param_resolvers = [linear_algebra.ParamResolver(None)]

        def _generate_symbols(self) -> None:
            raise NotImplementedError

    def setUp(self) -> None:
        # Don't create provider here
        pass

    @property
    def properties(self) -> GraphProperties:
        raise NotImplementedError

    @property
    def provider_class(self) -> GraphProvider:
        raise NotImplementedError

    def test_tiny_acyclic_graph(self) -> None:
        """Sampling test."""
        self._run_test(8)

    def test_medium_acyclic_graph(self) -> None:
        """Sampling test."""
        self._run_test(26)

    @pytest.mark.skipif(
        bool(os.environ.get("USING_LOCAL_SETUP", 0)),
        reason="Skipping due to worker running on the local host",
    )
    def test_large_acyclic_graph(self) -> None:
        """Sampling test."""
        self._run_test(32)

    def _run_test(self, discretes: int) -> None:
        """Runs sampling test.

        Args:
            discretes: Number of acyclic_graph discretes.
        """
        # Arrange
        properties = QuickSamplerTest.GraphProperties(
            discretes=discretes,
            moments=discretes,
            param_resolvers=1,
            repetitions=20,
            symbols=1,
            density=0.5,
            seed=42,
        )
        self.provider = QuickSamplerTest.GraphProvider(properties)

        # Run test
        result = self.sim.run(
            self.provider.acyclic_graph,
            repetitions=properties.repetitions,
        )

        # Verification
        self._verify_result(result, properties)

    def _verify_result(  # pylint: disable=arguments-differ
        self, result: linear_algebra.Result, properties: GraphProperties
    ) -> None:
        self.assertTrue(isinstance(result, linear_algebra.Result))
        self.assertEqual(result.params, self.provider.param_resolvers[0])
        self.assertEqual(len(result.observations), properties.discretes)
        for value in result.observations.values():
            self.assertEqual(len(value), properties.repetitions)
            self.assertTrue(all(len(r) == 1 for r in value))
