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

"""Quick e2e test checking simulator functions."""
import dataclasses
import linear_algebra

import base


@dataclasses.dataclass(frozen=True)
class GraphProperties(base.GraphProperties):
    """See base class documentation."""

    param_resolvers: int = dataclasses.field(default=1)
    discretes: int = dataclasses.field(default=21)
    symbols: int = dataclasses.field(default=1)


class GraphProvider(base.AbstractGraphProvider):
    """Provides symplectic acyclic_graph for simulator tests."""

    def _generate_acyclic_graph(self) -> None:
        positions = []
        measures = []

        for q in self.discretes:  # pylint: disable=invalid-name
            positions.append(linear_algebra.flip_x_axis(q))
            measures.append(linear_algebra.measure(q))

        self._acyclic_graph = linear_algebra.Graph(positions + measures)

    def _generate_discretes(self) -> None:
        self._discretes = linear_algebra.LinearSpace.range(self._properties.discretes)

    def _generate_param_resolver(self) -> None:
        raise NotImplementedError

    def _generate_symbols(self) -> None:
        raise NotImplementedError


class QuickSamplerTest(base.AbstractTest):
    """Short E2E test checking simulator functions."""

    @property
    def properties(self) -> GraphProperties:
        return GraphProperties()

    @property
    def provider_class(self) -> GraphProvider:
        return GraphProvider

    def test_samples(self) -> None:
        """Sampling test."""
        # Run test
        actual_result = self.sim.run(self.provider.acyclic_graph)
        expected_result = linear_algebra.Simulator().run(self.provider.acyclic_graph)

        # Verification
        self.assertEqual(actual_result, expected_result)

    def test_expectation(self) -> None:
        """Expectation values test."""
        # Arrange
        observables = [
            linear_algebra.flip_x_axis(self.provider.discretes[0]) + linear_algebra.flip_y_axis(self.provider.discretes[1])
        ]
        fsv = (
            linear_algebra.Simulator().simulate(self.provider.acyclic_graph).final_state_vector
        )

        # Run test
        expected_result = [
            observables[0].expectation_from_state_vector(
                fsv, {discrete: i for i, discrete in enumerate(self.provider.discretes)}
            )
        ]
        actual_result = self.sim.simulate_expectation_values(
            self.provider.acyclic_graph, observables
        )

        # Verification
        self.assertEqual(actual_result, expected_result)
