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
# Lint as: python3
"""Tests for asic_la.asic_simulator.
"""
import unittest
import sympy
import linear_algebra
import linear_algebra.experiments as ce
import numpy as np

from asic_la import asic_simulator
import asic_la.asic_simulator_helpers as helpers
from asic_la.sharded_probability_function import complex_workaround as cw
from asic_la.testutils import build_random_acyclic_graph, generate_pbaxisum

NUM_DISCRETEDS = 21  # the max total number of discretes used for testing


def to_array(arr):
    return np.array(arr.real) + 1j * np.array(arr.imag)


class AsicSimulatorTest(unittest.TestCase):
    def test_quick_sim_fallback(self):
        discretes = linear_algebra.LinearSpace.range(6)
        acyclic_graph = linear_algebra.Graph(
            [linear_algebra.flip_x_axis(discretes[1]), linear_algebra.flip_x_axis(discretes[5]), linear_algebra.flip_x_axis(discretes[3])]
        )
        acyclic_graph += [linear_algebra.measure(discretes[i]) for i in range(6)]
        normal_sim = linear_algebra.Simulator()
        normal_result = normal_sim.run(acyclic_graph).data.to_string()
        asic_sim = asic_simulator.ASICSimulator()
        asic_result = asic_sim.compute_samples(acyclic_graph).data.to_string()
        # The sample should be entirely determanistic since we only used X building_blocks.
        self.assertEqual(normal_result, asic_result)

    def test_local_operations_only(self):
        discretes = linear_algebra.LinearSpace.range(NUM_DISCRETEDS)
        acyclic_graph = linear_algebra.Graph(
            [linear_algebra.flip_x_axis(discretes[1]), linear_algebra.flip_x_axis(discretes[5]), linear_algebra.flip_x_axis(discretes[8])]
        )
        acyclic_graph += [linear_algebra.measure(discretes[i]) for i in range(NUM_DISCRETEDS)]
        normal_sim = linear_algebra.Simulator()
        normal_result = normal_sim.run(acyclic_graph).data.to_string()
        asic_sim = asic_simulator.ASICSimulator()
        asic_result = asic_sim.compute_samples(acyclic_graph).data.to_string()
        # The sample should be entirely determanistic since we only used X building_blocks.
        self.assertEqual(normal_result, asic_result)

    def test_with_global_operation(self):
        discretes = linear_algebra.LinearSpace.range(NUM_DISCRETEDS)
        acyclic_graph = linear_algebra.Graph(
            [
                linear_algebra.flip_x_axis(discretes[0]),
                linear_algebra.flip_x_axis(discretes[3]),
                linear_algebra.flip_x_axis(discretes[15]),
            ]
        )
        acyclic_graph += [linear_algebra.measure(discretes[i]) for i in range(NUM_DISCRETEDS)]
        normal_sim = linear_algebra.Simulator()
        normal_result = normal_sim.run(acyclic_graph).data.to_string()
        asic_sim = asic_simulator.ASICSimulator()
        asic_result = asic_sim.compute_samples(acyclic_graph).data.to_string()
        # The sample should be entirely determanistic since we only used X building_blocks.
        self.assertEqual(normal_result, asic_result)

    def test_lots_of_swaps(self):
        discretes = linear_algebra.LinearSpace.range(NUM_DISCRETEDS)
        acyclic_graph = linear_algebra.Graph()
        acyclic_graph += linear_algebra.flip_x_axis(discretes[0])
        acyclic_graph += [
            linear_algebra.SWAP(discretes[i], discretes[i + 1]) for i in range(NUM_DISCRETEDS - 1)
        ]
        acyclic_graph += [linear_algebra.measure(discretes[i]) for i in range(NUM_DISCRETEDS)]
        normal_sim = linear_algebra.Simulator()
        normal_result = normal_sim.run(acyclic_graph).data.to_string()
        asic_sim = asic_simulator.ASICSimulator()
        asic_result = asic_sim.compute_samples(acyclic_graph).data.to_string()
        # The sample should be entirely determanistic since we only used
        # X and SWAP building_blocks.
        self.assertEqual(normal_result, asic_result)

    def test_lots_of_cnots(self):
        discretes = linear_algebra.LinearSpace.range(NUM_DISCRETEDS)
        acyclic_graph = linear_algebra.Graph()
        acyclic_graph += linear_algebra.flip_x_axis(discretes[0])
        acyclic_graph += [
            linear_algebra.exclusive_or(discretes[i], discretes[i + 1]) for i in range(NUM_DISCRETEDS - 1)
        ]
        acyclic_graph += [linear_algebra.measure(discretes[i]) for i in range(NUM_DISCRETEDS)]
        normal_sim = linear_algebra.Simulator()
        normal_result = normal_sim.run(acyclic_graph).data.to_string()
        asic_sim = asic_simulator.ASICSimulator()
        asic_result = asic_sim.compute_samples(acyclic_graph).data.to_string()
        # The sample should be entirely determanistic since we only used
        # X and SWAP building_blocks.
        self.assertEqual(normal_result, asic_result)

    def test_lots_of_repetitions(self):
        discretes = linear_algebra.LinearSpace.range(NUM_DISCRETEDS)
        acyclic_graph = linear_algebra.Graph()
        acyclic_graph += linear_algebra.flip_x_axis(discretes[0])
        acyclic_graph += [
            linear_algebra.exclusive_or(discretes[i], discretes[i + 1]) for i in range(NUM_DISCRETEDS - 1)
        ]
        acyclic_graph += [linear_algebra.measure(discretes[i]) for i in range(NUM_DISCRETEDS)]
        normal_sim = linear_algebra.Simulator()
        normal_result = normal_sim.run(
            acyclic_graph, repetitions=10000
        ).data.to_string()
        asic_sim = asic_simulator.ASICSimulator()
        asic_result = asic_sim.compute_samples(
            acyclic_graph, repetitions=10000
        ).data.to_string()
        # The sample should be entirely determanistic since we only used
        # X and SWAP building_blocks.
        self.assertEqual(normal_result, asic_result)

    def test_hard_acyclic_graph_inverted(self):
        # This test will run a "hard" acyclic_graph, and then run its inverse.
        # The final state should always be exactly the |00...> state.
        discretes = linear_algebra.GridSpace.rect(2, NUM_DISCRETEDS // 2 + NUM_DISCRETEDS % 2)
        acyclic_graph = ce.random_rotations_between_grid_interaction_subgraphs_acyclic_graph(
            discretes, 10
        )
        acyclic_graph += linear_algebra.inverse(acyclic_graph)
        acyclic_graph += [linear_algebra.measure(x) for x in discretes]
        asic_sim = asic_simulator.ASICSimulator()
        asic_result = asic_sim.compute_final_state_vector(acyclic_graph)
        expected = np.zeros(2 ** len(discretes))
        expected[0] = 1.0
        tol = np.finfo(np.float32).eps * 10
        np.testing.assert_allclose(
            np.array(asic_result.real).ravel(), expected.real, atol=tol, rtol=tol
        )
        np.testing.assert_allclose(
            np.array(asic_result.imag).ravel(), expected.imag, atol=tol, rtol=tol
        )

    def test_get_amplitudes_sanity_check(self):
        result = helpers.get_amplitudes(
            building_blocks=[np.eye(2)],
            discrete_indices_per_building_block=((0,),),
            num_discretes=NUM_DISCRETEDS,
            bitstrings=np.array([0, 1], dtype=np.uint32),
        )
        np.testing.assert_almost_equal(
            result, np.array([1.0 + 0.0j, 0.0 + 0.0j])
        )

    def test_get_amplitudes_sanity_check_imag(self):
        result = helpers.get_amplitudes(
            building_blocks=[np.eye(2) * 1j],  # Adds a global phase of 1j.
            discrete_indices_per_building_block=((0,),),
            num_discretes=NUM_DISCRETEDS,
            bitstrings=np.array([0, 1], dtype=np.uint32),
        )
        np.testing.assert_almost_equal(
            result, np.array([0.0 + 1.0j, 0.0 + 0.0j])
        )

    def test_compute_amplitudes_sanity_check(self):
        discretes = linear_algebra.LinearSpace.range(NUM_DISCRETEDS)
        acyclic_graph = linear_algebra.Graph()
        acyclic_graph += [linear_algebra.flip_x_axis(discretes[9]), linear_algebra.flip_y_axis(discretes[NUM_DISCRETEDS - 1])]
        # If I don't put this here, linear_algebra only runs a 2 discrete acyclic_graph.
        acyclic_graph += [linear_algebra.I(x) for x in discretes]
        sim = asic_simulator.ASICSimulator()
        # Check bit 9 and NUM_DISCRETEDS - 1 being flipped.
        result = sim.compute_amplitudes(acyclic_graph, [0b000000000100000000001])
        np.testing.assert_almost_equal(result, np.array([0.0 + 1.0j]))

    def test_compute_amplitudes_across_globals(self):
        discretes = linear_algebra.LinearSpace.range(NUM_DISCRETEDS)
        acyclic_graph = linear_algebra.Graph()
        acyclic_graph += [linear_algebra.flip_x_axis(discretes[1]), linear_algebra.flip_y_axis(discretes[2])]
        # If I don't put this here, linear_algebra only runs a 2 discrete acyclic_graph.
        acyclic_graph += [linear_algebra.I(x) for x in discretes]
        sim = asic_simulator.ASICSimulator()
        # Check bit 1 and 2 being flipped.
        result = sim.compute_amplitudes(acyclic_graph, [0b011000000000000000000])
        np.testing.assert_almost_equal(result, np.array([0.0 + 1.0j]))

    def test_hard_acyclic_graph_consistent_amplitudes(self):
        discretes = linear_algebra.GridSpace.rect(2, NUM_DISCRETEDS // 2 + NUM_DISCRETEDS % 2)
        acyclic_graph = ce.random_rotations_between_grid_interaction_subgraphs_acyclic_graph(
            discretes, 2
        )
        normal_sim = linear_algebra.Simulator()
        asic_sim = asic_simulator.ASICSimulator()
        # Randomly sampled from keyboard smashing distribution.
        bitstrings = [
            0b011000010000100010101,
            0b011111110101101001111,
            0b010000111011101000011,
            0b000011101111011011000,
            0b111110101101111111011,
        ]
        normal_results = normal_sim.compute_amplitudes(acyclic_graph, bitstrings)
        asic_results = asic_sim.compute_amplitudes(acyclic_graph, bitstrings)
        np.testing.assert_almost_equal(
            np.asarray(normal_results), np.asarray(asic_results)
        )

    def test_gradients_random_acyclic_graph(self):
        Nparams = 10
        Nexponents = 5
        depth = 30
        N = NUM_DISCRETEDS
        string_length = 2
        num_pbaxistrings = 2
        acyclic_graph, discretes, resolver = build_random_acyclic_graph(
            Nparams=Nparams, Nexponents=Nexponents, depth=depth, N=N
        )
        # we need to make sure that the acyclic_graph actually acts on all
        # discretes, otherwise getting the linear_algebra reference results is
        # complicated.
        acyclic_graph.append([linear_algebra.flip_x_axis(q) for q in discretes])

        op_discretes = []
        for op in acyclic_graph.all_operations():
            op_discretes.extend(list(op.discretes))
        op_discretes = sorted(list(set(op_discretes)))
        pbaxisum = generate_pbaxisum(num_pbaxistrings, op_discretes, string_length)
        prob_basis_axis_coeffs = [s.coefficient for s in pbaxisum]

        asic_sim = asic_simulator.ASICSimulator()
        actual_gradients, actual_expectations = asic_sim.compute_gradients(
            acyclic_graph, pbaxisum, resolver
        )
        # we need double precision here to get accurate gradients
        simulator = linear_algebra.Simulator(dtype=np.complex128)
        linear_algebra_result = simulator.simulate(acyclic_graph, resolver)
        params = linear_algebra.parameter_symbols(acyclic_graph)
        exp_acyclic_graphs = [linear_algebra.Graph() for _ in range(num_pbaxistrings)]
        accumulator = np.zeros_like(linear_algebra_result.final_state_vector)
        for n, pbaxistring in enumerate(pbaxisum):
            exp_acyclic_graphs[n] += [p(q) for q, p in pbaxistring.items()]
            obs_result = simulator.simulate(
                exp_acyclic_graphs[n],
                discrete_order=op_discretes,
                initial_state=linear_algebra_result.final_state_vector.ravel(),
            )
            accumulator += obs_result.final_state_vector * prob_basis_axis_coeffs[n]
        g1 = np.dot(linear_algebra_result.final_state_vector.conj(), accumulator)
        delta = 1e-8
        g2 = {}
        for param in params:
            shifted_dict = {k: v for k, v in resolver.param_dict.items()}
            shifted_dict[param.name] = resolver.param_dict[param.name] + delta
            shifted_resolver = linear_algebra.ParamResolver(shifted_dict)
            linear_algebra_result_shifted = simulator.simulate(acyclic_graph, shifted_resolver)
            accumulator = np.zeros_like(linear_algebra_result_shifted.final_state_vector)
            for n, pbaxistring in enumerate(pbaxisum):
                obs_result = simulator.simulate(
                    exp_acyclic_graphs[n],
                    discrete_order=op_discretes,
                    initial_state=linear_algebra_result_shifted.final_state_vector.ravel(),
                )
                accumulator += obs_result.final_state_vector * prob_basis_axis_coeffs[n]
            g2[param] = np.dot(
                linear_algebra_result_shifted.final_state_vector.conj(), accumulator
            )

        # testing
        np.testing.assert_allclose(
            g1, actual_expectations, atol=1e-5, rtol=1e-5
        )
        for s, g in actual_gradients.items():
            np.testing.assert_allclose(
                g, np.real(g2[s] - g1) / delta, atol=1e-5, rtol=1e-5
            )

    def test_expectations_random_acyclic_graph(self):
        Nparams = 10
        Nexponents = 5
        depth = 40
        N = NUM_DISCRETEDS
        string_length = 2
        num_pbaxistrings = 2
        num_pbaxisums = 2
        acyclic_graph, discretes, resolver = build_random_acyclic_graph(
            Nparams=Nparams, Nexponents=Nexponents, depth=depth, N=N
        )
        # we need to make sure that the acyclic_graph actually acts on all
        # discretes, otherwise getting the linear_algebra reference results is
        # complicated.
        acyclic_graph.append([linear_algebra.flip_x_axis(q) for q in discretes])

        pbaxisums = []
        for _ in range(num_pbaxisums):
            pbaxisums.append(
                generate_pbaxisum(num_pbaxistrings, discretes, string_length)
            )
        prob_basis_axis_coeffs = [[s.coefficient for s in ps] for ps in pbaxisums]

        asic_sim = asic_simulator.ASICSimulator()
        actual_expectations_1 = asic_sim.compute_expectations(
            acyclic_graph, pbaxisums, resolver
        )

        simulator = linear_algebra.Simulator()
        linear_algebra_result = simulator.simulate(acyclic_graph, resolver)
        exp_acyclic_graphs = [None] * num_pbaxisums
        expected = []
        for m, pbaxisum in enumerate(pbaxisums):
            exp_acyclic_graphs[m] = [linear_algebra.Graph() for _ in range(num_pbaxistrings)]
            accumulator = np.zeros_like(linear_algebra_result.final_state_vector)
            for n, pbaxistring in enumerate(pbaxisum):
                exp_acyclic_graphs[m][n] += [p(q) for q, p in pbaxistring.items()]
                obs_result = simulator.simulate(
                    exp_acyclic_graphs[m][n],
                    discrete_order=discretes,
                    initial_state=linear_algebra_result.final_state_vector.ravel(),
                )
                accumulator += (
                    obs_result.final_state_vector * prob_basis_axis_coeffs[m][n]
                )
            expected.append(
                np.dot(linear_algebra_result.final_state_vector.conj(), accumulator)
            )
        # testing
        np.testing.assert_allclose(
            expected, actual_expectations_1, atol=1e-5, rtol=1e-5
        )

    def test_final_state_vector_random_acyclic_graph(self):
        Nparams = 10
        Nexponents = 5
        depth = 40
        N = NUM_DISCRETEDS
        acyclic_graph, discretes, resolver = build_random_acyclic_graph(
            Nparams=Nparams, Nexponents=Nexponents, depth=depth, N=N
        )

        # we need to make sure that the acyclic_graph actually acts on all
        # discretes
        acyclic_graph.append([linear_algebra.flip_x_axis(q) for q in discretes])

        asic_sim = asic_simulator.ASICSimulator()
        actual = asic_sim.compute_final_state_vector(acyclic_graph, resolver)
        actual = to_array(actual)

        state = np.zeros(2 ** N)
        state[0] = 1.0
        state = state.reshape((2,) * N)
        simulator = linear_algebra.Simulator()

        expected = simulator.simulate(
            linear_algebra.resolve_parameters(acyclic_graph, resolver),
            discrete_order=discretes,
            initial_state=state.ravel(),
        )
        np.testing.assert_allclose(
            np.ravel(actual), expected.final_state_vector, atol=1e-5, rtol=1e-5
        )


class ASICSimulatorBasicMethodsBase(unittest.TestCase):
    num_discretes = NUM_DISCRETEDS

    def setUp(self) -> None:
        self.discretes = linear_algebra.LinearSpace.range(self.num_discretes)
        self.sym = sympy.Symbol("a")
        self.acyclic_graph = linear_algebra.Graph(
            [linear_algebra.rotate_x_axis(self.sym).on(q) for q in self.discretes]
        )
        self.progress_callback = unittest.mock.Mock()
        self.asic_sim = asic_simulator.ASICSimulator()


class ASICSimulatorBasicMethodsSampleTest(ASICSimulatorBasicMethodsBase):
    def test_compute_samples(self) -> None:
        self.acyclic_graph.append([linear_algebra.measure(q) for q in self.discretes])
        params = linear_algebra.ParamResolver({self.sym: sympy.pi})
        actual = self.asic_sim.compute_samples(
            self.acyclic_graph, params, 10, self.progress_callback
        )

        expected = linear_algebra.Simulator().run(self.acyclic_graph, params, 10)
        self.assertEqual(actual.data.to_string(), expected.data.to_string())
        self.progress_callback.assert_called_once_with(1, 1)

    def test_compute_samples_sweep(self) -> None:
        num_sweeps = 3
        self.acyclic_graph.append([linear_algebra.measure(q) for q in self.discretes])
        params = [{self.sym: sympy.pi * val} for val in range(num_sweeps)]

        actual = self.asic_sim.compute_samples_sweep(
            self.acyclic_graph, params, 10, self.progress_callback
        )
        expected = linear_algebra.Simulator().run_sweep(self.acyclic_graph, params, 10)
        self.assertEqual(actual, expected)
        # check progress is reported correctly
        self.progress_callback.assert_has_calls(
            [
                unittest.mock.call(i, num_sweeps)
                for i in range(1, num_sweeps + 1)
            ]
        )

    def test_compute_samples_batch(self) -> None:
        num_sweeps = 3
        num_batch = 2
        self.acyclic_graph.append([linear_algebra.measure(q) for q in self.discretes])
        params = [{self.sym: sympy.pi * val} for val in range(num_sweeps)]

        actual = self.asic_sim.compute_samples_batch(
            [self.acyclic_graph] * num_batch,
            [params] * num_batch,
            10,
            self.progress_callback,
        )
        self.assertEqual(*actual)
        # check progress is reported correctly
        total_work = num_sweeps * num_batch
        self.progress_callback.assert_has_calls(
            [
                unittest.mock.call(i, total_work)
                for i in range(1, total_work + 1)
            ]
        )


class ASICSimulatorBasicMethodsExpectationTest(ASICSimulatorBasicMethodsBase):
    def test_compute_expectations(self) -> None:
        params = linear_algebra.ParamResolver({self.sym: sympy.pi})
        pbaxisum = generate_pbaxisum(2, self.discretes, 2)
        actual = self.asic_sim.compute_expectations(
            self.acyclic_graph, [pbaxisum], params, self.progress_callback
        )

        expected = linear_algebra.Simulator().simulate_expectation_values(
            self.acyclic_graph, [pbaxisum], params
        )
        np.testing.assert_allclose(actual, expected, atol=1e-3)
        self.progress_callback.assert_called_once_with(1, 1)

    def test_compute_expectations_sweep(self) -> None:
        num_sweeps = 3
        pbaxisum = generate_pbaxisum(2, self.discretes, 2)
        params = [{self.sym: val} for val in range(num_sweeps)]

        actual = self.asic_sim.compute_expectations_sweep(
            self.acyclic_graph, [pbaxisum], params, self.progress_callback
        )
        expected = linear_algebra.Simulator().simulate_expectation_values_sweep(
            self.acyclic_graph, [pbaxisum], params
        )
        np.testing.assert_allclose(actual, expected, atol=1e-3)

        self.progress_callback.assert_has_calls(
            [
                unittest.mock.call(i, num_sweeps)
                for i in range(1, num_sweeps + 1)
            ]
        )

    def test_compute_expectations_batch(self) -> None:
        num_sweeps = 3
        num_batch = 2
        pbaxisum = generate_pbaxisum(2, self.discretes, 2)
        params = [{self.sym: val} for val in range(num_sweeps)]

        actual = self.asic_sim.compute_expectations_batch(
            [self.acyclic_graph] * num_batch,
            [[pbaxisum]] * num_batch,
            [params] * num_batch,
            self.progress_callback,
        )
        np.testing.assert_allclose(*actual, atol=1e-7)

        total_work = num_sweeps * num_batch
        self.progress_callback.assert_has_calls(
            [
                unittest.mock.call(i, total_work)
                for i in range(1, total_work + 1)
            ]
        )


class ASICSimulatorBasicMethodsTestQSIM(ASICSimulatorBasicMethodsSampleTest):
    """Basic Method tests for discrete ranges that default to QSIM"""

    num_discretes = 10


if __name__ == "__main__":
    unittest.main()
