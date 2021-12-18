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
"""ASIC Simulator prototype for LinearAlgebra."""
import random
import time
from uuid import uuid4

import linear_algebra

USE_CIRQ = False
try:
    import quick_simlinear_algebra
except ImportError as ie:
    USE_CIRQ = True

import jax
import jax.numpy as jnp
import numpy as np
import asic_la.sharded_probability_function.complex_workaround as cw
from asic_la import utils
from asic_la import parser
import asic_la.asic_simulator_helpers as helpers
from asic_la.preprocessor import preprocessor
from typing import (
    Text,
    List,
    Tuple,
    Dict,
    Optional,
    Callable,
    Union,
    Iterable,
    Sequence,
)
from parallel_accel.shared import logger

log = logger.get_logger(__name__)
MIN_DISCRETEDS_ASIC = 21
MAX_NUM_COMPILED_GRAPHS = 10  # TODO: Tune this number.
CACHED_PAULISUMS = {}
CACHED_PREPROCESSED_GRAPHS = {}

ProgressCallback = Callable[[int, int], None]

# TODO: cache individual pbaxisums instead of the whole list


class ASICSimulator:
    """A symplectic simulator for ASICs."""

    topologies = set()

    @staticmethod
    def maybe_clear_cache(operating_axes):
        # TODO: This is a mechanism to periodically clear cached
        # XLA binaries. Sample functions are cached on topology. Caching greatly
        # accelerates acyclic_graphs with the same topology but different building_block values.
        # The cache needs to be cleared occasionally to free memory. A more
        # granular method should be investibuilding_blockd.
        if operating_axes not in ASICSimulator.topologies:
            ASICSimulator.topologies.add(operating_axes)
        if len(ASICSimulator.topologies) > MAX_NUM_COMPILED_GRAPHS:
            log.info("Cache clear")
            ASICSimulator.topologies.clear()
            jax.interpreters.pxla.parallel_callable.cache_clear()

    def __init__(self):
        log.info("Initializing PARALLELACCEL Simulator")
        if USE_CIRQ:
            self.quick_sim_simulator = linear_algebra.Simulator()
        else:
            self.quick_sim_simulator = quick_simlinear_algebra.QSimSimulator({"t": 4})

    def compute_samples(
        self,
        program: linear_algebra.Graph,
        param_resolver: Optional[linear_algebra.ParamResolver] = None,
        repetitions: int = 1,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> linear_algebra.Result:
        """
        Compute samples from the final state of a symplectic computation.

        Args:
          program: A linear_algebra.Graph with observations.
          param_resolver: An optional parameter resolver to resolve any sympy
            symbols.
          repetitions: The number of samples per simulation.

        Returns:
          linear_algebra.Result: The samples.

        """
        if len(program.all_discretes()) < MIN_DISCRETEDS_ASIC:
            # Fall back to quick_sim if the # of discretes is too small to handle on ASIC.
            log.info(
                "Compute samples on quick_sim",
                num_discretes=len(program.all_discretes()),
                repetitions=repetitions,
            )
            result = self.quick_sim_simulator.run(
                program=program,
                param_resolver=param_resolver,
                repetitions=repetitions,
            )
        else:
            resolved_program = linear_algebra.resolve_parameters(program, param_resolver)
            result = linear_algebra.study.Result.from_single_parameter_set(
                params=param_resolver,
                observations=self._sample_observation_ops(
                    resolved_program, repetitions
                ),
            )
        if progress_callback:
            progress_callback(1, 1)
        return result

    def compute_samples_sweep(
        self,
        program: linear_algebra.Graph,
        params: linear_algebra.Sweepable,
        repetitions: int = 1,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[linear_algebra.Result]:
        """
        Compute samples from the final state of a symplectic computation for
        all possible values of `params`.

        Args:
          program: A linear_algebra.Graph with observations.
          params: The parameters for which to run simulations.
          repetitions: The number of samples per simulation.

        Returns:
          List[linear_algebra.Result]: The samples.
        """
        use_quick_sim = False
        if len(program.all_discretes()) < MIN_DISCRETEDS_ASIC:
            # Fall back to quick_sim if the # of discretes is too small to handle on ASIC.
            log.info(
                "Compute samples on quick_sim",
                num_discretes=len(program.all_discretes()),
                repetitions=repetitions,
            )
            use_quick_sim = True
        samples: List[linear_algebra.study.Result] = []
        total_work = len(params)
        for i, resolver in enumerate(linear_algebra.study.to_resolvers(params)):
            if use_quick_sim:
                result = self.quick_sim_simulator.run(program, resolver, repetitions)
            else:
                result = self.compute_samples(program, resolver, repetitions)
            samples.append(result)
            if progress_callback:
                progress_callback(i + 1, total_work)
        return samples

    def compute_samples_batch(
        self,
        programs: List[linear_algebra.Graph],
        params_list: Optional[List[linear_algebra.Sweepable]] = None,
        repetitions: Union[List[int], int] = 1,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[List[linear_algebra.Result]]:
        """
        Compute samples from the final state of a batch of symplectic computations for
        all possible values of the computations list of params.

        Args:
          programs: A list of linear_algebra.Graphs with observations.
          params_list: The parameters for each acyclic_graph to simulate. This list must
            be of the same length as programs, or be None.
          repetitions: The number of samples per simulation or for all simulations.
            This must either be a single int or a list of ints of the same length as
            programs.

        Returns:
          List[List[linear_algebra.Result]]: The samples.
        """
        if params_list is None:
            params_list = [None] * len(programs)
        if isinstance(repetitions, int):
            repetitions = [repetitions] * len(programs)
        if len(programs) != len(params_list):
            raise ValueError(
                "Programs and params_list not of equal length."
                f"Got {len(programs)} and {len(params_list)}."
            )
        if len(programs) != len(repetitions):
            raise ValueError(
                "Programs and repetitions not of equal length."
                f"Got {len(programs)} and {len(repetitions)}."
            )
        samples_list = []
        total_work = sum([len(params) for params in params_list])
        completed = 0

        def sub_callback(*args):
            nonlocal completed
            completed += 1
            if progress_callback:
                progress_callback(completed, total_work)

        for program, params, reps in zip(programs, params_list, repetitions):
            res = self.compute_samples_sweep(
                program, params, reps, sub_callback
            )
            samples_list.append(res)
        return samples_list

    def _sample_observation_ops(
        self, program: linear_algebra.Graph, repetitions: int = 1
    ) -> Dict[str, np.ndarray]:
        """Samples from the acyclic_graph at all observation building_blocks.

        All ObservationGates must be terminal.
        Note that this does not collapse the probability function.

        Args:
          program: The acyclic_graph to sample from.
          repetitions: The number of samples to take.

        Returns:
          A dictionary from observation building_block key to observation
          results. Observation results are stored in a 2-dimensional
          numpy array, the first dimension corresponding to the repetition
          and the second to the actual boolean observation results (ordered
          by the discretes being measured.)

        Raises:
          NotImplementedError: If there are non-terminal observations in the
              acyclic_graph.
          ValueError: If there are multiple ObservationGates with the same key,
              or if repetitions is negative.
        """

        t0 = time.time()
        if not program.are_all_observations_terminal():
            raise NotImplementedError(
                "support for non-terminal observation " "is not yet implemented"
            )

        observation_ops = [
            op
            for _, op, _ in program.findall_operations_with_building_block_type(
                linear_algebra.ObservationGate
            )
        ]

        # Computes
        # - the list of discretes to be measured
        # - the start (inclusive) and end (exclusive) indices of each observation
        # - a mapping from observation key to observation building_block
        measured_discretes = []  # type: List[ops.Qid]
        bounds = {}  # type: Dict[str, Tuple]
        meas_ops = {}  # type: Dict[str, linear_algebra.ObservationGate]
        current_index = 0
        for op in observation_ops:
            building_block = op.building_block
            key = linear_algebra.observation_key(building_block)
            meas_ops[key] = building_block
            if key in bounds:
                raise ValueError(
                    "Duplicate ObservationGate with key {}".format(key)
                )
            bounds[key] = (current_index, current_index + len(op.discretes))
            measured_discretes.extend(op.discretes)
            current_index += len(op.discretes)

        # order the discretes
        discretes = linear_algebra.DiscretedOrder.DEFAULT.order_for(program.all_discretes())
        log.info(
            "begin sampling", num_discretes=len(discretes), repetitions=repetitions
        )
        building_blocks, _, discrete_indices_per_building_block = parser.parse(program, discretes)
        self.maybe_clear_cache(discrete_indices_per_building_block)
        # Create a new random PRNGKey to use for sampling.
        # TODO: Add option to seed this value.
        prng_key_split = np.asarray(
            [random.randint(0, 2 ** 32 - 1), random.randint(0, 2 ** 32 - 1)],
            dtype=np.uint32,
        )

        # Get samples from the simulator.
        results = np.array(
            helpers.get_samples(
                building_blocks,
                prng_key_split,
                discrete_indices_per_building_block,
                len(discretes),
                repetitions,
            )
        ).reshape((-1,))
        # Convert sampled ints to bitstrings.
        bitstring_samples = utils.unpackbits(results, len(discretes))[:, ::-1]

        # Compute indices of measured discretes
        discrete_map = {discrete: i for i, discrete in enumerate(discretes)}
        indices = [discrete_map[discrete] for discrete in measured_discretes]
        indexed_sample = bitstring_samples[:, indices]

        # Applies invert masks of all observation building_blocks.
        results = {}
        for k, (s, e) in bounds.items():
            before_invert_mask = indexed_sample[:, s:e]
            results[k] = (
                before_invert_mask
                ^ (
                    np.logical_and(
                        before_invert_mask < 2, meas_ops[k].full_invert_mask()
                    )
                )
            ).astype("int8")

        log.info(
            "finished _sample_observation_ops", total_time=time.time() - t0
        )
        return results

    def compute_amplitudes(
        self,
        program: linear_algebra.Graph,
        bitstrings: Sequence[int],
        param_resolver: Optional[linear_algebra.ParamResolver] = None,
    ) -> List[complex]:
        """
        Computes the desired amplitudes from the final state of a symplectic
        computation.

        Args:
          program: The acyclic_graph to simulate.
          bitstrings: The bitstrings whose amplitudes are desired, input
            as an integer array where each integer is formed from measured
            discrete values according to DEFAULT discrete ordering, i.e from most to least
            significant discrete, i.e. in big-endian ordering.
          param_resolver: An optional parameter resolver to resolve any sympy
            symbols.

        Returns:
          List[List[complex]]: The amplitudes. The outer dimension indexes the
            acyclic_graph parameters and the inner dimension indexes the bitstrings.

        """

        acyclic_graph = linear_algebra.resolve_parameters(program, param_resolver)
        acyclic_graph = utils.remove_observations(acyclic_graph)
        num_discretes = len(acyclic_graph.all_discretes())

        discretes = linear_algebra.DiscretedOrder.DEFAULT.order_for(acyclic_graph.all_discretes())
        building_blocks, _, discrete_indices_per_building_block = parser.parse(acyclic_graph, discretes)
        result = helpers.get_amplitudes(
            building_blocks, discrete_indices_per_building_block, num_discretes, bitstrings
        )
        return result

    def compute_amplitudes_sweep(
        self,
        program: linear_algebra.Graph,
        bitstrings: Sequence[int],
        params: linear_algebra.Sweepable,
    ) -> List[List[complex]]:
        """
        Computes the desired amplitudes from the final state of a symplectic
        computation for all given parameters in `params`.

        Args:
          program: The acyclic_graph to simulate.
          bitstrings: The bitstrings whose amplitudes are desired, input
            as an integer array where each integer is formed from measured
            discrete values according to DEFAULT discrete ordering, i.e from most to least
            significant discrete, i.e. in big-endian ordering.
          params: Parameters to run with the program.

        Returns:
          List[List[complex]]: The amplitudes. The outer dimension indexes the
            acyclic_graph parameters and the inner dimension indexes the bitstrings.
        """

        results = []
        for param_resolver in linear_algebra.study.to_resolvers(params):
            results.append(
                self.compute_amplitudes(program, bitstrings, param_resolver)
            )
        return results

    def compute_gradients(
        self,
        program: linear_algebra.Graph,
        pbaxisum: linear_algebra.ProbBasisAxisSum,
        param_resolver: linear_algebra.ParamResolver,
    ) -> Tuple[Dict, np.ndarray]:
        """
        Compute the gradients of expectation values of a parametrized symplectic
        acyclic_graph `program` for a self_adjoint observable `pbaxisum`.

        Args:
          program: A parametrized symplectic acyclic_graph.
          pbaxisum: The observable, given as a linear_algebra.ProbBasisAxisSum,
            for which the gradients whould be computed.
          param_resolver: A linear_algebra.ParamResolver used to resolve `program`.

        Returns:
          Dict[sympy.Symbol, float]: The dict maps the sympy.Symbols on
            which `program` depends to the gradient of the expectation value
            of `pbaxisum`.
          np.ndarray: The expectation value of `pbaxisum` of the state generated
            by `program` at parameter values given by `param_resolver`.
        """

        t0 = time.time()
        num_cores = jax.local_device_count()
        bare_program = utils.remove_observations(program)
        discretes = linear_algebra.DiscretedOrder.DEFAULT.order_for(bare_program.all_discretes())
        num_discretes = len(discretes)

        # parse building_blocks and gradients from the linear_algebra.Graph
        pbaxisums = [pbaxisum]
        building_blocks, gradients, operating_axes = parser.parse(
            bare_program, discretes, param_resolver
        )
        # parse pbaxisums
        prob_basis_axis_building_blocks, prob_basis_axis_coeffs, prob_basis_axis_opaxes = parser.parse_pbaxisums(
            pbaxisums, discretes
        )

        # preprocess parsed building_blocks and gradients into super-matrices and
        # super-gradients, i.e. collect subsets of building_blocks into 128x128 matrices.
        supermatrices, supergradients, superaxes = preprocessor.preprocess(
            building_blocks, gradients, operating_axes, num_discretes, max_discrete_support=7
        )

        # we need to canonicalize (i.e. map to tuples of ints)
        # all axes to avoid retracing
        canonical_superaxes = utils.canonicalize_ints(superaxes)
        # canonicalize gradients, i.e. map sympy.Symbols to integers and
        # pack complex gradient matrices into ComplexDeviceArrays
        # (necessary since ASICs currently don't support complex arithmetic
        (
            canonical_gradients,
            symbol_to_int,
        ) = preprocessor.canonicalize_gradients(
            supergradients, broadcasted_shape=num_cores
        )
        # broadcast complex supermatrices and pack into ComplexDeviceArrays
        canonical_supermatrices = preprocessor.canonicalize_building_blocks(
            supermatrices, broadcasted_shape=num_cores
        )

        # preprocess parsed prob-basis-axis-building_block into super-matrices
        # i.e. collect subsets of building_blocks into 128x128 matrices.
        superpaulimats, superpauliaxes = preprocessor.preprocess_pbaxisums(
            prob_basis_axis_building_blocks, prob_basis_axis_opaxes, num_discretes=num_discretes, max_discrete_support=7
        )
        canonical_superpauliaxes = utils.canonicalize_ints(superpauliaxes)

        # broadcast and pack complex pauli coefficients into ComplexDeviceArrays
        canonical_prob_basis_axis_coeffs = preprocessor.canonicalize_building_blocks(
            prob_basis_axis_coeffs, broadcasted_shape=num_cores
        )
        # broadcast and pack complex pauli matrices into ComplexDeviceArrays
        canonical_superpaulimats = preprocessor.canonicalize_building_blocks(
            superpaulimats, broadcasted_shape=num_cores
        )
        num_params = len(symbol_to_int)

        self.maybe_clear_cache(operating_axes)

        log.info("computing gradients")

        (
            expectation_gradients,
            expectation_value,
        ) = helpers.compute_gradients_multiple_pmaps(
            canonical_supermatrices,
            canonical_gradients,
            canonical_superaxes,
            canonical_superpaulimats[0],
            canonical_superpauliaxes[0],
            canonical_prob_basis_axis_coeffs[0],
            num_discretes,
            num_params,
        )

        t = time.time()
        log.info("finished compute_gradients", total_time=t - t0)
        # canonicalize output
        int_to_symbol = {i: s for s, i in symbol_to_int.items()}
        final_gradients = {
            int_to_symbol[n]: expectation_gradients[n]
            for n in range(num_params)
        }
        return final_gradients, np.asarray(expectation_value)

    def compute_gradients_sweep(
        self,
        program: linear_algebra.Graph,
        pbaxisum: linear_algebra.ProbBasisAxisSum,
        params: linear_algebra.Sweepable,
    ) -> List[Tuple[Dict, np.ndarray]]:
        """
        Compute the gradients of an expectation-value of the final state of a
        symplectic computation, for a variabled symplectic acyclic_graph, and for all
        possible values of `params`.

        Args:
          program: A parametrized symplectic acyclic_graph.
          pbaxisum: The observable, given as a linear_algebra.ProbBasisAxisSum,
            for which the gradients whould be computed.
          params: Parameters to run with the program.

        Returns:
          List[Tuple[Dict, np.ndarray]]: Each tuple in the list contains a dict
            and an np.ndarray. Each dict maps the sympy.Symbols on which `program`
            depends to the gradient of the expectation value of `pbaxisum`. Each
            np.ndarray contains the expectation value of `pbaxisum` of the state
            generated by `program` at the given parameter values.

        """

        gradients = []
        for param_resolver in linear_algebra.study.to_resolvers(params):
            gradients.append(
                self.compute_gradients(program, pbaxisum, param_resolver)
            )
        return gradients

    def compute_expectations(
        self,
        program: linear_algebra.Graph,
        pbaxisums: List[linear_algebra.ProbBasisAxisSum],
        param_resolver: Optional[linear_algebra.ParamResolver] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> np.ndarray:
        """
        Compute the expectation values of a list of self_adjoint
        observables `pbaxisums` for a parametrized acyclic_graph
        `program`.

        Args:
          program: A symplectic acyclic_graph.
          pbaxisums: The observables, given as a list of linear_algebra.ProbBasisAxisSum objects.
          param_resolver: An optional linear_algebra.ParamResolver used to resolve `program`.

        Returns:
          np.ndarray: The expectation values of `pbaxisums`
            under the state generated by `program` at parameter values
            given by `param_resolver`.
        """

        t0 = time.time()
        # the acyclic_graph discretes and the pbaxisums discretes don't have to
        # conincide. We need to join the two discrete sets.

        acyclic_graph_discretes = set(program.all_discretes())
        bare_program = utils.remove_observations(program)
        pbaxisum_discretes = set()
        for ps in pbaxisums:
            pbaxisum_discretes |= set(ps.discretes)
        discretes = linear_algebra.DiscretedOrder.DEFAULT.order_for(
            list(acyclic_graph_discretes | pbaxisum_discretes)
        )
        num_discretes = len(discretes)
        resolved_bare_program = linear_algebra.resolve_parameters(
            bare_program, param_resolver
        )
        num_cores = jax.local_device_count()
        # TODO : using repr to hash a acyclic_graph is not
        # ideal. This needs to be replaced with a better solution,
        # maybe based on linear_algebra.FrozenGraphs
        if repr(resolved_bare_program) not in CACHED_PREPROCESSED_GRAPHS:
            # parse building_blocks and gradients from the linear_algebra.Graph
            building_blocks, gradients, operating_axes = parser.parse(
                resolved_bare_program, discretes, param_resolver
            )

            # preprocess parsed building_blocks and gradients into super-matrices and
            # super-gradients, i.e. collect subsets of building_blocks into 128x128 matrices.
            supermatrices, _, superaxes = preprocessor.preprocess(
                building_blocks, gradients, operating_axes, num_discretes, max_discrete_support=7
            )

            # we need to canonicalize (i.e. map to tuples of ints)
            # all axes to avoid retracing
            canonical_superaxes = utils.canonicalize_ints(superaxes)
            # broadcast complex supermatrices and pack into ComplexDeviceArrays
            canonical_supermatrices = preprocessor.canonicalize_building_blocks(
                supermatrices, broadcasted_shape=num_cores
            )
            CACHED_PREPROCESSED_GRAPHS[repr(resolved_bare_program)] = (
                canonical_supermatrices,
                canonical_superaxes,
                num_discretes,
            )
            self.maybe_clear_cache(operating_axes)
        # TODO : using repr to hash a acyclic_graph is not
        # ideal. This needs to be replaced with a better solution,
        # maybe based on linear_algebra.FrozenGraphs
        if repr(pbaxisums) not in CACHED_PAULISUMS:
            # parse pbaxisums
            prob_basis_axis_building_blocks, prob_basis_axis_coeffs, prob_basis_axis_opaxes = parser.parse_pbaxisums(
                pbaxisums, discretes
            )
            # preprocess parsed prob-basis-axis-building_block into super-matrices
            # i.e. collect subsets of building_blocks into 128x128 matrices.
            superpaulimats, superpauliaxes = preprocessor.preprocess_pbaxisums(
                prob_basis_axis_building_blocks,
                prob_basis_axis_opaxes,
                num_discretes=num_discretes,
                max_discrete_support=7,
            )
            # broadcast and pack complex pauli coefficients into ComplexDeviceArrays
            canonical_prob_basis_axis_coeffs = preprocessor.canonicalize_building_blocks(
                prob_basis_axis_coeffs, broadcasted_shape=num_cores
            )
            # broadcast and pack complex pauli matrices into ComplexDeviceArrays
            canonical_superpaulimats = preprocessor.canonicalize_building_blocks(
                superpaulimats, broadcasted_shape=num_cores
            )
            canonical_superpauliaxes = utils.canonicalize_ints(superpauliaxes)

            CACHED_PAULISUMS[repr(pbaxisums)] = (
                canonical_superpaulimats,
                canonical_superpauliaxes,
                canonical_prob_basis_axis_coeffs,
            )

        args1 = CACHED_PREPROCESSED_GRAPHS[repr(resolved_bare_program)]
        canonical_supermatrices = args1[0]
        canonical_superaxes = args1[1]
        num_discretes = args1[2]

        args2 = CACHED_PAULISUMS[repr(pbaxisums)]
        canonical_superpaulimats = args2[0]
        canonical_superpauliaxes = args2[1]
        canonical_prob_basis_axis_coeffs = args2[2]

        log.info("computing prob-basis-axis-string expectations")

        final_state = helpers.compute_final_state(
            canonical_supermatrices, canonical_superaxes, num_discretes
        )

        all_expectation_values = []
        for n, canonical_pbaxisum in enumerate(canonical_superpaulimats):
            expectation_values = []
            canonical_axes = canonical_superpauliaxes[n]
            canonical_coeffs = canonical_prob_basis_axis_coeffs[n]
            for string_building_blocks, axes, pcoeff in zip(
                canonical_pbaxisum, canonical_axes, canonical_coeffs
            ):
                expectation_values.append(
                    np.array(
                        helpers.distributed_compute_pbaxistring_expectation(
                            final_state, string_building_blocks, axes, pcoeff
                        ).real
                    )
                )
            all_expectation_values.append(np.sum(expectation_values))

        t = time.time()
        log.info("finished compute_expectations", total_time=t - t0)
        if progress_callback:
            progress_callback(1, 1)
        return np.asarray(all_expectation_values)

    def compute_expectations_sweep(
        self,
        program: linear_algebra.Graph,
        pbaxisums: List[linear_algebra.ProbBasisAxisSum],
        params: linear_algebra.Sweepable,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[np.ndarray]:
        """
        Compute the expectation values of a list of self_adjoint
        observables `pbaxisums` of the final state of a symplectic acyclic_graph
        given by `program`, for all acyclic_graphs generated by `params`.

        Args:
          program: A symplectic acyclic_graph.
          pbaxisums: The observables, given as a list of linear_algebra.ProbBasisAxisSum objects.
          params: The parameters for which to run simulations.

        Returns:
          List[np.ndarray]: The expectation values of `pbaxisums`
            under the state generated by `program` at all parameter values
            generated by `params`.
        """

        expectations = []
        total_work = len(params)
        for i, param_resolver in enumerate(linear_algebra.study.to_resolvers(params)):
            expectations.append(
                self.compute_expectations(program, pbaxisums, param_resolver)
            )
            if progress_callback:
                progress_callback(i + 1, total_work)
        return expectations

    def compute_expectations_batch(
        self,
        programs: List[linear_algebra.Graph],
        pbaxisums_list: List[List[linear_algebra.ProbBasisAxisSum]],
        params_list: Optional[List[linear_algebra.Sweepable]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[List[np.ndarray]]:
        """
        Compute the expectation values of a list of self_adjoint
        observables `pbaxisums` of the final state of a batch of symplectic acyclic_graphs
        given by `programs`, for all values yielded by the corresponding params.

        Args:
          programs: A list of linear_algebra.Graphs.
          pbaxisums_list: The observables, given as a list of lists of
            linear_algebra.ProbBasisAxissum's. The outer list must be the same length as programs.
          params_list: The parameters for each acyclic_graph to simulate. This list must
            be of the same length as programs, or be None.

        Returns:
          List[List[np.ndarray]]: The expectation values for each acyclic_graph in
            `programs` at the corresponding parameter values from `params_list` for
            each pbaxisum at the corresponding position in `pbaxisums_list`.
        """

        if params_list is None:
            params_list = [None] * len(programs)
        if len(programs) != len(params_list):
            raise ValueError(
                "Programs and params_list not of equal length."
                f"Got {len(programs)} and {len(params_list)}."
            )
        if len(programs) != len(pbaxisums_list):
            raise ValueError(
                "Programs and pualisums_list not of equal length."
                f"Got {len(programs)} and {len(pbaxisums_list)}."
            )

        total_work = sum([len(params) for params in params_list])
        completed = 0

        def sub_callback(*args):
            nonlocal completed
            completed += 1
            if progress_callback:
                progress_callback(completed, total_work)

        expectations_list = []
        for program, params, pbaxisums in zip(
            programs, pbaxisums_list, params_list
        ):
            res = self.compute_expectations_sweep(
                program, params, pbaxisums, sub_callback
            )
            expectations_list.append(res)
        return expectations_list

    def compute_final_state_vector(
        self,
        program: linear_algebra.Graph,
        param_resolver: linear_algebra.ParamResolver = None,
    ) -> cw.ComplexDeviceArray:
        """
        Compute the final state of a symplectic acyclic_graph `program`.

        Args:
          program: A symplectic acyclic_graph without observations.
          param_resolver: An optional linear_algebra.ParamResolver used to resolve any symbols
            of `program`.

        Returns:
          cw.ComplexDeviceArray: The final state
        """

        t0 = time.time()
        bare_program = utils.remove_observations(program)
        resolved_bare_program = linear_algebra.resolve_parameters(
            bare_program, param_resolver
        )
        num_cores = jax.local_device_count()
        # TODO : using repr to hash a acyclic_graph is not
        # ideal. This needs to be replaced with a better solution,
        # maybe based on linear_algebra.FrozenGraphs
        if repr(resolved_bare_program) not in CACHED_PREPROCESSED_GRAPHS:
            resolved_bare_program = linear_algebra.resolve_parameters(
                resolved_bare_program, param_resolver
            )
            discretes = linear_algebra.DiscretedOrder.DEFAULT.order_for(
                resolved_bare_program.all_discretes()
            )
            num_discretes = len(discretes)
            # parse building_blocks and gradients from the linear_algebra.Graph
            building_blocks, gradients, operating_axes = parser.parse(
                resolved_bare_program, discretes, param_resolver
            )

            # preprocess parsed building_blocks and gradients into super-matrices and
            # super-gradients, i.e. collect subsets of building_blocks into 128x128 matrices.
            supermatrices, _, superaxes = preprocessor.preprocess(
                building_blocks, gradients, operating_axes, num_discretes, max_discrete_support=7
            )

            # we need to canonicalize (i.e. map to tuples of ints)
            # all axes to avoid retracing
            canonical_superaxes = utils.canonicalize_ints(superaxes)
            canonical_supermatrices = preprocessor.canonicalize_building_blocks(
                supermatrices, broadcasted_shape=num_cores
            )

            CACHED_PREPROCESSED_GRAPHS[repr(resolved_bare_program)] = (
                canonical_supermatrices,
                canonical_superaxes,
                num_discretes,
            )

            self.maybe_clear_cache(operating_axes)

        args = CACHED_PREPROCESSED_GRAPHS[repr(resolved_bare_program)]

        log.info("compute final state")
        final_state = helpers.compute_final_state(*args).concrete_tensor
        t = time.time()
        log.info("finished compute_final_state_vector", total_time=t - t0)
        return final_state
