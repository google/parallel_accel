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

"""This module defines core components for E2E testing.

In order to easily create resuable tests components, two helper classes were
introduced:
   - GraphProperties is a dataclass that wraps common properties for all e2e
     tests: number of acyclic_graph discretes, symbols and parameter resolvers.
   - AbstractGraphProvider is an abstract class that handles creating acyclic_graphs
     from the input GraphProperties parameters. The concrete implementation
     should define actual logic for creating linear_algebra object (`_generate_XXX`
     methods).

All tests are expected to inherit AbstractTest class, which spawns and
initializes ParallelAccel Client for local testing. Additionally, following properties
must be provided:
   - `properties` attribute that should return `GraphProperties` type object
   - `provider_class` attribute that should return `AbstractGraphTest` type
      class

There are also few extra abstract classes that inherits AbstractTest class and
provides preconfigured GraphProperties/custom GraphProvider classes:
   - `AbstractGraphTest` is a general class for testing simulator function
     results correctness.
   - `AbstractComplicatedGraphTest` is a general class for testing simulator
     function against more complicated symplectic acyclic_graph and verifying results
     accuracy.
   - `AbstractSweepTest` is a general class for testing simulator sweep function
     results correctness.
   - `AbstractBatchTest` is a general class for testing simulator batch function
     results correctness.
"""

import abc
import dataclasses
import logging
import random
import time
from typing import Any, Callable, List, Optional, Tuple
import unittest
import linear_algebra
import numpy as np
import sympy

from parallel_accel.client import containers
from test_config import client_config


@dataclasses.dataclass(frozen=True)
class GraphProperties:
    """Symplectic acyclic_graph properties.

    Properties:
        param_resolvers: Number of input ParamResolver objects.
        discretes: Number of acyclic_graph discretes.
        symbols: Number of symbols.
    """

    param_resolvers: int
    discretes: int
    symbols: int


class AbstractGraphProvider(abc.ABC):
    """A helper class that provides symplectic acyclic_graph for the end to end tests."""

    def __init__(self, properties: GraphProperties) -> None:
        """Creates GraphProvider class instance.

        Args:
            properties: Graph properties.
        """
        self._properties = properties

        self._acyclic_graph: Optional[linear_algebra.Graph] = None
        self._param_resolvers: Optional[
            List[linear_algebra.ParamResolverOrSimilarType]
        ] = None
        self._discretes: Optional[List[linear_algebra.GridSpace]] = None
        self._symbols: Optional[Tuple[sympy.Symbol]] = None

    @property
    def acyclic_graph(self) -> linear_algebra.Graph:
        """Graph to be simulated."""
        if self._acyclic_graph:
            return self._acyclic_graph

        self._generate_acyclic_graph()
        return self._acyclic_graph

    @property
    def param_resolvers(self) -> List[linear_algebra.ParamResolverOrSimilarType]:
        """List of param resolvers."""
        if self._param_resolvers:
            return self._param_resolvers

        self._generate_param_resolver()
        return self._param_resolvers

    @property
    def symbols(self) -> Tuple[sympy.Symbol]:
        """Symbols for the acyclic_graph."""
        if self._symbols:
            return self._symbols

        self._generate_symbols()
        return self._symbols

    @property
    def discretes(self) -> List[linear_algebra.GridSpace]:
        """Discreteds for the acyclic_graph."""
        if self._discretes:
            return self._discretes

        self._generate_discretes()
        return self._discretes

    @abc.abstractmethod
    def _generate_acyclic_graph(self) -> None:
        """Generates acyclic_graph from the input properties."""

    @abc.abstractmethod
    def _generate_param_resolver(self) -> None:
        """Generates param resolvers from the input properties."""

    def _generate_discretes(self) -> None:
        """Generates discretes from the input properties."""
        self._discretes = linear_algebra.GridSpace.rect(1, self._properties.discretes)

    @abc.abstractmethod
    def _generate_symbols(self) -> None:
        """Generates symbols from the input properties."""


class AbstractTest(unittest.TestCase, abc.ABC):
    """Base class for e2e tests."""

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.container = containers.Client()

        # Load client configuration from dict
        cls.container.core.config.from_dict(client_config)
        cls.sim = cls.container.simulators.LinearAlgebraSimulator()

        cls.provider: AbstractGraphProvider = None

    @classmethod
    def tearDown(cls) -> None:
        """See base class documentation."""
        cls.container.shutdown_resources()

    def setUp(self) -> None:
        """See base class documentation."""
        self.provider = self.provider_class(self.properties)

    @property
    @abc.abstractmethod
    def properties(self) -> GraphProperties:
        """Graph properties."""

    @property
    @abc.abstractmethod
    def provider_class(self) -> AbstractGraphProvider:
        """GraphProvider class."""


class AbstractGraphTest(AbstractTest):
    """Base class for complicated acyclic_graph testing."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties:
        """Symplectic acyclic_graph properties.

        Properties:
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            symbols: Number of symbols.
        """

        param_resolvers: int = dataclasses.field(default=2)
        discretes: int = dataclasses.field(default=26)
        symbols: int = dataclasses.field(default=123)

    class GraphProvider(AbstractGraphProvider):
        """Provides symplectic acyclic_graph for results correctness validation."""

        def _generate_acyclic_graph(self) -> None:
            pbaxis = [linear_algebra.flip_x_axis, linear_algebra.flip_y_axis, linear_algebra.flip_z_axis]

            self._acyclic_graph = linear_algebra.Graph()
            # pylint: disable=invalid-name
            for s in self.symbols:
                for q in self.discretes:
                    self._acyclic_graph += random.choice(pbaxis)(q) ** s
            for q in self.discretes:
                self._acyclic_graph += linear_algebra.measure(q)
            # pylint: enable=invalid-name

        def _generate_param_resolver(self) -> None:
            self._param_resolvers = [
                linear_algebra.ParamResolver(
                    {str(s): random.random() for s in self.symbols}
                )
            ] * self._properties.param_resolvers

        def _generate_symbols(self) -> None:
            self._symbols = sympy.symbols(f"s0:{self._properties.symbols}")

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.logger = logging.getLogger(cls.__name__)

    @property
    def provider_class(self) -> GraphProvider:
        return AbstractGraphTest.GraphProvider

    def _benchmark(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Benchmarks simulator function.

        Args:
            func: Simulator function to be called.

        Returns:
            Simulation result.
        """
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        self.logger.debug("%s took %d seconds", func.__name__, end - start)
        return result

    @abc.abstractmethod
    def _verify_result(self, result: Any) -> None:
        """Verifies simulation result.

        Args:
            result: Simulation result.
        """


class AbstractComplicatedGraphTest(AbstractTest):
    """Base class for complicated acyclic_graph testing."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties:
        """Symplectic acyclic_graph properties.

        Properties:
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            symbols: Number of symbols.
        """

        param_resolvers: int = dataclasses.field(default=1)
        discretes: int = dataclasses.field(default=26)
        symbols: int = dataclasses.field(default=1)

    class GraphProvider(AbstractGraphProvider):
        """Provides complicated symplectic acyclic_graph for results accuracy
        verification."""

        def __init__(self, properties: GraphProperties) -> None:
            """Creates ComplicatedGraphProvider class instance.

            Args:
                properties: Graph properties.
            """
            super().__init__(properties)
            self._angle: Optional[float] = None

        @property
        def angle(self) -> float:
            """Angle for the acyclic_graph."""
            if self._angle:
                return self._angle

            self._angle = random.uniform(-5, 5)
            return self._angle

        def _generate_acyclic_graph(self) -> None:
            # GHZ state entangles everything, expect only two kinds of returns.
            # Probability of each possible return is sinusoidal in the angle.
            self._acyclic_graph = linear_algebra.Graph()
            self._acyclic_graph += linear_algebra.flip_x_axis_angle(
                exponent=(self.symbols[0] / np.pi), global_shift=-0.5
            )(self.discretes[0])

            # pylint: disable=invalid-name
            for q0, q1 in zip(self.discretes, self.discretes[1:]):
                self._acyclic_graph += linear_algebra.exclusive_or(q0, q1)
            for q in self.discretes:
                self._acyclic_graph += linear_algebra.measure(q)
            # pylint: enable=invalid-name

        def _generate_param_resolver(self) -> None:
            self._param_resolvers = [linear_algebra.ParamResolver({"s": self.angle})]

        def _generate_symbols(self) -> None:
            self._symbols = [sympy.Symbol("s")]

    @property
    def provider_class(self) -> GraphProvider:
        return AbstractComplicatedGraphTest.GraphProvider


class AbstractSweepTest(AbstractTest):
    """Base class for sweeping tests."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties(GraphProperties):
        """Symplectic acyclic_graph properties.

        Properties:
            depth: Graph depth.
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            symbols: Number of symbols.
        """

        depth: int

    class GraphProvider(AbstractGraphProvider):
        """Provides symplectic acyclic_graph for testing sweep over different parameter
        values."""

        def __init__(
            self, properties: "AbstractSweepTest.GraphProperties"
        ) -> None:
            super().__init__(properties)

            # Override type annotation
            self._properties: AbstractSweepTest.GraphProperties = properties

        def _generate_acyclic_graph(self) -> None:
            self._acyclic_graph = linear_algebra.Graph()
            # pylint: disable=invalid-name
            if self._properties.symbols == 0:
                for _ in range(self._properties.depth):
                    for q in self.discretes:
                        self._acyclic_graph += linear_algebra.flip_x_axis(q)
            else:
                for d in range(self._properties.depth):
                    for q, s in zip(
                        self.discretes,
                        self.symbols[
                            d
                            * self._properties.discretes : (d + 1)
                            * self._properties.discretes
                        ],
                    ):
                        self._acyclic_graph += linear_algebra.flip_x_axis(q) ** s
            # pylint: enable=invalid-name
            self._acyclic_graph += [linear_algebra.measure(q) for q in self.discretes]

        def _generate_param_resolver(self) -> None:
            self._param_resolvers = [
                linear_algebra.ParamResolver(
                    {
                        f"s_{n}": random.random()
                        for n in range(self._properties.symbols)
                    }
                )
                for _ in range(self._properties.param_resolvers)
            ]

        def _generate_symbols(self) -> None:
            factor = (
                self._properties.depth * self._properties.discretes
                + self._properties.symbols
            ) // self._properties.symbols
            self._symbols = [
                sympy.Symbol(f"s_{n}") for n in range(self._properties.symbols)
            ]
            self._symbols *= factor

    @property
    def provider_class(self) -> "AbstractSweepTest.GraphProvider":
        return AbstractSweepTest.GraphProvider


class AbstractBatchTest(AbstractTest):
    """Base class for batch tests."""

    @dataclasses.dataclass(frozen=True)
    class GraphProperties(AbstractSweepTest.GraphProperties):
        """Symplectic acyclic_graph properties.

        Properties:
            acyclic_graphs: Number of acyclic_graphs.
            depth: Graph depth.
            param_resolvers: Number of input ParamResolver objects.
            discretes: Number of acyclic_graph discretes.
            symbols: Number of symbols.
        """

        acyclic_graphs: int

    class GraphProvider(AbstractSweepTest.GraphProvider):
        """Provides symplectic acyclic_graph for testing sweep over different parameter
        values."""

        def __init__(
            self, properties: "AbstractBatchTest.GraphProperties"
        ) -> None:
            super().__init__(properties)

            self._acyclic_graphs: Optional[List[linear_algebra.Graph]] = None

            # Override type annotation
            self._properties: AbstractBatchTest.GraphProperties = properties

        @property
        def acyclic_graphs(self) -> List[linear_algebra.Graph]:
            """Graphs to be simulated."""
            if self._acyclic_graphs:
                return self._acyclic_graphs

            self._generate_acyclic_graphs()
            return self._acyclic_graphs

        @property
        def param_resolvers(self) -> List[List[linear_algebra.ParamResolver]]:
            return super().param_resolvers

        def _generate_acyclic_graphs(self) -> None:
            self._acyclic_graphs = []
            for _ in range(self._properties.acyclic_graphs):
                self._generate_acyclic_graph()
                self._acyclic_graphs.append(self._acyclic_graph)

        def _generate_param_resolver(self) -> None:
            self._param_resolvers = [
                [
                    linear_algebra.ParamResolver(
                        {
                            f"s_{n}": random.random()
                            for n in range(self._properties.symbols)
                        }
                    )
                    for _ in range(self._properties.param_resolvers)
                ]
                for _ in range(self._properties.acyclic_graphs)
            ]

    @property
    def provider_class(self) -> "AbstractBatchTest.GraphProvider":
        return AbstractBatchTest.GraphProvider
