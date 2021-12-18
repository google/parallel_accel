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
# pylint: disable=protected-access, invalid-name

"""Unit test for asic_sim_runner module"""
import importlib
import signal
from typing import Optional
import unittest
import unittest.mock
import uuid
import linear_algebra
import redis

from parallel_accel.shared import schemas
from parallel_accel.shared.redis import workers

import sim_jobs_manager
from asic_la import asic_simulator
import asic_sim_runner


class TestASICSimRunner(unittest.TestCase):
    """Tests ASICSimRunner behavior."""

    API_KEY = "api_key"
    JOB_ID = uuid.uuid4()

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.patchers = []

        cls.mocked_signal = unittest.mock.Mock(signal.signal)
        patcher = unittest.mock.patch("signal.signal", cls.mocked_signal)
        cls.patchers.append(patcher)

        for patcher in cls.patchers:
            patcher.start()

        importlib.reload(asic_sim_runner)

        cls.mocked_sim_jobs_manager = unittest.mock.Mock(
            spec=sim_jobs_manager.SimJobsManager
        )
        cls.mocked_worker_manager = unittest.mock.Mock(
            spec=workers.WorkersRedisStore
        )
        cls.mocked_simulator = unittest.mock.Mock(asic_simulator.ASICSimulator)

    @classmethod
    def tearDownClass(cls) -> None:
        """See base class documentation."""
        for patcher in cls.patchers:
            patcher.stop()

    def setUp(self) -> None:
        """See base class documentation."""
        self.runner = asic_sim_runner.ASICSimRunner(
            self.API_KEY,
            self.mocked_sim_jobs_manager,
            self.mocked_worker_manager,
            self.mocked_simulator,
        )

    def tearDown(self) -> None:
        """See base class documentation."""
        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

        self.mocked_simulator.compute_samples.side_effect = None

    def test_init(self) -> None:
        """Tests __init__ method behavior."""
        # Verification
        call_args_list = [
            ((x, self.runner.signal_handler),)
            for x in (signal.SIGINT, signal.SIGTERM)
        ]
        self.assertEqual(self.mocked_signal.call_args_list, call_args_list)

    def test_signal_handler(self) -> None:
        """Tests signal_handler method behavior."""
        # Run test
        self.runner.signal_handler(signal.SIGINT, None)

        # Verification
        self.mocked_worker_manager.set_shutting_down.assert_called_once_with(
            self.API_KEY
        )

    def test_run_no_job(self) -> None:
        """Tests run method behavior: next job is None"""
        # Test setup
        self._set_get_next_job()

        # Run test
        self.runner.run()

        # Verification
        self.mocked_worker_manager.set_idle.assert_called_once_with(
            self.API_KEY
        )
        self.mocked_sim_jobs_manager.get_next_job.assert_called_once_with(
            self.API_KEY
        )

    def test_run_get_job_exception(self) -> None:
        """Tests run method behavior: get_next_job raised exception"""
        # Test setup
        self.mocked_sim_jobs_manager.get_next_job.side_effect = (
            redis.ConnectionError
        )

        # Run test
        self.runner.run()

        # Verification
        self.mocked_worker_manager.set_idle.assert_called_once_with(
            self.API_KEY
        )
        self.mocked_worker_manager.set_error.assert_called_once_with(
            self.API_KEY, "Internal Simulator Error"
        )
        self.mocked_sim_jobs_manager.get_next_job.assert_called_once_with(
            self.API_KEY
        )

    def test_run_sample_job(self) -> None:
        """Tests run method behavior: get_next_job returned Sample job type"""
        # Test setup
        job = self._get_sample_job(21)
        self._set_get_next_job(job)

        result = linear_algebra.Result(
            params=linear_algebra.ParamResolver(None), observations={"test": []}
        )
        self.mocked_simulator.compute_samples.return_value = result

        # Run test
        self.runner.run()

        # Verification
        call_args_list = [((self.API_KEY,),)] * 2
        self.assertEqual(
            self.mocked_worker_manager.set_idle.call_args_list, call_args_list
        )
        self.mocked_worker_manager.set_processing_job.assert_called_once_with(
            self.API_KEY, self.JOB_ID
        )

        self.mocked_sim_jobs_manager.get_next_job.assert_called_once_with(
            self.API_KEY
        )
        self.mocked_sim_jobs_manager.clear_next_job.assert_called_once_with(
            self.API_KEY
        )
        self.mocked_sim_jobs_manager.set_job_in_progress.assert_called_once_with(
            job.id
        )
        self.mocked_sim_jobs_manager.set_job_complete.assert_called_once_with(
            job.id, result, schemas.SampleJobResultSchema
        )

        self.mocked_simulator.compute_samples.assert_called_once_with(
            job.context.acyclic_graph,
            job.context.param_resolver,
            job.context.repetitions,
            unittest.mock.ANY,
        )

    def test_run_sample_sweep_job(self) -> None:
        """Tests run method behavior: get_next_job returned Sample sweep job
        type"""
        job = schemas.SampleJob(
            id=self.JOB_ID,
            api_key=self.API_KEY,
            type=schemas.JobType.SAMPLE,
            context=schemas.SampleSweepJobContext(
                acyclic_graph=self._get_acyclic_graph(10),
                params=[linear_algebra.ParamResolver(None)],
                repetitions=1,
            ),
        )
        self._set_get_next_job(job)
        self.runner.run()

        self.mocked_simulator.compute_samples_sweep.assert_called_once_with(
            job.context.acyclic_graph,
            job.context.params,
            job.context.repetitions,
            unittest.mock.ANY,
        )

    def test_run_sample_batch_job(self) -> None:
        """Tests run method behavior: get_next_job returned Sample batch job
        type"""
        job = schemas.SampleJob(
            id=self.JOB_ID,
            api_key=self.API_KEY,
            type=schemas.JobType.SAMPLE,
            context=schemas.SampleBatchJobContext(
                acyclic_graphs=[self._get_acyclic_graph(10)],
                params=[[linear_algebra.ParamResolver(None)]],
                repetitions=1,
            ),
        )
        self._set_get_next_job(job)
        self.runner.run()

        self.mocked_simulator.compute_samples_batch.assert_called_once_with(
            job.context.acyclic_graphs,
            job.context.params,
            job.context.repetitions,
            unittest.mock.ANY,
        )

    def test_run_sample_job_failed(self) -> None:
        """Tests run method behavior: get_next_job returned Sample job type,
        simulation failed"""
        # Test setup
        job = self._get_sample_job(21)
        self._set_get_next_job(job)

        self.mocked_simulator.compute_samples.side_effect = Exception

        # Run test
        self.runner.run()

        # Verification
        call_args_list = [((self.API_KEY,),)] * 2
        self.assertEqual(
            self.mocked_worker_manager.set_idle.call_args_list, call_args_list
        )
        self.mocked_worker_manager.set_processing_job.assert_called_once_with(
            self.API_KEY, self.JOB_ID
        )

        self.mocked_sim_jobs_manager.get_next_job.assert_called_once_with(
            self.API_KEY
        )
        self.mocked_sim_jobs_manager.clear_next_job.assert_called_once_with(
            self.API_KEY
        )
        self.mocked_sim_jobs_manager.set_job_in_progress.assert_called_once_with(
            job.id
        )
        self.mocked_sim_jobs_manager.set_job_error.assert_called_once_with(
            job.id, "Simulator internal error occured"
        )

        self.mocked_simulator.compute_samples.assert_called_once_with(
            job.context.acyclic_graph,
            job.context.param_resolver,
            job.context.repetitions,
            unittest.mock.ANY,
        )

    def _get_acyclic_graph(self, num_discretes: int) -> linear_algebra.Graph:
        """Get a simple test acyclic_graph"""
        discretes = linear_algebra.LinearSpace.range(num_discretes)
        acyclic_graph = linear_algebra.Graph()
        for q in discretes:
            acyclic_graph += linear_algebra.I(q)
        acyclic_graph += linear_algebra.Graph(
            linear_algebra.flip_x_axis(discretes[0]),
            linear_algebra.flip_x_axis(discretes[1]),
            linear_algebra.flip_x_axis(discretes[2]),
            linear_algebra.measure(discretes[0], key="m1"),
            linear_algebra.measure(discretes[1], key="m2"),
            linear_algebra.measure(discretes[2], key="m3"),
        )
        return acyclic_graph

    def _get_sample_job(self, num_discretes: int) -> schemas.SampleJob:
        """Gets sample job.

        Args:
            num_discretes: Number of discretes in the acyclic_graph.

        Returns:
            Job object.
        """

        acyclic_graph = self._get_acyclic_graph(num_discretes)
        return schemas.SampleJob(
            api_key=self.API_KEY,
            id=self.JOB_ID,
            type=schemas.JobType.SAMPLE,
            context=schemas.SampleJobContext(
                acyclic_graph=acyclic_graph,
                param_resolver=linear_algebra.ParamResolver(None),
                repetitions=1,
            ),
        )

    def _set_get_next_job(self, retval: Optional[schemas.Job] = None) -> None:
        """Configures mocked_simulator.get_next_job side effect.

        Args:
            retval: Optional value to be returned by get_next_job mocked method.
        """

        def side_effect(api_key: str) -> Optional[schemas.Job]:
            self.assertEqual(api_key, self.API_KEY)
            self.runner._running = False
            return retval

        self.mocked_sim_jobs_manager.get_next_job.side_effect = side_effect
