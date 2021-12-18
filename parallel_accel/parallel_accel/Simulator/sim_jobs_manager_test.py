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
# pylint: disable=protected-access

"""Tests sim_jobs_manager package"""
import importlib
import unittest
import unittest.mock
import uuid
import linear_algebra
import fakeredis

from parallel_accel.shared import schemas
from parallel_accel.shared.schemas.external import JobProgress
import sim_jobs_manager


class TestSimJobsManager(unittest.TestCase):
    """Tests SimJobsManager class behavior."""

    API_KEY = "test_api_key"
    JOB_ID = uuid.uuid4()

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.patchers = []

        patcher = unittest.mock.patch("redis.Redis", fakeredis.FakeStrictRedis)
        cls.patchers.append(patcher)

        for patcher in cls.patchers:
            patcher.start()

        importlib.reload(sim_jobs_manager)

        cls.manager = sim_jobs_manager.SimJobsManager()

    @classmethod
    def tearDownClass(cls) -> None:
        """See base class documentation."""
        for patcher in cls.patchers:
            patcher.stop()

    def setUp(self) -> None:
        """See base class documentation."""
        context = schemas.SampleJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
            repetitions=1,
        )

        obj = schemas.SampleJob(
            api_key=self.API_KEY,
            context=context,
            id=self.JOB_ID,
            type=schemas.JobType.SAMPLE,
        )
        obj_raw = schemas.encode(schemas.SampleJobSchema, obj)
        self.manager._connections[sim_jobs_manager.RedisInstances.JOBS].set(
            sim_jobs_manager.SimJobsManager.KeyType.CONTEXT.key_for(
                str(self.JOB_ID)
            ),
            obj_raw,
        )

        self.manager._connections[
            sim_jobs_manager.RedisInstances.JOBS_IDS
        ].lpush(self.API_KEY, str(self.JOB_ID))

        obj = schemas.JobResult(self.JOB_ID, schemas.JobStatus.NOT_STARTED)
        obj_raw = schemas.encode(schemas.JobResultSchema, obj)
        self.manager._connections[sim_jobs_manager.RedisInstances.JOBS].set(
            sim_jobs_manager.SimJobsManager.KeyType.STATUS.key_for(
                str(self.JOB_ID)
            ),
            obj_raw,
        )

    def test_get_next_job_expectation(self) -> None:
        """Tests get_next_job method behavior: EXPECTATION job type."""
        discretes = linear_algebra.LinearSpace.range(2)
        context = schemas.ExpectationJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
            operators=[
                linear_algebra.flip_x_axis(discretes[0]) + linear_algebra.flip_y_axis(discretes[1]),
                linear_algebra.flip_y_axis(discretes[0]) + linear_algebra.flip_x_axis(discretes[1]),
            ],
        )
        obj = schemas.ExpectationJob(
            api_key=self.API_KEY,
            context=context,
            id=self.JOB_ID,
            type=schemas.JobType.EXPECTATION,
        )
        obj_raw = schemas.encode(schemas.ExpectationJobSchema, obj)
        self.manager._connections[sim_jobs_manager.RedisInstances.JOBS].set(
            sim_jobs_manager.SimJobsManager.KeyType.CONTEXT.key_for(
                str(self.JOB_ID)
            ),
            obj_raw,
        )

        # Run test
        actual_context = self.manager.get_next_job(self.API_KEY)

        # Verification
        self.assertEqual(actual_context, obj)

    def test_get_next_job_sample(self) -> None:
        """Tests get_next_job method behavior: SAMPLE job type."""
        # Test setup
        discrete = linear_algebra.GridSpace(0, 0)
        context = schemas.SampleJobContext(
            acyclic_graph=linear_algebra.Graph(
                linear_algebra.flip_x_axis(discrete) ** 0.5,  # sqrt of NOT
                linear_algebra.measure(discrete, key="m"),  # observation building_block
            ),
            param_resolver=linear_algebra.ParamResolver(None),
            repetitions=1,
        )
        obj = schemas.SampleJob(
            api_key=self.API_KEY,
            context=context,
            id=self.JOB_ID,
            type=schemas.JobType.SAMPLE,
        )
        obj_raw = schemas.encode(schemas.SampleJobSchema, obj)
        self.manager._connections[sim_jobs_manager.RedisInstances.JOBS].set(
            sim_jobs_manager.SimJobsManager.KeyType.CONTEXT.key_for(
                str(self.JOB_ID)
            ),
            obj_raw,
        )

        # Run test
        actual_context = self.manager.get_next_job(self.API_KEY)

        # Verification
        self.assertEqual(actual_context, obj)

    def test_get_and_clear_job(self) -> None:
        """Tests get_next_job and clear_next_job methods behavior."""
        # Check queue is not empty
        result_raw = self.manager._connections[
            sim_jobs_manager.RedisInstances.JOBS_IDS
        ].lrange(str(self.API_KEY), 0, 1)
        self.assertEqual(result_raw, [str(self.JOB_ID)])

        # Run test
        self.manager.get_next_job(self.API_KEY)
        self.manager.clear_next_job(self.API_KEY)

        # Check queue and JOBS are empty
        result_raw = self.manager._connections[
            sim_jobs_manager.RedisInstances.JOBS
        ].get(
            sim_jobs_manager.SimJobsManager.KeyType.STATUS.key_for(
                str(self.JOB_ID)
            )
        )
        self.assertIsNone(result_raw)

        result_raw = self.manager._connections[
            sim_jobs_manager.RedisInstances.JOBS_IDS
        ].get(str(self.API_KEY))
        self.assertIsNone(result_raw)

    def test_set_job_complete(self) -> None:
        """Tests set_job_complete method behavior."""
        discretes = linear_algebra.LinearSpace.range(2)
        context = schemas.ExpectationJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
            operators=[
                linear_algebra.flip_x_axis(discretes[0]) + linear_algebra.flip_y_axis(discretes[1]),
                linear_algebra.flip_y_axis(discretes[0]) + linear_algebra.flip_x_axis(discretes[1]),
            ],
        )
        obj = schemas.ExpectationJob(
            api_key=self.API_KEY,
            context=context,
            id=self.JOB_ID,
            type=schemas.JobType.EXPECTATION,
        )
        obj_raw = schemas.encode(schemas.ExpectationJobSchema, obj)
        self.manager._connections[sim_jobs_manager.RedisInstances.JOBS].set(
            sim_jobs_manager.SimJobsManager.KeyType.CONTEXT.key_for(
                str(self.JOB_ID)
            ),
            obj_raw,
        )

        # Run test
        result = [0.1, 0.2]
        self.manager.set_job_complete(
            self.JOB_ID, result, schemas.ExpectationJobResultSchema
        )

        # Verification
        expected_result = schemas.JobResult(
            id=self.JOB_ID, status=schemas.JobStatus.COMPLETE, result=result
        )
        self._verify_result(expected_result)

    def test_set_job_error(self) -> None:
        """Tests set_job_error method behavior."""
        # Run test
        message = "Job failed"
        self.manager.set_job_error(self.JOB_ID, message)

        # Verification
        expected_result = schemas.JobResult(
            error_message=message,
            id=self.JOB_ID,
            status=schemas.JobStatus.ERROR,
        )
        self._verify_result(expected_result)

    def test_set_job_in_progress(self) -> None:
        """Tests set_job_in_progress method behavior."""
        # Run test
        self.manager.set_job_in_progress(self.JOB_ID)

        # Verification
        expected_result = schemas.JobResult(
            id=self.JOB_ID,
            status=schemas.JobStatus.IN_PROGRESS,
            progress=schemas.JobProgress(),
        )
        self._verify_result(expected_result)

    def test_set_job_in_progress_result_not_exist(self) -> None:
        """Tests set_job_in_progress method behavior: job result not exist."""
        self.manager._connections[sim_jobs_manager.RedisInstances.JOBS].delete(
            sim_jobs_manager.SimJobsManager.KeyType.STATUS.key_for(
                str(self.JOB_ID)
            )
        )

        # Run test
        self.manager.get_next_job(self.API_KEY)
        self.assertRaises(
            sim_jobs_manager.JobResultNotFoundError,
            self.manager.set_job_in_progress,
            self.JOB_ID,
        )

    def test_update_job_progress(self) -> None:
        """Tests job progress update."""
        self.manager.set_job_in_progress(self.JOB_ID)

        # Verification
        expected_result = schemas.JobResult(
            id=self.JOB_ID,
            status=schemas.JobStatus.IN_PROGRESS,
            progress=schemas.JobProgress(0, 1),
        )
        self._verify_result(expected_result)

        self.manager.update_job_progress(self.JOB_ID, JobProgress(3, 10))
        expected_result = schemas.JobResult(
            id=self.JOB_ID,
            status=schemas.JobStatus.IN_PROGRESS,
            progress=schemas.JobProgress(3, 10),
        )
        self._verify_result(expected_result)

    def _verify_result(self, expected_result: schemas.JobResult) -> None:
        result_raw = self.manager._connections[
            sim_jobs_manager.RedisInstances.JOBS
        ].get(
            sim_jobs_manager.SimJobsManager.KeyType.STATUS.key_for(
                str(self.JOB_ID)
            )
        )
        actual_result = schemas.decode(schemas.JobResultSchema, result_raw)
        self.assertEqual(actual_result, expected_result)

        result_raw = self.manager._connections[
            sim_jobs_manager.RedisInstances.JOBS
        ].lpop(
            sim_jobs_manager.SimJobsManager.KeyType.QUEUE.key_for(
                str(self.JOB_ID)
            )
        )
        actual_result = schemas.decode(schemas.JobResultSchema, result_raw)
        self.assertEqual(actual_result, expected_result)
