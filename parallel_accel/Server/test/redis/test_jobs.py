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
# pylint: disable=too-many-public-methods, wrong-import-order

"""Unit test for manager module"""
import datetime
import importlib
import os
import secrets
import time
import unittest
import unittest.mock
import uuid
import aiounittest
import linear_algebra
from parallel_accel.shared import schemas
import redis

from src.redis import jobs


class TestJobsManager(aiounittest.AsyncTestCase):
    """Tests JobsManager class behavior."""

    API_KEY = secrets.token_hex(16)
    DEADLINE = datetime.timedelta(seconds=100)
    JOB_ID = uuid.uuid4()

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.patchers = []

        cls.mocked_time = unittest.mock.Mock(spec=time.time)
        cls.mocked_time.return_value = 0
        patcher = unittest.mock.patch("time.time", cls.mocked_time)
        cls.patchers.append(patcher)

        cls.mocked_redis = unittest.mock.Mock(spec=redis.Redis)
        cls.mocked_redis.return_value = cls.mocked_redis
        patcher = unittest.mock.patch("redis.Redis", cls.mocked_redis)
        cls.patchers.append(patcher)

        for patcher in cls.patchers:
            patcher.start()

        importlib.reload(jobs)

        os.environ["REDISHOST"] = "localhost"

        cls.manager = jobs.JobsManager()

    @classmethod
    def tearDownClass(cls) -> None:
        """See base class documentation."""
        del os.environ["REDISHOST"]

        for patcher in cls.patchers:
            patcher.stop()

    def tearDown(self) -> None:
        """See base class documentation."""
        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

    def test_create_job(self) -> None:
        """Tests create_job method behavior."""
        # Test setup
        data = schemas.SampleJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
        )
        context = schemas.encode(schemas.SampleJobContextSchema, data)
        job_type = schemas.JobType.SAMPLE

        self.mocked_redis.set.return_value = True
        self.mocked_redis.rpush.return_value = True

        # Run test
        job_uuid = self.manager.create_job(self.API_KEY, job_type, context)

        # Verification

        ## Verify Redis SET calls
        data = schemas.Job(
            api_key=self.API_KEY, context=context, id=job_uuid, type=job_type
        )
        serialized_data = schemas.encode(schemas.JobSchema, data)
        expected_call_args = [
            (
                (
                    jobs.JobsManager.KeyType.CONTEXT.key_for(job_uuid),
                    serialized_data,
                    datetime.timedelta(days=7),
                ),
            ),
        ]

        data = schemas.JobResult(job_uuid, schemas.JobStatus.NOT_STARTED)
        serialized_data = schemas.encode(schemas.JobResultSchema, data)
        expected_call_args.append(
            (
                (
                    jobs.JobsManager.KeyType.STATUS.key_for(job_uuid),
                    serialized_data,
                    datetime.timedelta(days=7),
                ),
            )
        )
        self.assertEqual(
            self.mocked_redis.set.call_args_list, expected_call_args
        )

        ## Verify Redis RPUSH calls
        expected_call_args = [
            (
                (
                    jobs.JobsManager.KeyType.QUEUE.key_for(job_uuid),
                    serialized_data,
                ),
            ),
            (
                (
                    self.API_KEY,
                    job_uuid,
                ),
            ),
        ]
        self.assertEqual(
            self.mocked_redis.rpush.call_args_list, expected_call_args
        )

        ## Verify Redis EXPIRE calls
        args = [jobs.JobsManager.KeyType.QUEUE.key_for(job_uuid), self.API_KEY]
        expected_call_args = [((x, datetime.timedelta(days=7)),) for x in args]
        self.assertEqual(
            self.mocked_redis.expire.call_args_list, expected_call_args
        )

    def test_create_job_failed_to_create_key(self) -> None:
        """Tests create_job method behavior: failed to create job context key"""
        # Test setup
        data = schemas.SampleJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
        )
        context = schemas.encode(schemas.SampleJobContextSchema, data)
        job_type = schemas.JobType.SAMPLE

        self.mocked_redis.set.return_value = False

        # Run test
        with self.assertRaises(jobs.SetJobContextError):
            self.manager.create_job(self.API_KEY, job_type, context)

    def test_flush_job_queue(self) -> None:
        """Tests flush_job_queue method behavior."""
        # Test setup
        self.mocked_redis.exists.return_value = 1
        self.mocked_redis.llen.return_value = 1
        self.mocked_redis.lrange.return_value = [str(self.JOB_ID)]

        # Run test
        self.manager.flush_job_queue(self.API_KEY)

        # Verification
        self.mocked_redis.exists.assert_called_once_with(self.API_KEY)
        self.mocked_redis.llen.assert_called_once_with(self.API_KEY)
        self.mocked_redis.lrange.assert_called_once_with(self.API_KEY, 0, 1)

        call_args_list = [((self.API_KEY,),)] + [
            ((x.key_for(str(self.JOB_ID)),),) for x in jobs.JobsManager.KeyType
        ]
        self.assertEqual(
            self.mocked_redis.delete.call_args_list, call_args_list
        )

    def test_flush_job_queue_not_found(self) -> None:
        """Tests flush_job_queue method behavior: queue not found."""
        # Test setup
        self.mocked_redis.exists.return_value = 0

        # Run test
        self.manager.flush_job_queue(self.API_KEY)

        # Verification
        self.mocked_redis.exists.assert_called_once_with(self.API_KEY)
        for command in ("delete", "llen", "lrange"):
            getattr(self.mocked_redis, command).assert_not_called()

    def test_get_job_queue(self) -> None:
        """Tests get_job_queue method behavior."""
        # Test setup
        self.mocked_redis.exists.return_value = 1
        self.mocked_redis.llen.return_value = 1
        self.mocked_redis.lrange.return_value = [str(self.JOB_ID)]

        # Run test
        queue = self.manager.get_job_queue(self.API_KEY)

        # Verification
        self.assertEqual(queue.ids, [str(self.JOB_ID)])
        self.mocked_redis.exists.assert_called_once_with(self.API_KEY)
        self.mocked_redis.llen.assert_called_once_with(self.API_KEY)
        self.mocked_redis.lrange.assert_called_once_with(self.API_KEY, 0, 1)

    def test_get_job_queue_not_exists(self) -> None:
        """Tests get_job_queue method behavior."""
        # Test setup
        self.mocked_redis.exists.return_value = 0

        # Run test
        queue = self.manager.get_job_queue(self.API_KEY)

        # Verification
        self.assertEqual(queue.ids, [])
        self.mocked_redis.exists.assert_called_once_with(self.API_KEY)
        for command in ("llen", "lrange"):
            getattr(self.mocked_redis, command).assert_not_called()

    async def test_subscribe_job_status(self) -> None:
        """Tests subscribe_job_status method behavior."""
        # Test setup
        job_id = str(self.JOB_ID)
        job_results = [
            schemas.JobResult(
                id=self.JOB_ID,
                status=schemas.JobStatus.IN_PROGRESS,
                progress=schemas.JobProgress(),
            ),
            schemas.JobResult(
                self.JOB_ID, schemas.JobStatus.ERROR, error_message="error"
            ),
        ]

        serialized_job_results = [
            schemas.encode(schemas.JobResultSchema, x) for x in job_results
        ]
        self.mocked_redis.blpop.side_effect = [
            (job_id, x) for x in serialized_job_results
        ]
        self.mocked_redis.exists.return_value = 1

        # Run test
        async for status in self.manager.subscribe_job_status(
            job_id, self.DEADLINE
        ):
            self.assertEqual(status, job_results.pop(0))

        # Verification
        self.mocked_redis.exists.assert_called_once_with(
            jobs.JobsManager.KeyType.CONTEXT.key_for(job_id)
        )
        self.assertEqual(
            self.mocked_redis.blpop.call_args_list,
            [
                (
                    (jobs.JobsManager.KeyType.QUEUE.key_for(job_id),),
                    {
                        "timeout": (
                            self.DEADLINE
                            - datetime.timedelta(
                                seconds=self.mocked_time.return_value
                            )
                        ).total_seconds()
                    },
                )
            ]
            * 2,
        )
        self.mocked_redis.delete.assert_called_once_with(
            jobs.JobsManager.KeyType.CONTEXT.key_for(job_id)
        )

    async def test_subscribe_job_status_context_not_found(self) -> None:
        """Tests subscribe_job_status method behavior: job not found."""
        # Test setup
        job_id = str(self.JOB_ID)

        self.mocked_redis.exists.return_value = 0

        # Run test
        with self.assertRaises(jobs.shared_redis.JobNotFoundError):
            async for _ in self.manager.subscribe_job_status(
                job_id, self.DEADLINE
            ):
                pass

        # Verification
        self.mocked_redis.exists.assert_called_once_with(
            jobs.JobsManager.KeyType.CONTEXT.key_for(job_id)
        )
        self.assertFalse(self.mocked_redis.blpop.called)
        self.assertFalse(self.mocked_redis.delete.called)

    async def test_subscribe_job_status_timeout(self) -> None:
        """Tests subscribe_job_status method behavior: BLPOP timed out."""
        # Test setup
        job_id = str(self.JOB_ID)

        self.mocked_redis.exists.return_value = 1
        self.mocked_redis.blpop.side_effect = [None]

        # Run test
        async for status in self.manager.subscribe_job_status(
            job_id, self.DEADLINE
        ):
            self.assertIsNone(status)

        # Verification
        self.mocked_redis.exists.assert_called_once_with(
            jobs.JobsManager.KeyType.CONTEXT.key_for(job_id)
        )
        self.mocked_redis.blpop.called_once_with(
            jobs.JobsManager.KeyType.QUEUE.key_for(job_id),
            timeout=(
                self.DEADLINE
                - datetime.timedelta(seconds=self.mocked_time.return_value)
            ).total_seconds(),
        )
        self.assertFalse(self.mocked_redis.delete.called)

    def test_get_job_status(self) -> None:
        """Tests get_job_status method behavior"""
        # Test setup
        job_id = str(self.JOB_ID)

        self.mocked_redis.exists.return_value = 1

        result = schemas.JobResult(
            id=self.JOB_ID, status=schemas.JobStatus.NOT_STARTED
        )
        serialized = schemas.encode(schemas.JobResultSchema, result)
        self.mocked_redis.get.return_value = serialized

        # Run test
        job_status = self.manager.get_job_status(job_id)

        # Verification
        self.assertEqual(job_status, result.status)
        self.mocked_redis.get.assert_called_once_with(
            jobs.JobsManager.KeyType.STATUS.key_for(job_id)
        )

    def test_get_job_status_not_found(self) -> None:
        """Tests get_job_status method behavior: job not found."""
        # Test setup
        job_id = str(self.JOB_ID)

        self.mocked_redis.exists.return_value = 0

        # Run test
        with self.assertRaises(jobs.shared_redis.JobResultNotFoundError):
            self.manager.get_job_status(job_id)

        # Verification
        self.mocked_redis.get.assert_not_called()

    def test_get_job_type(self) -> None:
        """Tests get_job_type method behavior"""
        # Test setup
        job_id = str(self.JOB_ID)

        self.mocked_redis.exists.return_value = 1

        context = schemas.SampleJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
        )
        job = schemas.Job(
            api_key=self.API_KEY,
            context=context,
            id=self.JOB_ID,
            type=schemas.JobType.SAMPLE,
        )
        serialized = schemas.encode(schemas.SampleJobSchema, job)
        self.mocked_redis.get.return_value = serialized

        # Run test
        job_type = self.manager.get_job_type(job_id)

        # Verification
        self.assertEqual(job_type, job.type)
        self.mocked_redis.exists.assert_called_once_with(
            jobs.JobsManager.KeyType.CONTEXT.key_for(job_id)
        )
        self.mocked_redis.get.assert_called_once_with(
            jobs.JobsManager.KeyType.CONTEXT.key_for(job_id)
        )

    def test_get_job_type_not_found(self) -> None:
        """Tests get_job_type method behavior: job not found."""
        # Test setup
        job_id = str(self.JOB_ID)

        self.mocked_redis.exists.return_value = 0

        # Run test
        with self.assertRaises(jobs.shared_redis.JobNotFoundError):
            self.manager.get_job_type(job_id)

        # Verification
        self.mocked_redis.exists.assert_called_once_with(
            jobs.JobsManager.KeyType.CONTEXT.key_for(job_id)
        )

    def test_has_job_queue(self) -> None:
        """Tests has_job_queue method behavior"""
        # Test setup
        self.mocked_redis.exists.return_value = 1

        # Run test
        result = self.manager.has_jobs_queue(self.API_KEY)

        # Verification
        self.assertTrue(result)
        self.mocked_redis.exists.assert_called_once_with(self.API_KEY)

    def test_has_pending_job(self) -> None:
        """Tests has_pending_job method behavior"""
        # Test setup
        self.mocked_redis.exists.return_value = 1
        self.mocked_redis.llen.return_value = 1
        self.mocked_redis.lrange.return_value = [str(self.JOB_ID)]

        # Run test
        result = self.manager.has_pending_job(self.API_KEY, self.JOB_ID)

        # Verification
        self.assertTrue(result)
        call_args_list = [((self.API_KEY,),)] * 2
        self.assertEqual(
            self.mocked_redis.exists.call_args_list, call_args_list
        )
        self.mocked_redis.llen.assert_called_once_with(self.API_KEY)
        self.mocked_redis.lrange.assert_called_once_with(self.API_KEY, 0, 1)

    def test_has_pending_job_no_queue(self) -> None:
        """Tests has_pending_job method behavior: no job queue"""
        # Test setup
        self.mocked_redis.exists.return_value = 0

        # Run test
        result = self.manager.has_pending_job(self.API_KEY, self.JOB_ID)

        # Verification
        self.assertFalse(result)
        self.mocked_redis.exists.assert_called_once_with(self.API_KEY)
        for command in ("llen", "lrange"):
            getattr(self.mocked_redis, command).assert_not_called()

    def test_is_same_api_key(self) -> None:
        """Tests is_same_api_key method behavior."""
        # Test setup
        self.mocked_redis.exists.return_value = 1

        context = schemas.SampleJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
        )
        job = schemas.Job(
            api_key=self.API_KEY,
            context=context,
            id=self.JOB_ID,
            type=schemas.JobType.SAMPLE,
        )
        serialized = schemas.encode(schemas.SampleJobSchema, job)
        self.mocked_redis.get.return_value = serialized

        # Run test
        self.assertTrue(self.manager.is_same_api_key(self.JOB_ID, self.API_KEY))
        self.assertFalse(
            self.manager.is_same_api_key(
                self.JOB_ID, self.API_KEY + self.API_KEY
            )
        )

    def test_is_same_job_type(self) -> None:
        """Tests is_same_job_type method behavior."""
        # Test setup
        self.mocked_redis.exists.return_value = 1

        context = schemas.SampleJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
        )
        job = schemas.Job(
            api_key=self.API_KEY,
            context=context,
            id=self.JOB_ID,
            type=schemas.JobType.SAMPLE,
        )
        serialized = schemas.encode(schemas.SampleJobSchema, job)
        self.mocked_redis.get.return_value = serialized

        # Run test
        self.assertTrue(
            self.manager.is_same_job_type(self.JOB_ID, schemas.JobType.SAMPLE)
        )
        self.assertFalse(
            self.manager.is_same_job_type(
                self.JOB_ID, schemas.JobType.EXPECTATION
            )
        )
