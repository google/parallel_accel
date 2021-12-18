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

"""Unit test for workers module"""
import importlib
import os
import secrets
import time
import unittest
import unittest.mock
import uuid
import redis

from parallel_accel.shared import schemas
from parallel_accel.shared.redis import base, workers


class TestWorkersRedisStore(unittest.TestCase):
    """Tests WorkersRedisStore class behavior."""

    API_KEY = secrets.token_hex(16)
    JOB_ID = uuid.uuid4()

    WORKER = schemas.WorkerInternal(schemas.WorkerState.OFFLINE)

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.patchers = []

        cls.mocked_time = unittest.mock.Mock(spec=time.time)
        patcher = unittest.mock.patch("time.time", cls.mocked_time)
        cls.patchers.append(patcher)

        cls.mocked_redis = unittest.mock.Mock(spec=redis.Redis)
        cls.mocked_redis.return_value = cls.mocked_redis
        patcher = unittest.mock.patch("redis.Redis", cls.mocked_redis)
        cls.patchers.append(patcher)

        for patcher in cls.patchers:
            patcher.start()

        importlib.reload(base)
        importlib.reload(workers)

        os.environ["REDISHOST"] = "localhost"

        cls.store = workers.WorkersRedisStore()
        cls.connection = cls.store._connections[workers.RedisInstances.WORKERS]

    @classmethod
    def tearDownClass(cls) -> None:
        """See base class documentation."""
        del os.environ["REDISHOST"]

        for patcher in cls.patchers:
            patcher.stop()

    def setUp(self) -> None:
        """See base class documentation."""
        self.connection.exists.return_value = 1

        serialized = schemas.encode(schemas.WorkerSchema, self.WORKER)
        self.connection.get.return_value = serialized

    def tearDown(self) -> None:
        """See base class documentation."""
        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

        self.connection.exists.side_effect = None

    def test_get_worker(self) -> None:
        """Tests get_worker method: worker exists."""
        # Run test
        worker = self.store.get_worker(self.API_KEY)

        # Verifciation
        self.assertEqual(worker, self.WORKER)

        self._verify_redis_exists_call()

        result = self.connection.get.called_once_with(self.API_KEY)
        self.assertTrue(result)

    def test_get_worker_not_exists(self) -> None:
        """Tests get_worker method: worker does not exist."""
        # Set up
        self.connection.exists.return_value = 0

        # Run test
        with self.assertRaises(workers.WorkerNotFoundError):
            self.store.get_worker(self.API_KEY)

        # Verification
        self._verify_redis_exists_call()

    def test_has_worker(self) -> None:
        """Tests has_worker method."""
        # Run test
        result = self.store.has_worker(self.API_KEY)
        self.assertTrue(result)

        # Verifciation
        self._verify_redis_exists_call()

    def test_set_offline(self) -> None:
        """Tests set_offline method."""
        # Run test
        self.store.set_offline(self.API_KEY)

        # Verification
        data = schemas.WorkerInternal(state=schemas.WorkerState.OFFLINE)
        self._verify_redis_exists_call()
        self._verify_redis_set_call(data)

    def test_set_offline_same_state(self) -> None:
        """Tests set_offline method: worker already in OFFLINE state"""
        # Set up
        worker = schemas.WorkerInternal(state=schemas.WorkerState.OFFLINE)
        serialized = schemas.encode(schemas.WorkerSchema, worker)
        self.connection.get.return_value = serialized

        # Run test
        self.store.set_offline(self.API_KEY)

        # Verification
        self._verify_redis_exists_call()
        self.mocked_redis.set.assert_not_called()

    def test_set_offline_not_exist(self) -> None:
        """Tests set_offline method: worker does not exist"""
        # Set up
        self.connection.exists.side_effect = [0, 1]

        # Run test
        self.store.set_offline(self.API_KEY)

        # Verification
        data = schemas.WorkerInternal(state=schemas.WorkerState.OFFLINE)
        self._verify_redis_exists_call()
        self._verify_redis_set_call(data)

    def test_set_booting(self) -> None:
        """Tests set_booting method."""
        # Run test
        self.store.set_booting(self.API_KEY)

        # Verification
        data = schemas.WorkerInternal(state=schemas.WorkerState.BOOTING)
        self._verify_redis_exists_call()
        self._verify_redis_set_call(data)

    def test_set_idle(self) -> None:
        """Tests set_idle method."""
        # Run test
        self.store.set_idle(self.API_KEY)

        # Verification
        data = schemas.WorkerInternal(state=schemas.WorkerState.IDLE)
        self._verify_redis_exists_call()
        self._verify_redis_set_call(data)

    def test_set_processing_job(self) -> None:
        """Tests set_processing_job method."""
        job_id = uuid.uuid4()
        now = 1234567890

        self.mocked_time.return_value = now

        # Run test
        self.store.set_processing_job(self.API_KEY, job_id)

        # Verification
        data = schemas.WorkerInternal(
            state=schemas.WorkerState.PROCESSING_JOB,
            job_id=job_id,
            job_timestamp=now,
        )
        self._verify_redis_exists_call()
        self._verify_redis_set_call(data)

    def test_set_error(self) -> None:
        """Tests set_error method."""
        error = "Some error"

        # Run test
        self.store.set_error(self.API_KEY, error)

        # Verification
        data = schemas.WorkerInternal(
            state=schemas.WorkerState.ERROR, error=error
        )
        self._verify_redis_exists_call()
        self._verify_redis_set_call(data)

    def test_set_shutting_down(self) -> None:
        """Tests set_shutting_down method."""
        # Run test
        self.store.set_shutting_down(self.API_KEY)

        # Verification
        data = schemas.WorkerInternal(state=schemas.WorkerState.SHUTTING_DOWN)
        self._verify_redis_exists_call()
        self._verify_redis_set_call(data)

    def _verify_redis_set_call(self, data: schemas.Worker) -> None:
        """Verifies calls to the mocked redis.Redis.set() function.

        Args:
            data: Worker object that was passed to the set() function.
        """
        serialized = schemas.encode(schemas.WorkerInternalSchema, data)
        result = self.connection.set.called_once_with(self.API_KEY, serialized)
        self.assertTrue(result)

    def _verify_redis_exists_call(self) -> None:
        """Verifies calls to the mocked redis.Redis.exists() function."""
        result = self.connection.exists.called_once_with(self.API_KEY)
        self.assertTrue(result)
