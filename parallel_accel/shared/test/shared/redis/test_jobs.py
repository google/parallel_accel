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

"""Unit test for jobs module"""
import importlib
import os
import secrets
import unittest
import unittest.mock
import uuid
import redis

from parallel_accel.shared.redis import base, jobs


class TestJobsRedisStore(unittest.TestCase):
    """Tests JobsRedisStore class behavior."""

    API_KEY = secrets.token_hex(16)
    JOB_ID = str(uuid.uuid4())

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.patchers = []

        cls.mocked_redis = unittest.mock.Mock(spec=redis.Redis)
        cls.mocked_redis.return_value = cls.mocked_redis
        patcher = unittest.mock.patch("redis.Redis", cls.mocked_redis)
        cls.patchers.append(patcher)

        for patcher in cls.patchers:
            patcher.start()

        importlib.reload(base)
        importlib.reload(jobs)

        os.environ["REDISHOST"] = "localhost"

        cls.store = jobs.JobsRedisStore()

    @classmethod
    def tearDownClass(cls) -> None:
        """See base class documentation."""
        del os.environ["REDISHOST"]

        for patcher in cls.patchers:
            patcher.stop()

    def setUp(self) -> None:
        """See base class documentation."""
        for connection in self.store._connections.values():
            connection.exists.return_value = 1

    def tearDown(self) -> None:
        """See base class documentation."""
        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

    def test_has_job(self) -> None:
        """Tests has_job method."""
        # Run test
        result = self.store.has_job(self.JOB_ID)
        self.assertTrue(result)

        # Verifciation
        self.store._connections[
            base.RedisInstances.JOBS
        ].exists.assert_called_once_with(
            jobs.JobsRedisStore.KeyType.CONTEXT.key_for(self.JOB_ID)
        )

    def test_has_job_result(self) -> None:
        """Tests has_job_result method."""
        # Run test
        result = self.store.has_job_results(self.JOB_ID)
        self.assertTrue(result)

        # Verifciation
        self.store._connections[
            base.RedisInstances.JOBS
        ].exists.assert_called_once_with(
            jobs.JobsRedisStore.KeyType.STATUS.key_for(self.JOB_ID)
        )
