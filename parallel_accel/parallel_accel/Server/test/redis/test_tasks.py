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
# pylint: disable=protected-access, wrong-import-order

"""Unit test for tasks module"""
import datetime
import importlib
import os
import time
import unittest
import unittest.mock
import uuid
import aiounittest
import redis

from parallel_accel.shared import schemas

from src import redis as shared_redis


class TestTasksRedisStore(aiounittest.AsyncTestCase):
    """Tests TasksRedisStore class behavior."""

    DEADLINE = datetime.timedelta(seconds=100)
    TASK_ID = uuid.uuid4()

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

        importlib.reload(shared_redis)

        os.environ["REDISHOST"] = "localhost"

        cls.store = shared_redis.TasksRedisStore()
        cls.connection = cls.store._connections[
            shared_redis.tasks.RedisInstances.TASKS
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        """See base class documentation."""
        del os.environ["REDISHOST"]

        for patcher in cls.patchers:
            patcher.stop()

    def setUp(self) -> None:
        """See base class documentation."""
        self.connection.exists.return_value = 0

    def tearDown(self) -> None:
        """See base class documentation."""
        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

    def test_create_task_exists(self) -> None:
        """Tests create_task method behavior: task already exists"""
        # Test setup
        self.connection.exists.return_value = 1

        # Run test
        with self.assertRaises(shared_redis.TaskExistError):
            self.store.create_task(self.TASK_ID)

        # Verification
        self._verify_redis_exists_call()

        self.assertFalse(self.connection.rpush.called)
        self.assertFalse(self.connection.set.called)

    def test_create_task_not_exist(self) -> None:
        """Tests create_task method behavior: task not exist"""
        # Test setup
        self.connection.exists.return_value = 0

        # Run test
        self.store.create_task(self.TASK_ID)

        # Verification
        self._verify_redis_exists_call()

        self.mocked_redis.set.assert_called_once_with(
            shared_redis.TasksRedisStore.KeyType.MARKER.key_for(self.TASK_ID),
            shared_redis.TasksRedisStore.KeyType.MARKER.name.lower(),
            shared_redis.TasksRedisStore._KEY_EXPIRE_TIME,
        )

        status = schemas.TaskStatus(schemas.TaskState.PENDING)
        self._verify_redis_rpush_call(status)

    def test_has_task(self) -> None:
        """Tests has_task method behavior."""
        # Run test
        result = self.store.has_task(self.TASK_ID)

        # Verification
        self.assertFalse(result)

        self._verify_redis_exists_call()

    def test_set_completed(self) -> None:
        """Tests set_completed method behavior."""
        status = schemas.TaskStatus(
            schemas.TaskState.DONE, "Unexpected error", False
        )

        # Run test
        self.store.set_completed(self.TASK_ID, status.success, status.error)

        # Verification
        self._verify_redis_rpush_call(status, datetime.timedelta(minutes=15))

    def test_set_running(self) -> None:
        """Tests set_running method behavior."""
        # Run test
        self.store.set_running(self.TASK_ID)

        # Verification
        status = schemas.TaskStatus(schemas.TaskState.RUNNING)
        self._verify_redis_rpush_call(status)

    async def test_subscribe(self) -> None:
        """Tests subscribe method behavior."""
        # Test setup
        self.connection.exists.return_value = 1

        items = [
            schemas.TaskStatus(x)
            for x in (schemas.TaskState.PENDING, schemas.TaskState.RUNNING)
        ]
        items.append(
            schemas.TaskStatus(schemas.TaskState.DONE, "Some error", False)
        )
        self.connection.blpop.side_effect = [
            (
                shared_redis.TasksRedisStore.KeyType.QUEUE.key_for(
                    self.TASK_ID
                ),
                schemas.encode(schemas.TaskStatusSchema, x),
            )
            for x in items
        ]

        # Run test
        generated_items = [
            x async for x in self.store.subscribe(self.TASK_ID, self.DEADLINE)
        ]
        self.assertEqual(generated_items, items)

        # Verification
        call_list = [
            (
                (
                    shared_redis.TasksRedisStore.KeyType.QUEUE.key_for(
                        self.TASK_ID
                    ),
                ),
                {
                    "timeout": (
                        self.DEADLINE
                        - datetime.timedelta(
                            seconds=self.mocked_time.return_value
                        )
                    ).total_seconds()
                },
            )
            for _ in range(len(items))
        ]
        self.assertEqual(self.connection.blpop.call_args_list, call_list)

        result = self.connection.delete.called_once_with(self.TASK_ID)
        self.assertTrue(result)

        self._verify_redis_exists_call()

    async def test_subscribe_not_exists(self) -> None:
        """Tests subscribe method behavior: task not exists."""
        # Test setup
        self.connection.exists.return_value = 0

        # Run test
        with self.assertRaises(shared_redis.TaskNotFoundError):
            async for _ in self.store.subscribe(self.TASK_ID, self.DEADLINE):
                pass

        # Verification
        self._verify_redis_exists_call()
        self.assertFalse(self.mocked_redis.blpop.called)
        self.assertFalse(self.mocked_redis.delete.called)

    async def test_subscribe_timeout(self) -> None:
        """Tests subscribe method behavior: BLPOP timed out."""
        # Test setup
        self.connection.exists.return_value = 1
        self.connection.blpop.side_effect = [None]

        # Run test
        async for status in self.store.subscribe(self.TASK_ID, self.DEADLINE):
            self.assertIsNone(status)

        # Verification
        self._verify_redis_exists_call()
        self.connection.blpop.called_once_with(
            shared_redis.TasksRedisStore.KeyType.QUEUE.key_for(self.TASK_ID),
            timeout=(
                self.DEADLINE
                - datetime.timedelta(seconds=self.mocked_time.return_value)
            ).total_seconds(),
        )
        self.assertFalse(self.connection.delete.called)

    def _verify_redis_exists_call(self) -> None:
        """Verifies calls to the mocked redis.Redis.exists() function."""
        self.connection.exists.assert_called_once_with(
            shared_redis.TasksRedisStore.KeyType.MARKER.key_for(self.TASK_ID)
        )

    def _verify_redis_rpush_call(
        self,
        status: schemas.TaskStatus,
        expire_time: datetime.timedelta = shared_redis.TasksRedisStore._KEY_EXPIRE_TIME,
    ) -> None:
        """Verifies calls to the mocked redis.Redis.rpush() function.

        Args:
            status: TaskStatus object that was pushed to the Redis queue.
            expire_time: Key expiration time that was set.
        """
        serialized = schemas.encode(schemas.TaskStatusSchema, status)
        self.connection.rpush.assert_called_once_with(
            shared_redis.TasksRedisStore.KeyType.QUEUE.key_for(self.TASK_ID),
            serialized,
        )
        self.connection.expire.assert_called_once_with(
            shared_redis.TasksRedisStore.KeyType.QUEUE.key_for(self.TASK_ID),
            expire_time,
        )
