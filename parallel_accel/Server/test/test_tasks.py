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
"""Unit test for tasks module"""
import asyncio
import unittest
import unittest.mock
import uuid
import aiounittest
import sanic

from src import redis, tasks


class TestAsyncTask(aiounittest.AsyncTestCase):
    """Tests AsyncTask class behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.mocked_store = unittest.mock.Mock(spec=redis.TasksRedisStore)
        cls.coro_called = False

    def setUp(self) -> None:
        """See base class documentation."""
        tasks.TasksManager.AsyncTask.initialize(self.mocked_store)
        self.task = tasks.TasksManager.AsyncTask(self._test_coroutine)

    def tearDown(self) -> None:
        """See base class documentation."""
        self.coro_called = False
        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

    async def test_call(self) -> None:
        """Tests __call__ method behavior."""
        # Run test
        await self.task()

        # Verification
        self.assertTrue(self.coro_called)
        self.mocked_store.create_task.assert_called_once_with(self.task.id)
        self.mocked_store.set_running.assert_called_once_with(self.task.id)
        self.mocked_store.set_completed.assert_called_once_with(
            self.task.id, True, None
        )

    async def _test_coroutine(self) -> None:
        """Coroutine function passed to the AsyncTask."""
        self.coro_called = True


class TestTasksManager(aiounittest.AsyncTestCase):
    """Tests TasksManager class behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.mocked_app = unittest.mock.Mock(spec=sanic.Sanic)
        cls.mocked_store = unittest.mock.Mock(spec=redis.TasksRedisStore)

    def setUp(self) -> None:
        """See base class documentation."""
        self.manager = tasks.TasksManager()
        self.manager.initialize(self.mocked_app, self.mocked_store)

    def tearDown(self) -> None:
        """See base class documentation."""
        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

    async def test_post_coro(self) -> None:
        """Tests post_coro method behavior."""

        async def test_coroutine():
            pass

        # Run test
        task_id = self.manager.post_coro(test_coroutine)

        # Wait for coroutines
        for coro in (
            y
            for x in self.mocked_app.add_task.call_args
            for y in x
            if asyncio.iscoroutine(y)
        ):
            await coro

        # Verification
        self.assertTrue(isinstance(task_id, uuid.UUID))
        self.assertTrue(self.mocked_app.add_task.called)
