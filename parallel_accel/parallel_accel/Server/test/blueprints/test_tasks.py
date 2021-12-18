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
import datetime
import http
import secrets
import time
import typing
import uuid
import unittest
import unittest.mock

from parallel_accel.shared import schemas

from src import containers, redis


class TestTasksBlueprint(unittest.TestCase):
    """Tests TasksBlueprint class behavior."""

    API_KEY = secrets.token_hex(16)
    TASK_ID = uuid.uuid4()

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.mocked_store = unittest.mock.Mock(spec=redis.TasksRedisStore)
        cls.mocked_store.return_value = cls.mocked_store

        cls.container = containers.ApplicationContainer()
        cls.container.tasks_store.override(cls.mocked_store)
        cls.container.sanic_app.add_kwargs(app_name=cls.__name__)

        cls.app = cls.container.sanic_app()

    @classmethod
    def tearDownClass(cls) -> None:
        """See base class documentation."""
        cls.container.shutdown_resources()

    def tearDown(self) -> None:
        """See base class documentation."""
        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

    def test_event_stream(self) -> None:
        """Tests GET /<task_id>/stream request handler."""
        items = [
            schemas.TaskStatus(x)
            for x in (schemas.TaskState.PENDING, schemas.TaskState.RUNNING)
        ]

        async def subscribe_side_effect(
            task_id: uuid.UUID, deadline: datetime.timedelta
        ) -> None:
            self.assertEqual(task_id, self.TASK_ID)
            self.assertAlmostEqual(
                deadline.total_seconds(), time.time() + 600, 1
            )
            for item in items:
                yield item

        self.mocked_store.has_task.return_value = True
        self.mocked_store.subscribe.side_effect = subscribe_side_effect

        # Run test
        headers = self.__get_request_headers()
        url = self.__get_request_url()

        _, response = self.app.test_client.get(url, headers=headers)

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.OK)

        events = [
            schemas.decode(schemas.TaskStatusEventSchema, x)
            for x in response.text.strip().split("\n\n")
        ]
        expected_events = [
            schemas.TaskStatusEvent(
                data=x,
                event=schemas.TaskStatusEvent.__name__,
                id=self.TASK_ID,
            )
            for x in items
        ]
        self.assertEqual(events, expected_events)

        self._verify_redis_store()

    def test_event_stream_timeout(self) -> None:
        """Tests GET /<task_id>/stream request handler: stream timed out"""

        async def subscribe_side_effect(
            task_id: uuid.UUID, deadline: datetime.timedelta
        ) -> None:
            self.assertEqual(task_id, self.TASK_ID)
            self.assertAlmostEqual(
                deadline.total_seconds(), time.time() + 600, 1
            )
            yield None

        self.mocked_store.has_task.return_value = True
        self.mocked_store.subscribe.side_effect = subscribe_side_effect

        # Run test
        headers = self.__get_request_headers()
        url = self.__get_request_url()

        _, response = self.app.test_client.get(url, headers=headers)

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.OK)
        event = schemas.StreamTimeoutEvent(id=self.TASK_ID)
        serialized_event = schemas.encode(
            schemas.StreamTimeoutEventSchema, event
        )
        self.assertEqual(response.text, serialized_event)

        self._verify_redis_store()

    def test_event_stream_not_exists(self) -> None:
        """Tests GET /<task_id>/stream request handler: task not exist."""
        self.mocked_store.has_task.return_value = False

        # Run test
        headers = self.__get_request_headers()
        url = self.__get_request_url()

        _, response = self.app.test_client.get(url, headers=headers)

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.NOT_FOUND)

    def _verify_redis_store(self) -> None:
        """Verifies calls to mocked Redis store."""
        self.mocked_store.has_task.assert_called_once_with(self.TASK_ID)
        self.mocked_store.subscribe.assert_called_once()

    def __get_request_url(self) -> str:
        """Gets request URL."""
        return f"/api/v1/tasks/{str(self.TASK_ID)}/stream"

    def __get_request_headers(self) -> typing.Dict[str, str]:
        """Gets request headers."""
        return {"X-API-Key": self.API_KEY}
