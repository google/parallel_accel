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
"""Unit test for worker module"""
import http
import secrets
import typing
import uuid
import unittest
import unittest.mock

from parallel_accel.shared import redis, schemas

from src import containers, tasks, worker_manager


class TestWorkerBlueprint(unittest.TestCase):
    """Tests WorkerBlueprint class behavior."""

    API_KEY = secrets.token_hex(16)

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.mocked_workers_store = unittest.mock.Mock(
            spec=redis.WorkersRedisStore
        )
        cls.mocked_workers_store.return_value = cls.mocked_workers_store

        cls.mocked_tasks_manager = unittest.mock.Mock(spec=tasks.TasksManager)
        cls.mocked_tasks_manager.return_value = cls.mocked_tasks_manager

        cls.mocked_workers_manager = unittest.mock.Mock(
            spec=worker_manager.ASICWorkerManager
        )
        cls.mocked_workers_manager.return_value = cls.mocked_workers_manager

        cls.container = containers.ApplicationContainer()
        cls.container.tasks_manager.override(cls.mocked_tasks_manager)
        cls.container.worker_manager.override(cls.mocked_workers_manager)
        cls.container.workers_store.override(cls.mocked_workers_store)
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

    def test_handle_worker_command(self) -> None:
        """Tests POST /{command} request handler."""
        # Test setup
        job_id = uuid.uuid4()
        self.mocked_tasks_manager.post_coro.return_value = job_id

        for command in worker_manager.WorkerCommand:
            # Run test
            headers = self.__get_request_headers()
            _, response = self.app.test_client.post(
                f"/api/v1/worker/{command.name.lower()}",
                headers=headers,
            )

            # Verification
            self.assertEqual(response.status_code, http.HTTPStatus.OK)

            task = schemas.TaskSubmitted(job_id)
            serialized_task = schemas.TaskSubmittedSchema.dump(task)
            self.assertEqual(response.json, serialized_task)

        # Verification
        call_args_list = [
            ((self.mocked_workers_manager.handle_command, self.API_KEY, x),)
            for x in worker_manager.WorkerCommand
        ]
        self.assertEqual(
            self.mocked_tasks_manager.post_coro.call_args_list, call_args_list
        )

    def test_handle_get_status(self) -> None:
        """Tests GET /status request handler"""
        # Test setup
        self.mocked_workers_store.has_worker.return_value = True

        worker = schemas.Worker(schemas.WorkerState.OFFLINE)
        self.mocked_workers_store.get_worker.return_value = worker

        # Run test
        headers = self.__get_request_headers()
        _, response = self.app.test_client.get(
            "/api/v1/worker/status",
            headers=headers,
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.OK)

        serialized_worker = schemas.WorkerSchema.dump(worker)
        self.assertEqual(response.json, serialized_worker)

        self.mocked_workers_store.has_worker.assert_called_once_with(
            self.API_KEY
        )
        self.mocked_workers_store.get_worker.assert_called_once_with(
            self.API_KEY
        )

    def test_handle_get_status_not_found(self) -> None:
        """Tests GET /status request handler: worker not found."""
        # Test setup
        self.mocked_workers_store.has_worker.return_value = False

        # Run test
        headers = self.__get_request_headers()
        _, response = self.app.test_client.get(
            "/api/v1/worker/status",
            headers=headers,
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.NOT_FOUND)

        self.mocked_workers_store.has_worker.assert_called_once_with(
            self.API_KEY
        )

    def __get_request_headers(self) -> typing.Dict[str, str]:
        """Gets request headers."""
        return {"X-API-Key": self.API_KEY}
