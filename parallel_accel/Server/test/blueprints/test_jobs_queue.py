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
"""Unit test for jobs_queue module"""
import http
import secrets
import typing
import uuid
import unittest
import unittest.mock

from parallel_accel.shared import schemas

from src import containers, redis, tasks


class TestJobsQueueBlueprint(unittest.TestCase):
    """Tests JobsQueueBlueprint class behavior."""

    API_KEY = secrets.token_hex(16)
    JOB_ID = uuid.uuid4()

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.mocked_store = unittest.mock.Mock(spec=redis.JobsManager)
        cls.mocked_store.return_value = cls.mocked_store

        cls.mocked_manager = unittest.mock.Mock(spec=tasks.TasksManager)
        cls.mocked_manager.return_value = cls.mocked_manager

        cls.container = containers.ApplicationContainer()
        cls.container.jobs_manager.override(cls.mocked_store)
        cls.container.tasks_manager.override(cls.mocked_manager)
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

    def test_handle_flush_queue(self) -> None:
        """Tests DELETE / request handler."""
        # Test setup
        self.mocked_manager.post_function.return_value = self.JOB_ID

        # Run test
        headers = self.__get_request_headers()
        _, response = self.app.test_client.delete(
            "/api/v1/jobs/queue",
            headers=headers,
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.OK)

        task = schemas.TaskSubmitted(self.JOB_ID)
        serialized_task = schemas.TaskSubmittedSchema.dump(task)
        self.assertEqual(response.json, serialized_task)

        self.mocked_manager.post_function.assert_called_once_with(
            self.mocked_store.flush_job_queue, self.API_KEY
        )

    def test_handle_get_queue(self) -> None:
        """Tests GET / request handler."""
        # Test setup
        queue = schemas.JobsQueue([self.JOB_ID])
        self.mocked_store.get_job_queue.return_value = queue

        # Run test
        headers = self.__get_request_headers()
        _, response = self.app.test_client.get(
            "/api/v1/jobs/queue",
            headers=headers,
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.OK)

        serialized_queue = schemas.JobsQueueSchema.dump(queue)
        self.assertEqual(response.json, serialized_queue)

        self.mocked_store.get_job_queue.assert_called_once_with(self.API_KEY)

    def test_handle_get_pending_job(self) -> None:
        """Tests GET /<:job_id> request handler"""
        # Test setup
        self.mocked_store.has_pending_job.return_value = True
        self.mocked_store.get_job_status.return_value = (
            schemas.JobStatus.NOT_STARTED
        )
        self.mocked_store.get_job_type.return_value = schemas.JobType.SAMPLE

        # Run test
        headers = self.__get_request_headers()
        _, response = self.app.test_client.get(
            f"/api/v1/jobs/queue/{str(self.JOB_ID)}", headers=headers
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.OK)

        job = schemas.PendingJob(
            self.JOB_ID,
            self.mocked_store.get_job_status.return_value,
            self.mocked_store.get_job_type.return_value,
        )
        serialized_job = schemas.PendingJobSchema.dump(job)
        self.assertEqual(response.json, serialized_job)

        self.mocked_store.has_pending_job.assert_called_once_with(
            self.API_KEY, str(self.JOB_ID)
        )
        self.mocked_store.get_job_status.assert_called_once_with(
            str(self.JOB_ID)
        )
        self.mocked_store.get_job_type.assert_called_once_with(str(self.JOB_ID))

    def test_handle_get_pending_job_not_found(self) -> None:
        """Tests GET /<:job_id> request handler: job not found."""
        # Test setup
        self.mocked_store.has_pending_job.return_value = False

        # Run test
        headers = self.__get_request_headers()
        _, response = self.app.test_client.get(
            f"/api/v1/jobs/queue/{str(self.JOB_ID)}", headers=headers
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.NOT_FOUND)
        self.mocked_store.has_pending_job.assert_called_once_with(
            self.API_KEY, str(self.JOB_ID)
        )

    def __get_request_headers(self) -> typing.Dict[str, str]:
        """Gets request headers."""
        return {"X-API-Key": self.API_KEY}
