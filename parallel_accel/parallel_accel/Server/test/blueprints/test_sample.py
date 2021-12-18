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
"""Unit test for jobs module: sample job type"""
import datetime
import http
import secrets
import time
import typing
import uuid
import unittest
import unittest.mock
import linear_algebra

from parallel_accel.shared import redis as shared_redis, schemas

from src import containers, redis


class TestSampleBlueprint(unittest.TestCase):
    """Tests SampleBlueprint class behavior."""

    API_KEY = secrets.token_hex(16)
    JOB_ID = uuid.uuid4()

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.mocked_manager = unittest.mock.Mock(spec=redis.JobsManager)
        cls.mocked_manager.return_value = cls.mocked_manager

        cls.mocked_workers_store = unittest.mock.Mock(
            spec=shared_redis.WorkersRedisStore
        )
        cls.mocked_workers_store.return_value = cls.mocked_workers_store

        cls.container = containers.ApplicationContainer()
        cls.container.jobs_manager.override(cls.mocked_manager)
        cls.container.workers_store.override(cls.mocked_workers_store)
        cls.container.sanic_app.add_kwargs(app_name=cls.__name__)
        cls.app = cls.container.sanic_app()

    @classmethod
    def tearDownClass(cls) -> None:
        """See base class documentation."""
        cls.container.shutdown_resources()

    def tearDown(self) -> None:
        """See base class documentation."""
        self.mocked_manager.create_job.side_effect = None

        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

    def test_create_job(self) -> None:
        """Tests POST /submit request handler."""
        # Test setup
        self.mocked_manager.create_job.return_value = str(self.JOB_ID)
        self.mocked_workers_store.get_worker.return_value = (
            schemas.WorkerInternal(schemas.WorkerState.IDLE)
        )

        # Run test
        context = schemas.SampleJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
        )
        data = schemas.encode(schemas.SampleJobContextSchema, context)
        headers = self.__get_request_headers()

        _, response = self.app.test_client.post(
            "/api/v1/jobs/sample/submit", headers=headers, data=data
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.CREATED)

        job = schemas.JobSubmitted(self.JOB_ID)
        expected_json = schemas.JobSubmittedSchema.dump(job)
        self.assertEqual(response.json, expected_json)

        self.mocked_manager.create_job.called_once_with(
            self.API_KEY, schemas.JobType.SAMPLE, context
        )

    def test_create_batch_job(self) -> None:
        """Tests POST /batch/submit request handler."""
        # Test setup
        self.mocked_manager.create_job.return_value = str(self.JOB_ID)
        self.mocked_workers_store.get_worker.return_value = (
            schemas.WorkerInternal(schemas.WorkerState.IDLE)
        )

        # Run test
        context = schemas.SampleBatchJobContext(
            acyclic_graphs=[linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1)))],
            params=[linear_algebra.ParamResolver(None)],
            repetitions=1,
        )
        data = schemas.encode(schemas.SampleBatchJobContextSchema, context)
        headers = self.__get_request_headers()

        _, response = self.app.test_client.post(
            "/api/v1/jobs/sample/batch/submit", headers=headers, data=data
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.CREATED)

        job = schemas.JobSubmitted(self.JOB_ID)
        expected_json = schemas.JobSubmittedSchema.dump(job)
        self.assertEqual(response.json, expected_json)

        self.mocked_manager.create_job.called_once_with(
            self.API_KEY, schemas.JobType.SAMPLE, context
        )

    def test_create_sweep_job(self) -> None:
        """Tests POST /sweep/submit request handler."""
        # Test setup
        self.mocked_manager.create_job.return_value = str(self.JOB_ID)
        self.mocked_workers_store.get_worker.return_value = (
            schemas.WorkerInternal(schemas.WorkerState.IDLE)
        )

        # Run test
        context = schemas.SampleSweepJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            params=linear_algebra.ParamResolver(None),
        )
        data = schemas.encode(schemas.SampleSweepJobContextSchema, context)
        headers = self.__get_request_headers()

        _, response = self.app.test_client.post(
            "/api/v1/jobs/sample/sweep/submit", headers=headers, data=data
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.CREATED)

        job = schemas.JobSubmitted(self.JOB_ID)
        expected_json = schemas.JobSubmittedSchema.dump(job)
        self.assertEqual(response.json, expected_json)

        self.mocked_manager.create_job.called_once_with(
            self.API_KEY, schemas.JobType.SAMPLE, context
        )

    def test_create_job_bad_request(self) -> None:
        """Tests POST /submit request handler: bad request"""
        # Test setup
        self.mocked_workers_store.get_worker.return_value = (
            schemas.WorkerInternal(schemas.WorkerState.IDLE)
        )

        # Run test
        headers = self.__get_request_headers()

        _, response = self.app.test_client.post(
            "/api/v1/jobs/sample/submit", headers=headers
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.BAD_REQUEST)

    def test_create_job_failed_to_store_job_context(self) -> None:
        """Tests POST /submit request handler: failed to store job context in
        the Redis store."""
        # Test setup
        self.mocked_manager.create_job.side_effect = redis.jobs.CreateJobError
        self.mocked_workers_store.get_worker.return_value = (
            schemas.WorkerInternal(schemas.WorkerState.IDLE)
        )

        # Run test
        context = schemas.SampleJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
        )
        data = schemas.encode(schemas.SampleJobContextSchema, context)
        headers = self.__get_request_headers()

        _, response = self.app.test_client.post(
            "/api/v1/jobs/sample/submit", headers=headers, data=data
        )

        # Verification
        self.assertEqual(
            response.status_code, http.HTTPStatus.INTERNAL_SERVER_ERROR
        )

    def test_create_job_worker_offline(self) -> None:
        """Tests POST /submit request handler: worker is offline"""
        # Test setup
        self.mocked_workers_store.get_worker.return_value = (
            schemas.WorkerInternal(schemas.WorkerState.OFFLINE)
        )

        # Run test
        context = schemas.SampleJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
        )
        data = schemas.encode(schemas.SampleJobContextSchema, context)
        headers = self.__get_request_headers()

        _, response = self.app.test_client.post(
            "/api/v1/jobs/sample/submit", headers=headers, data=data
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.BAD_REQUEST)

    def test_handle_status_stream(self) -> None:
        """Tests GET /{uuid}/stream request handler."""
        # Test setup
        job_id = str(uuid.uuid4())

        items = [schemas.SampleJobResult(job_id, schemas.JobStatus.NOT_STARTED)]

        async def subscribe_side_effect(
            job_uuid: str, deadline: datetime.timedelta
        ) -> None:
            self.assertEqual(job_uuid, job_id)
            self.assertAlmostEqual(
                deadline.total_seconds(), time.time() + 600, 1
            )
            for item in items:
                yield item

        self.mocked_manager.has_job.return_value = True
        self.mocked_manager.is_same_api_key.return_value = True
        self.mocked_manager.is_same_job_type.return_value = True
        self.mocked_manager.subscribe_job_status.side_effect = (
            subscribe_side_effect
        )

        # Run test
        headers = self.__get_request_headers()
        _, response = self.app.test_client.get(
            f"/api/v1/jobs/sample/{job_id}/stream", headers=headers
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.OK)

        event = schemas.SampleJobStatusEvent(
            event=schemas.SampleJobStatusEvent.__name__,
            id=job_id,
            data=items[0],
            timestamp=int(time.time()),
        )
        serialized_event = schemas.encode(
            schemas.SampleJobStatusEventSchema, event
        )
        self.assertEqual(response.text, serialized_event)

        self.mocked_manager.has_job.called_once_with(job_id)
        self.mocked_manager.is_same_api_key.called_once_with(
            job_id, self.API_KEY
        )
        self.mocked_manager.is_same_job_type.called_once_with(
            job_id, schemas.JobType.EXPECTATION
        )
        self.mocked_manager.subscribe_job_status.called_once_with(
            self.API_KEY, job_id
        )

    def test_handle_status_stream_timeout(self) -> None:
        """Tests GET /{uuid}/stream request handler: stream timed out."""
        # Test setup
        job_id = str(uuid.uuid4())

        async def subscribe_side_effect(
            job_uuid: str, deadline: datetime.timedelta
        ) -> None:
            self.assertEqual(job_uuid, job_id)
            self.assertAlmostEqual(
                deadline.total_seconds(), time.time() + 600, 1
            )
            yield None

        self.mocked_manager.has_job.return_value = True
        self.mocked_manager.is_same_api_key.return_value = True
        self.mocked_manager.is_same_job_type.return_value = True
        self.mocked_manager.subscribe_job_status.side_effect = (
            subscribe_side_effect
        )

        # Run test
        headers = self.__get_request_headers()
        _, response = self.app.test_client.get(
            f"/api/v1/jobs/sample/{job_id}/stream", headers=headers
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.OK)

        event = schemas.StreamTimeoutEvent(id=job_id)
        serialized_event = schemas.encode(
            schemas.StreamTimeoutEventSchema, event
        )
        self.assertEqual(response.text, serialized_event)

        self.mocked_manager.has_job.called_once_with(job_id)
        self.mocked_manager.is_same_api_key.called_once_with(
            job_id, self.API_KEY
        )
        self.mocked_manager.is_same_job_type.called_once_with(
            job_id, schemas.JobType.EXPECTATION
        )
        self.mocked_manager.subscribe_job_status.called_once_with(
            self.API_KEY, job_id
        )

    def test_handle_status_stream_job_not_found(self) -> None:
        """Tests GET /{uuid}/stream request handler: job was not found."""
        # Test setup
        job_id = str(uuid.uuid4())

        self.mocked_manager.has_job.return_value = False

        # Run test
        headers = self.__get_request_headers()
        _, response = self.app.test_client.get(
            f"/api/v1/jobs/sample/{job_id}/stream", headers=headers
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.NOT_FOUND)

        data = schemas.APIError(
            http.HTTPStatus.NOT_FOUND, f"No experiment found with id {job_id}"
        )
        self.assertEqual(response.json, schemas.APIErrorSchema.dump(data))

        self.mocked_manager.has_job.called_once_with(job_id)
        self.assertFalse(self.mocked_manager.is_same_api_key.called)
        self.assertFalse(self.mocked_manager.is_same_job_type.called)
        self.assertFalse(self.mocked_manager.subscribe_job_status.called)

    def test_handle_status_stream_wrong_api_key(self) -> None:
        """Tests GET /{uuid}/stream request handler: wrong API key."""
        # Test setup
        job_id = str(uuid.uuid4())

        self.mocked_manager.has_job.return_value = True
        self.mocked_manager.is_same_api_key.return_value = False

        # Run test
        headers = self.__get_request_headers()
        _, response = self.app.test_client.get(
            f"/api/v1/jobs/sample/{job_id}/stream", headers=headers
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.NOT_FOUND)

        data = schemas.APIError(
            http.HTTPStatus.NOT_FOUND, f"No experiment found with id {job_id}"
        )
        self.assertEqual(response.json, schemas.APIErrorSchema.dump(data))

        self.mocked_manager.has_job.called_once_with(job_id)
        self.mocked_manager.is_same_api_key.called_once_with(
            job_id, self.API_KEY
        )
        self.assertFalse(self.mocked_manager.is_same_job_type.called)
        self.assertFalse(self.mocked_manager.subscribe_job_status.called)

    def test_handle_status_stream_wrong_job_type(self) -> None:
        """Tests GET /{uuid}/stream request handler: wrong job type."""
        # Test setup
        job_id = str(uuid.uuid4())

        self.mocked_manager.has_job.return_value = True
        self.mocked_manager.is_same_api_key.return_value = True
        self.mocked_manager.is_same_job_type.return_value = False

        # Run test
        headers = self.__get_request_headers()
        _, response = self.app.test_client.get(
            f"/api/v1/jobs/exp/{job_id}/stream", headers=headers
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.BAD_REQUEST)

        data = schemas.APIError(
            http.HTTPStatus.BAD_REQUEST,
            f"{job_id} is not {schemas.JobType.EXPECTATION.name} job",
        )
        self.assertEqual(response.json, schemas.APIErrorSchema.dump(data))

        self.mocked_manager.has_job.called_once_with(job_id)
        self.mocked_manager.is_same_api_key.called_once_with(
            job_id, self.API_KEY
        )
        self.mocked_manager.is_same_job_type.called_once_with(
            job_id, schemas.JobType.EXPECTATION
        )
        self.assertFalse(self.mocked_manager.subscribe_job_status.called)

    def __get_request_headers(self) -> typing.Dict[str, str]:
        """Gets request headers."""
        return {"X-API-Key": self.API_KEY}
