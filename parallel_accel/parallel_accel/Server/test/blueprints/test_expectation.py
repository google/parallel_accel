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
"""Unit test for jobs module: expectation job type"""
import http
import secrets
import typing
import uuid
import unittest
import unittest.mock
import linear_algebra

from parallel_accel.shared import redis as shared_redis, schemas

from src import containers, redis


class TestExpectationBlueprint(unittest.TestCase):
    """Tests ExpectationBlueprint class behavior."""

    API_KEY = secrets.token_hex(16)

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
        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

    def test_create_job(self) -> None:
        """Tests POST /submit request handler."""
        # Test setup
        job_id = str(uuid.uuid4())
        self.mocked_manager.create_job.return_value = job_id
        self.mocked_workers_store.get_worker.return_value = (
            schemas.WorkerInternal(schemas.WorkerState.IDLE)
        )

        # Run test
        discretes = linear_algebra.LinearSpace.range(2)
        context = schemas.ExpectationJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
            operators=[
                linear_algebra.flip_x_axis(discretes[0]) + linear_algebra.flip_y_axis(discretes[1]),
                linear_algebra.flip_y_axis(discretes[0]) + linear_algebra.flip_x_axis(discretes[1]),
            ],
        )
        data = schemas.encode(schemas.ExpectationJobContextSchema, context)
        headers = self.__get_request_headers()

        _, response = self.app.test_client.post(
            "/api/v1/jobs/exp/submit", headers=headers, data=data
        )

        # Verification
        self.assertEqual(response.status_code, http.HTTPStatus.CREATED)

        job = schemas.JobSubmitted(job_id)
        expected_json = schemas.JobSubmittedSchema.dump(job)
        self.assertEqual(response.json, expected_json)

        self.mocked_manager.create_job.called_once_with(
            self.API_KEY, schemas.JobType.EXPECTATION, context
        )

    def __get_request_headers(self) -> typing.Dict[str, str]:
        """Gets request headers."""
        return {"X-API-Key": self.API_KEY}
