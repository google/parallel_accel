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
"""This module defines /jobs/queue endpoint blueprints"""
import http
import uuid
import sanic

from parallel_accel.shared import schemas

from .. import redis, tasks
from . import utils


class JobsQueueBlueprint(sanic.Blueprint):  # pylint: disable=abstract-method
    """Handles /jobs/queue endpoint."""

    def __init__(
        self, jobs_manager: redis.JobsManager, tasks_manager: tasks.TasksManager
    ) -> None:
        """Creates JobsQueueBlueprint class instance.

        Args:
            jobs_manager: JobsManager instance.
        """
        super().__init__(self.__class__.__name__, "/jobs/queue")
        self.ctx.manager = jobs_manager
        self.ctx.tasks_manager = tasks_manager

        handler = utils.wrap_handler(self.handle_flush_queue)
        self.add_route(handler, "/", frozenset({"DELETE"}))

        handler = utils.wrap_handler(self.handle_get_queue)
        self.add_route(handler, "/", frozenset({"GET"}))

        handler = utils.wrap_handler(self.handle_get_pending_job)
        self.add_route(handler, r"/<job_id:uuid>", frozenset({"GET"}))

    async def handle_flush_queue(
        self, request: sanic.request.Request
    ) -> sanic.response.HTTPResponse:
        """Handles DELETE / endpoint request.

        Args:
            request: Incoming HTTP request.

        Returns:
            HTTP response holding JSON encoded async task id.
        """
        sanic.log.logger.debug("Handling flush job queue request")

        task_id = self.ctx.tasks_manager.post_function(
            self.ctx.manager.flush_job_queue, request.ctx.api_key
        )

        data = schemas.TaskSubmitted(task_id)
        serialized = schemas.encode(schemas.TaskSubmittedSchema, data)

        return utils.make_api_response(serialized)

    async def handle_get_queue(
        self, request: sanic.request.Request
    ) -> sanic.response.HTTPResponse:
        """Handles GET / endpoint request.

        Args:
            request: Incoming HTTP request.

        Returns:
            HTTP response holding JSON encoded list of jobs ids.
        """
        sanic.log.logger.debug("Handling get job queue request")

        queue = self.ctx.manager.get_job_queue(request.ctx.api_key)
        serialized = schemas.encode(schemas.JobsQueueSchema, queue)

        return utils.make_api_response(serialized)

    async def handle_get_pending_job(
        self, request: sanic.request.Request, job_id: uuid.UUID
    ) -> sanic.response.HTTPResponse:
        """Handles GET /{id} endpoint request.

        Args:
            request: Incoming HTTP request.
            job_id: Unique job id.

        Returns:
            HTTP response holding JSON encoded APIError or PendingJob object.
        """
        sanic.log.logger.debug("Handling get pending job request")
        job_id = str(job_id)

        if not self.ctx.manager.has_pending_job(request.ctx.api_key, job_id):
            status = http.HTTPStatus.NOT_FOUND
            data = schemas.APIError(
                status, f"No matching job objects for id {job_id}"
            )
            schema = schemas.APIErrorSchema
        else:
            job_status = self.ctx.manager.get_job_status(job_id)
            job_type = self.ctx.manager.get_job_type(job_id)
            status = http.HTTPStatus.OK
            data = schemas.PendingJob(job_id, job_status, job_type)
            schema = schemas.PendingJobSchema

        serialized = schemas.encode(schema, data)
        return utils.make_api_response(serialized, status)
