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
# pylint: disable=abstract-method,too-many-ancestors

"""This module defines /jobs endpoint blueprints"""
import enum
import http
from typing import Optional
import os
import uuid
import marshmallow
import sanic

from parallel_accel.shared import redis as shared_redis, schemas

from .. import redis
from . import utils


class SimulationBlueprint(sanic.Blueprint):
    """Base class for simulation job requests."""

    class ContextType(enum.IntEnum):
        """Submitted job context type."""

        SIMPLE = enum.auto()
        SWEEP = enum.auto()
        BATCH = enum.auto()

    def __init__(
        self,
        jobs_manager: redis.JobsManager,
        workers_store: shared_redis.WorkersRedisStore,
        endpoint: str,
    ) -> None:
        """Creates JobBlueprint class instance.

        Args:
            jobs_manager: JobsManager instance.
            worker_store: WorkersRedisStore instance.
            job_type: Simulation job type.
        """
        super().__init__(self.__class__.__name__, f"/jobs/{endpoint}")

        self.ctx.jobs_manager = jobs_manager
        self.ctx.workers_store = workers_store

        handler = utils.wrap_handler(self.handle_status_stream)
        self.add_route(handler, r"/<job_id:uuid>/stream")

        for prefix, handler in (
            ("", self.handle_submit_simple_job),
            ("batch", self.handle_submit_batch_job),
            ("sweep", self.handle_submit_sweep_job),
        ):
            handler = utils.wrap_handler(handler)
            endpoint = os.path.join("/", prefix, "submit")
            self.add_route(handler, endpoint, frozenset({"POST"}))

    async def handle_submit_batch_job(
        self, request: sanic.request.Request
    ) -> sanic.response.HTTPResponse:
        """Handles POST /batch/submit endpoint request.

        Args:
            request: Incoming HTTP request.

        Returns:
            HTTPResponse object.
        """
        return self._handle_submit_job(
            request, SimulationBlueprint.ContextType.BATCH
        )

    async def handle_submit_simple_job(
        self, request: sanic.request.Request
    ) -> sanic.response.HTTPResponse:
        """Handles POST /submit endpoint request.

        Args:
            request: Incoming HTTP request.

        Returns:
            HTTPResponse object.
        """
        return self._handle_submit_job(
            request, SimulationBlueprint.ContextType.SIMPLE
        )

    async def handle_submit_sweep_job(
        self, request: sanic.request.Request
    ) -> sanic.response.HTTPResponse:
        """Handles POST /sweep/submit endpoint request.

        Args:
            request: Incoming HTTP request.

        Returns:
            HTTPResponse object.
        """
        return self._handle_submit_job(
            request, SimulationBlueprint.ContextType.SWEEP
        )

    async def handle_status_stream(
        self, request: sanic.request.Request, job_id: uuid.UUID
    ) -> sanic.response.HTTPResponse:
        """Handles GET /{uuid}/stream endpoint request.

        Args:
            request: Incoming HTTP request.
            uuid: Unique job id.

        Returns:
            HTTP response holding JSON encoded experiment results or API error.
        """
        sanic.log.logger.debug("Handling stream job status request")

        job_id = str(job_id)

        if not self.ctx.jobs_manager.has_job(
            job_id
        ) or not self.ctx.jobs_manager.is_same_api_key(
            job_id, request.ctx.api_key
        ):
            status = http.HTTPStatus.NOT_FOUND
            data = schemas.APIError(
                status, f"No experiment found with id {job_id}"
            )
            body = schemas.encode(schemas.APIErrorSchema, data)
            return utils.make_api_response(body, status)

        if not self.ctx.jobs_manager.is_same_job_type(
            job_id, self.ctx.job_type
        ):
            status = http.HTTPStatus.BAD_REQUEST
            data = schemas.APIError(
                status, f"{job_id} is not {self.ctx.job_type.name} job"
            )
            body = schemas.encode(schemas.APIErrorSchema, data)
            return utils.make_api_response(body, status)

        async def generate(
            response: sanic.response.StreamingHTTPResponse,
        ) -> None:
            """Generates server events stream.

            Args:
                response: Outgoing HTTPResponse object.
            """
            app = sanic.Sanic.get_app(request.app.name)
            task = utils.StreamHeartbeat(response)
            task.start(app)

            try:
                async for result in self.ctx.jobs_manager.subscribe_job_status(
                    job_id, request.ctx.deadline
                ):
                    event = self._make_event(job_id, result)
                    await response.write(event)
            finally:
                task.stop()

        return sanic.response.stream(
            generate, content_type="text/event-stream; charset=utf-8"
        )

    def _handle_submit_job(
        self, request: sanic.request.Request, context_type: ContextType
    ) -> sanic.response.HTTPResponse:
        """Handles submit job request.

        Args:
            request: Incoming HTTP request.
            context_type: Type of submitted job context.

        Returns:
            HTTPResponse object.

        Throws:
            marshmallow.ValidationError if the received job context does not
            match expected context type.
        """
        sanic.log.logger.debug(
            "Handling submit %s %s job request",
            context_type.name.lower(),
            self.ctx.job_type.name.lower(),
        )

        worker = self.ctx.workers_store.get_worker(request.ctx.api_key)
        if worker.state == schemas.WorkerState.OFFLINE:
            status = http.HTTPStatus.BAD_REQUEST
            data = schemas.APIError(
                status,
                "ASIC worker is offline. Please start the worker before"
                " submitting a new job.",
            )
            schema = schemas.APIErrorSchema
            return utils.make_api_response(schemas.encode(schema, data), status)

        context = request.json
        errors = self.ctx.context_schema_map[context_type].validate(context)
        if errors:
            sanic.log.logger.error("Invalid job context")
            raise marshmallow.ValidationError(errors)

        try:
            job_id = self.ctx.jobs_manager.create_job(
                request.ctx.api_key, self.ctx.job_type, context
            )

            status = http.HTTPStatus.CREATED
            data = schemas.JobSubmitted(job_id)
            schema = schemas.JobSubmittedSchema
        except redis.jobs.CreateJobError:
            status = http.HTTPStatus.INTERNAL_SERVER_ERROR
            data = schemas.APIError(status, "Failed to create a new job")
            schema = schemas.APIErrorSchema

        return utils.make_api_response(schemas.encode(schema, data), status)

    def _make_event(
        self, job_id: str, result: Optional[schemas.JobResult]
    ) -> str:
        """Creates serialized JobStatusEvent object.

        Args:
            job_id: Unique job id.
            result: Optional JobResult object.

        Returns:
            JSON encoded JobStatusEvent object if the result is not None,
            StreamTimeoutEvent otherwise.
        """
        if result is not None:
            event = schemas.JobStatusEvent(
                data=result, event=self.ctx.sse_schema_name, id=job_id
            )
            schema = schemas.JobStatusEventSchema
        else:
            event = schemas.StreamTimeoutEvent(id=job_id)
            schema = schemas.StreamTimeoutEventSchema

        return schemas.encode(schema, event)


#####################################
# Job type specific blueprints      #
#####################################


class SampleBlueprint(SimulationBlueprint):
    """Handles /jobs/sample endpoint."""

    def __init__(
        self,
        jobs_manager: redis.JobsManager,
        workers_store: shared_redis.WorkersRedisStore,
    ) -> None:
        """Creates SampleBlueprint class instance.

        Args:
            jobs_manager: JobsManager instance.
            worker_store: WorkersRedisStore instance.
        """
        super().__init__(jobs_manager, workers_store, "sample")
        self.ctx.context_schema_map = {
            SimulationBlueprint.ContextType.BATCH: schemas.SampleBatchJobContextSchema,
            SimulationBlueprint.ContextType.SIMPLE: schemas.SampleJobContextSchema,
            SimulationBlueprint.ContextType.SWEEP: schemas.SampleSweepJobContextSchema,
        }
        self.ctx.job_type = schemas.JobType.SAMPLE
        self.ctx.sse_schema_name = schemas.SampleJobStatusEvent.__name__


class ExpectationBlueprint(SimulationBlueprint):
    """Handles /jobs/exp endpoint."""

    def __init__(
        self,
        jobs_manager: redis.JobsManager,
        workers_store: shared_redis.WorkersRedisStore,
    ) -> None:
        """Creates ExpectationBlueprint class instance.

        Args:
            jobs_manager: JobsManager instance.
            worker_store: WorkersRedisStore instance.
        """
        super().__init__(jobs_manager, workers_store, "exp")
        self.ctx.context_schema_map = {
            SimulationBlueprint.ContextType.BATCH: schemas.ExpectationBatchJobContextSchema,
            SimulationBlueprint.ContextType.SIMPLE: schemas.ExpectationJobContextSchema,
            SimulationBlueprint.ContextType.SWEEP: schemas.ExpectationSweepJobContextSchema,
        }
        self.ctx.job_type = schemas.JobType.EXPECTATION
        self.ctx.sse_schema_name = schemas.ExpectationJobStatusEvent.__name__
