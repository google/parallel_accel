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
"""This module defines /tasks endpoint blueprints"""
import http
from typing import Optional
import uuid
import sanic

from parallel_accel.shared import schemas

from .. import redis
from . import utils


class TasksBlueprint(sanic.Blueprint):  # pylint: disable=abstract-method
    """Handles /tasks endpoint."""

    def __init__(self, store: redis.TasksRedisStore) -> None:
        """Creates TasksRedisStore class instance.

        Args:
            store: Instance of TasksRedisStore
        """
        super().__init__(self.__class__.__name__, "/tasks")
        self.ctx.store = store

        handler = utils.wrap_handler(self.handle_event_stream)
        self.add_route(handler, r"/<task_id:uuid>/stream")

    async def handle_event_stream(
        self, request: sanic.request.Request, task_id: uuid.UUID
    ) -> sanic.response.StreamingHTTPResponse:
        """Handles GET /<task_id>/stream endpoint request.

        Args:
            request: Incoming HTTP Request object.
            task_id: Requested task id.

        Returns:
            HTTP Response object.
        """
        sanic.log.logger.debug("Handling event stream request")

        if not self.ctx.store.has_task(task_id):
            status = http.HTTPStatus.NOT_FOUND
            data = schemas.APIError(
                status, f"No task found with id {str(task_id)}"
            )
            body = schemas.encode(schemas.APIErrorSchema, data)
            return utils.make_api_response(body, status)

        async def generate(response: sanic.response.HTTPResponse) -> None:
            """Generates server events stream.

            Args:
                response: Outgoing HTTPResponse object.
            """
            app = sanic.Sanic.get_app(request.app.name)
            task = utils.StreamHeartbeat(response)
            task.start(app)

            try:
                async for status in self.ctx.store.subscribe(
                    task_id, request.ctx.deadline
                ):
                    event = self._make_event(task_id, status)
                    await response.write(event)
            finally:
                task.stop()

        return sanic.response.stream(
            generate, content_type="text/event-stream; charset=utf-8"
        )

    @staticmethod
    def _make_event(
        task_id: uuid.UUID, status: Optional[schemas.TaskStatus]
    ) -> str:
        """Serializes TaskStatus object to server side event.

        Args:
            task_id: Unique task id.
            status: Optional TaskStatus object.

        Returns:
            JSON encoded TaskStatusEvent object if the status is not None,
            StreamTimeoutEvent otherwise.
        """
        if status is not None:
            event = schemas.TaskStatusEvent(data=status, id=task_id)
            schema = schemas.TaskStatusEventSchema
        else:
            event = schemas.StreamTimeoutEvent(id=task_id)
            schema = schemas.StreamTimeoutEventSchema

        return schemas.encode(schema, event)
