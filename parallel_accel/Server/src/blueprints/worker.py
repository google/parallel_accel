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
"""This module defines /worker endpoint blueprints."""
import http
import sanic

from parallel_accel.shared import redis, schemas

from .. import tasks, worker_manager
from . import utils


class WorkerBlueprint(sanic.Blueprint):  # pylint: disable=abstract-method
    """Class /worker endpoint."""

    def __init__(
        self,
        store: redis.WorkersRedisStore,
        tasks_manager: tasks.TasksManager,
        asic_manager: worker_manager.ASICWorkerManager,
    ) -> None:
        """Creates WorkerBlueprint class instance.

        Args:
            store: WorkersRedisStore instance.
            tasks_manager: TasksManager instance.
            asic_manager: ASICWorkerManager instance.
        """
        super().__init__(self.__class__.__name__, "/worker")
        self.ctx.store = store
        self.ctx.tasks_manager = tasks_manager
        self.ctx.asic_manager = asic_manager

        handler = utils.wrap_handler(self.handle_get_status)
        self.add_route(handler, "/status")

        handler = utils.wrap_handler(self.handle_worker_command)
        pattern = "|".join(x.name.lower() for x in worker_manager.WorkerCommand)
        self.add_route(handler, f"/<command:{pattern}>", frozenset({"POST"}))

    async def handle_get_status(
        self, request: sanic.request.Request
    ) -> sanic.response.HTTPResponse:
        """Handles GET /status endpoint request.

        Args:
            request: Incoming HTTP Request object.

        Returns:
            HTTP Response object.
        """
        sanic.log.logger.debug("Handling worker status request")

        if not self.ctx.store.has_worker(request.ctx.api_key):
            status = http.HTTPStatus.NOT_FOUND
            data = schemas.APIError(
                status, "No worker object exists for given API key"
            )
            schema = schemas.APIErrorSchema
        else:
            status = http.HTTPStatus.OK
            data = self.ctx.store.get_worker(request.ctx.api_key)
            schema = schemas.WorkerSchema

        body = schemas.encode(schema, data)
        return utils.make_api_response(body, status)

    async def handle_worker_command(
        self, request: sanic.request.Request, command: str
    ) -> sanic.response.HTTPResponse:
        """Handles POST /{command} endpoint request.

        Args:
            request: Incoming HTTP request.
            command: Worker command.

        Returns:
            HTTP response holding JSON encoded async task id.
        """
        sanic.log.logger.debug("Handling %s worker command", command.upper())

        command = worker_manager.WorkerCommand[command.upper()]
        task_id = self.ctx.tasks_manager.post_coro(
            self.ctx.asic_manager.handle_command,
            request.ctx.api_key,
            command,
        )

        data = schemas.TaskSubmitted(task_id)
        serialized = schemas.encode(schemas.TaskSubmittedSchema, data)

        return utils.make_api_response(serialized)
