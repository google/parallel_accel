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
"""This module defines Sanic application wrapper

We are subclassing Sanic application to register custom error handler during the
class instance initialization.

The default Sanic error handler is sending HTML response with details about the
encountered error. Since this service only acts as the REST API for the website
we do not need those details.
"""
import http
import traceback
import typing
import marshmallow
import sanic

from parallel_accel.shared import schemas

from . import middleware


API_ERROR_MESSAGES: typing.Dict[http.HTTPStatus, str] = {
    http.HTTPStatus.BAD_REQUEST: "Invalid or missing body request",
    http.HTTPStatus.FORBIDDEN: "Missing or invalid API key",
    http.HTTPStatus.UNAUTHORIZED: "Unauthorized API request",
    http.HTTPStatus.NOT_FOUND: "Endpoint not found",
    http.HTTPStatus.INTERNAL_SERVER_ERROR: "Internal Server Error",
}

EXCEPTION_STATUS_MAP: typing.Dict[Exception, http.HTTPStatus] = {
    marshmallow.exceptions.ValidationError: http.HTTPStatus.BAD_REQUEST,
    sanic.exceptions.HeaderNotFound: http.HTTPStatus.BAD_REQUEST,
    sanic.exceptions.InvalidUsage: http.HTTPStatus.BAD_REQUEST,
    sanic.exceptions.Forbidden: http.HTTPStatus.FORBIDDEN,
    sanic.exceptions.Unauthorized: http.HTTPStatus.UNAUTHORIZED,
    sanic.exceptions.NotFound: http.HTTPStatus.NOT_FOUND,
}


class Application(sanic.Sanic):
    """Sanic application wrapper"""

    def __init__(
        self,
        blueprints: typing.List[sanic.Blueprint],
        api_version: str = "1",
        app_name: str = "WORKNIGAREA-API",
    ) -> None:
        """Creates Application class instance.

        Args:
            blueprints: List of API blueprints.
            api_version: API version.
            name: Application name.
        """
        app_name = f"{app_name}-V{api_version}"
        super().__init__(name=app_name)

        # Register error handler before declaring endpoints
        self.error_handler.add(Exception, self.__handle_exception)

        # Register /health endpoint
        self.add_route(self.handle_health_check, "/health")

        # Register API blueprints
        for blueprint in blueprints:
            blueprint.middleware(middleware.extract_api_key, "request")
            blueprint.middleware(
                middleware.calculate_request_deadline, "request"
            )

        blueprints = sanic.Blueprint.group(
            blueprints, url_prefix=f"/api/v{api_version}"
        )
        self.blueprint(blueprints)

    async def handle_health_check(
        self,
        _: sanic.request.Request,
    ) -> sanic.response.HTTPResponse:
        """Handles GET /health endpoint request.

        Generates empty response to satisfy readiness probe check.

        Args:
            request: Incoming HTTP request.

        Returns:
            HTTPResponse object.
        """
        # Health Check requires HTTP 200 OK status code
        return sanic.response.empty(status=http.HTTPStatus.OK)

    @staticmethod
    def __handle_exception(
        _: sanic.request.Request, exception: Exception
    ) -> sanic.response.HTTPResponse:
        """Application error handler.

        Args:
            request: Incoming HTTP request object.
            exception: Thrown exception.

        Returns:
            JSON object with detailed API error message.
        """
        status = EXCEPTION_STATUS_MAP.get(
            type(exception), http.HTTPStatus.INTERNAL_SERVER_ERROR
        )
        if status == http.HTTPStatus.INTERNAL_SERVER_ERROR:
            sanic.log.logger.error("Server error: %s", exception)
            traceback.print_exc()

        data = schemas.APIError(status, API_ERROR_MESSAGES[status])
        return sanic.response.text(
            schemas.encode(schemas.APIErrorSchema, data),
            status,
            content_type="application/json",
        )
