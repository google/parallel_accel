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
"""Blueprint utility functions"""
import asyncio
import functools
import http
import typing
import sanic

RequestHandler = typing.Callable[
    [sanic.request.Request, typing.Any], sanic.response.HTTPResponse
]


class StreamHeartbeat:
    """Helper class that periodically sends heartbeat signal to the client.

    During e2e tests we have noticed the long polling streaming requests fail
    with `requests.exceptions.ChunkedEncodingError`. It turns out we need to
    send a hearbeat signal to the client once in a while to keep the connection
    active.

    This is simple wrapper that sleeps for certian period of time and writes
    "\r\n" to the streaming response.
    """

    def __init__(self, response: sanic.response.StreamingHTTPResponse) -> None:
        """Creates HeartbeatTask class instance.

        Args:
            response: Target streaming HTTP response for sending heartbeat
            signals.
        """
        self._response = response
        self._running = False
        self._task: asyncio.Task = None

    def start(self, app: sanic.Sanic) -> None:
        """Starts stream heartbeat.

        Args:
            app: Reference to Sanic application. Each blueprint unit test
            registers a new Sanic application, thus we cannot retrive it using
            `sanic.Sanic.get_app()` without specificing application name.
        """
        sanic.log.logger.debug("Starting stream heartbeat")

        self._running = True
        self._task = app.loop.create_task(self._run())

    def stop(self) -> None:
        """Stops stream heartbeat."""
        sanic.log.logger.debug("Stopping stream heartbeat")

        self._running = False
        self._task.cancel()

    async def _run(self) -> None:
        """Heartbeat task main loop."""
        while self._running:
            await asyncio.sleep(15.0)
            if not self._running or not self._is_transport_open():
                # The stream might get closed while we were waiting for the next
                # heatbeat signal. Do not write response and just quit.
                break

            sanic.log.logger.debug("Sending heartbeat signal to the stream")
            await self._response.write("\r\n")

    def _is_transport_open(self) -> bool:
        """Checks if TCP transport is open.

        Returns:
            True if open.
        """
        return (
            self._response.stream.request.transport
            and not self._response.stream.request.transport.is_closing()
        )


def make_api_response(
    body: str,
    status: http.HTTPStatus = http.HTTPStatus.OK,
    headers: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> sanic.response.HTTPResponse:
    """Creates API response.

    Args:
        body: JSON encoded response body.
        status: HTTP status code.
        headers: Response headers.

    Returns:
        HTTPResponse object.
    """
    return sanic.response.text(body, status, headers, "application/json")


def wrap_handler(handler: RequestHandler) -> typing.Callable:
    """Wraps sanic.Blueprint request handler.

    Sanic Blueprints do not support class instance methods as handlers, so we
    need alternate way to register views in the blueprint __init__ function.
    This function mimics blueprint decorator behavior by wrapping our handler as
    a partial object.

    Args:
        handler: Blueprint instance method, a callable object. The object should
          take a `sanic.request.Request` as its first argument, and return a
          `sanic.response.HTTPResponse` object.

    Returns:
        Decorated instance method.
    """
    return functools.wraps(handler)(functools.partial(handler))
