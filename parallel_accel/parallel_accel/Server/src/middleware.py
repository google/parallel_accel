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
"""Sanic application middleware components."""
import datetime
import time
import sanic


def calculate_request_deadline(request: sanic.request.Request) -> None:
    """Calculates HTTP request deadline.

    Both Cloud Backend and ESPv2 application are configured to kill HTTP
    connection after 620 seconds (hard limit). To keep some error margin, our
    application will try to complete the request within 600 seconds.

    The deadline property is only used for streaming endpoints:
        - job result stream
        - task status stream

    Args:
        request: Incoming HTTP Request object.
    """
    if not request.path.endswith("/stream"):
        # The deadline context is only used for streaming endpoints.
        return

    request.ctx.deadline = datetime.timedelta(seconds=time.time() + 600)


def extract_api_key(request: sanic.request.Request) -> None:
    """Verifies if API token is present in the reuqest headers and extracts it's
    value to the request context.

    Args:
        request: Incoming HTTP Request object.

    Throws:
        sanic.exceptions.Unauthorized if the API key is missing.
    """
    api_key = request.headers.get("x-api-key", None)
    if not api_key:
        raise sanic.exceptions.Unauthorized("Missing API key")

    request.ctx.api_key = api_key
