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
"""Redis store helper functions"""
import datetime
import functools
import time
from typing import Optional
import sanic
import redis


async def blpop(
    connection: redis.Redis, key: str, timeout: float
) -> Optional[str]:
    """Asynchronous BLPOP command.

    The Redis.blpop is IO blocking function, so we need a wrappere that can be
    used together with the Sanic event loop.

    Args:
        connection: Redis connection to be used for BLPOP command.
        key: Key to be polled.
        timeout: Command timeout.

    Returns:
        Value stored under given key or None if polling timed out.
    """
    partial = functools.partial(connection.blpop, key, timeout=timeout)
    loop = sanic.app.get_event_loop()

    result = await loop.run_in_executor(None, partial)

    if not result:
        return None

    _, value = result
    return value


def compute_blpop_timeout(deadline: datetime.timedelta) -> float:
    """Computes BLPOP command timeout using context deadline.

    Args:
        deadline: Request deadline.

    Returns:
        BLPOP command timeout.
    """
    return (deadline - datetime.timedelta(seconds=time.time())).total_seconds()
