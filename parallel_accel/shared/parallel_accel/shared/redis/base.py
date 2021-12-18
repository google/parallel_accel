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
"""This module defines base class for Redis store communication."""
import abc
import enum
import typing
import redis

from .. import logger


class RedisInstances(enum.IntEnum):
    """Types of Redis databases."""

    JOBS = enum.auto()
    JOBS_IDS = enum.auto()
    WORKERS = enum.auto()


class KeyExistsError(KeyError):
    """Key already exists error."""

    def __init__(self, key: str, obj: str = "object") -> None:
        """Creates KeyExistsError class instance.

        Args:
            key: Key to be created.
            obj: Created value type.
        """
        super().__init__(f'{obj} object already exist for the "{key}" key')


class KeyNotFoundError(KeyError):
    """Key not found error."""

    def __init__(self, key: str, obj: str = "object") -> None:
        """Creates KeyNotFoundError class instance.

        Args:
            key: Requested key.
            obj: Expected value type.
        """
        super().__init__(f'No {obj} object was found for the "{key}" key')


class BaseRedisStore(
    metaclass=abc.ABCMeta
):  # pylint: disable=too-few-public-methods
    """Base class for managing object inside the Redis database."""

    def __init__(
        self, instances: typing.List[RedisInstances], host: str, port: int
    ) -> None:
        """Creates BaseRedisStore class instance.

        Args:
            instances: List of target Redis instances.
            host: Redis hostname.
            port: Redis port.
        """
        self._connections: typing.Dict[RedisInstances, redis.Redis] = {
            x: redis.Redis(
                host=host,
                port=port,
                db=x.value,
                decode_responses=True,
            )
            for x in instances
        }
        self._logger = logger.get_logger(self.__class__.__name__)

    def _has_key(self, key: str, store: RedisInstances) -> bool:
        """Checks whether given key exists in the store.

        Args:
            key: Key to be checked.
            store: Target Redis store.

        Returns:
            True if key exists.
        """
        return bool(self._connections[store].exists(key))
