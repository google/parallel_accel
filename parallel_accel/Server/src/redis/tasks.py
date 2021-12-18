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
"""This module defines TasksRedisStore component."""
import datetime
import enum
import os
from typing import Optional
import uuid

# pylint: disable=wrong-import-order
from parallel_accel.shared import logger, schemas, redis as shared_redis

# pylint: enable=wrong-import-order

from . import helpers


class TaskExistError(shared_redis.KeyExistsError):
    """Duplicate Task object error."""

    def __init__(self, task_id: uuid.UUID) -> None:
        """Creates TaskExistError class instance.

        Args:
            task_id: Requested task id.
        """
        super().__init__(str(task_id))


class TaskNotFoundError(shared_redis.KeyNotFoundError):
    """TaskStatus object not exist error."""

    def __init__(self, task_id: str) -> None:
        """Creates TaskNotFoundError class instance.

        Args:
            task_id: Requested task id.
        """
        super().__init__(str(task_id), schemas.TaskStatus.__name__)


class RedisInstances(enum.IntEnum):
    """See base class documentation."""

    TASKS = 9


class TasksRedisStore(shared_redis.base.BaseRedisStore):
    """Manages TasksStatus objects in the Redis store."""

    _KEY_EXPIRE_TIME = datetime.timedelta(days=1)

    class KeyType(enum.Enum):
        """Key types in TASKS store.

        Each asynchronous task has two unique keys in the TASKS Redis store:
            - MARKER key is dummy placeholder that allows API service to check
              whether the task exists.
            - QUEUE key is used for pushing new `schemas.TaskStatus` objects and
              streaming task status server side events.
        """

        MARKER = "marker"
        QUEUE = "queue"

        def key_for(self, task_id: uuid.UUID) -> str:
            """Gets key for given job id.

            Args:
                job_id: Unique job id.

            Returns:
                Formatted key.
            """
            return f"{str(task_id)}.{self.value}"

    def __init__(
        self,
        host: str = os.environ.get("REDISHOST", "localhost"),
        port: int = 6379,
    ) -> None:
        """Creates TasksRedisStore class instance.

        Args:
            host: Redis hostname.
            port: Redis port.
        """
        super().__init__([RedisInstances.TASKS], host, port)

    def create_task(self, task_id: uuid.UUID) -> None:
        """Creates a new task object.

        Args:
            task_id: Unique task id.
        """
        if self.has_task(task_id):
            self._logger.error(
                "TaskStatus already exists in the Redis store",
                task_id=str(task_id),
            )
            raise TaskExistError(task_id)

        self._connections[RedisInstances.TASKS].set(
            TasksRedisStore.KeyType.MARKER.key_for(task_id),
            TasksRedisStore.KeyType.MARKER.name.lower(),
            self._KEY_EXPIRE_TIME,
        )

        status = self._create_status(schemas.TaskState.PENDING)
        self._append(task_id, status)

        self._logger.debug(
            "Created a new TaskStatus object in the Redis store",
            task_id=str(task_id),
        )

        return task_id

    def has_task(self, task_id: uuid.UUID) -> bool:
        """Checks whether there is any task object for given id.

        Args:
            task_id: Requested task id.

        Returns:
            True if found input key.
        """
        return self._has_key(
            TasksRedisStore.KeyType.MARKER.key_for(task_id),
            RedisInstances.TASKS,
        )

    def set_completed(
        self, task_id: uuid.UUID, success: bool, error: Optional[str]
    ) -> None:
        """Sets task status DONE.

        Args:
            task_id: Unique task id.
            success: Indicates if task completed successfully.
            error: Task exception message.
        """
        status = self._create_status(schemas.TaskState.DONE, success, error)
        self._append(task_id, status, datetime.timedelta(minutes=15))

    def set_running(self, task_id: uuid.UUID) -> None:
        """Sets task status RUNNING.

        Args:
            task_id: Unique task id.
        """
        status = self._create_status(schemas.TaskState.RUNNING)
        self._append(task_id, status)

    async def subscribe(
        self, task_id: uuid.UUID, deadline: datetime.timedelta
    ) -> None:
        """Subscribes to task status changes.

        This function polls Redis store for task status changes and yields
        TaskStatus objects. If the task has not finished within the given
        deadline, the generator yields None and quits.

        Args:
            task_id: Requested task id.
            deadline: Time when the function should stop polling and return.

        Yields:
            TaskStatus object or None if the polling timed out.

        Throws:
            TaskNotFoundError if no task found for given id.
        """
        logger.context.bind(task_id=str(task_id))

        if not self.has_task(task_id):
            raise TaskNotFoundError(task_id)

        key = TasksRedisStore.KeyType.QUEUE.key_for(task_id)
        state = schemas.TaskState.PENDING
        timeout = False

        self._logger.debug("Subscribing to task status changes")
        while state is not schemas.TaskState.DONE and not timeout:
            pool_timeout = helpers.compute_blpop_timeout(deadline)
            value = await helpers.blpop(
                self._connections[RedisInstances.TASKS],
                key,
                pool_timeout,
            )

            if value:
                status = schemas.decode(schemas.TaskStatusSchema, value)
                state = status.state
            else:
                self._logger.debug("BLPOP command timed out")
                status = None
                timeout = True

            yield status

        if not timeout:
            self._logger.debug(
                f"Removing {TasksRedisStore.KeyType.MARKER.name} key"
            )
            self._connections[RedisInstances.TASKS].delete(
                TasksRedisStore.KeyType.MARKER.key_for(task_id)
            )

        logger.context.unbind("task_id")

    def _append(
        self,
        task_id: uuid.UUID,
        value: str,
        expire_time: datetime.timedelta = _KEY_EXPIRE_TIME,
    ) -> None:
        """Appends a new value to specific task list.

        Args:
            task_id: Unique task id.
            value: Value to be appended to the list.
            expire_time: Key expiration time.
        """
        self._logger.debug(
            "Pushing new TaskStatus object", task_id=str(task_id)
        )
        self._connections[RedisInstances.TASKS].rpush(
            TasksRedisStore.KeyType.QUEUE.key_for(task_id), value
        )
        self._connections[RedisInstances.TASKS].expire(
            TasksRedisStore.KeyType.QUEUE.key_for(task_id), expire_time
        )

    @staticmethod
    def _create_status(
        state: schemas.TaskState,
        success: Optional[bool] = None,
        error: Optional[str] = None,
    ) -> str:
        """Creates serialized TaskStatus object.

        Args:
            state: Current task state.
            success: Indicates if task completed successfully.
            error: Task exception message.

        Returns:
            JSON encoded TaskStatus object.
        """
        obj = schemas.TaskStatus(state, error, success)
        return schemas.encode(schemas.TaskStatusSchema, obj)
