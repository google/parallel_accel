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
"""This module defines class for managing Workers objects in the Redis store."""
import os
import time
import typing
import uuid

from ..schemas import (
    Worker,
    WorkerInternal,
    WorkerInternalSchema,
    WorkerState,
    encode,
    decode,
)
from .base import (
    BaseRedisStore,
    KeyNotFoundError,
    RedisInstances,
)


class WorkerNotFoundError(KeyNotFoundError):
    """Worker object not exist error."""

    def __init__(self, api_key: str) -> None:
        """Creates WorkerNotFoundError class instance.

        Args:
            api_key: API key associated with the worker.
        """
        super().__init__(api_key, Worker.__name__)


class WorkersRedisStore(BaseRedisStore):
    """Manages Worker objects in the Redis store."""

    def __init__(
        self,
        host: str = os.environ.get("REDISHOST", "localhost"),
        port: int = 6379,
    ) -> None:
        """Creates WorkersRedisStore class instance.

        Args:
            host: Redis hostname.
            port: Redis port.
        """
        super().__init__([RedisInstances.WORKERS], host, port)

    def get_workers_ids(self) -> typing.List[str]:
        """Gets a list of all workers keys."""
        return self._connections[RedisInstances.WORKERS].keys()

    def get_worker(self, api_key: str) -> WorkerInternal:
        """Gets worker status.

        Args:
            api_key: API key associated with the ASIC worker.

        Throws:
            KeyError if the worker object does not exist.
        """
        if not self.has_worker(api_key):
            self._logger.error(
                "No matching Worker object for given API key", api_key=api_key
            )
            raise WorkerNotFoundError(api_key)

        value = self._connections[RedisInstances.WORKERS].get(api_key)
        return decode(WorkerInternalSchema, typing.cast(str, value))

    def has_worker(self, api_key: str) -> bool:
        """Checks whether there is a ASIC worker for given API key.

        Args:
            api_key: Input API key.

        Returns:
            True if there is worker associated with the API key.
        """
        return self._has_key(api_key, RedisInstances.WORKERS)

    def set_booting(self, api_key: str) -> None:
        """Sets worker BOOTING state

        Args:
            api_key: API key associated with the ASIC worker.
        """
        self._create_if_not_exists(api_key)
        self._update_state(api_key, WorkerState.BOOTING)

    def set_error(self, api_key: str, error: str) -> None:
        """Sets worker ERROR state

        Args:
            api_key: API key associated with the ASIC worker.
            error: Error message.
        """
        self._create_if_not_exists(api_key)
        self._update_state(api_key, WorkerState.ERROR, error=error)

    def set_idle(self, api_key: str) -> None:
        """Sets worker IDLE state

        Args:
            api_key: API key associated with the ASIC worker.
        """
        self._create_if_not_exists(api_key)
        self._update_state(api_key, WorkerState.IDLE)

    def set_offline(self, api_key: str) -> None:
        """Sets worker OFFLINE state

        Args:
            api_key: API key associated with the ASIC worker.
        """
        self._create_if_not_exists(api_key)
        self._update_state(api_key, WorkerState.OFFLINE)

    def set_processing_job(self, api_key: str, job_id: uuid.UUID) -> None:
        """Sets worker PROCESSING_JOB state

        Args:
            api_key: API key associated with the ASIC worker.
            job_id: Currently processed job id.
        """
        self._create_if_not_exists(api_key)
        self._update_state(api_key, WorkerState.PROCESSING_JOB, job_id=job_id)

    def set_shutting_down(self, api_key: str) -> None:
        """Sets worker SHUTTING_DOWN state

        Args:
            api_key: API key associated with the ASIC worker.
            job_id: Currently processed job id.
        """
        self._create_if_not_exists(api_key)
        self._update_state(api_key, WorkerState.SHUTTING_DOWN)

    def _create_if_not_exists(self, api_key: str) -> None:
        """Checks if worker object exists for given API key and creates if
        object does not exist.

        Args:
            api_key: API key to be associated with the worker.
        """
        if self.has_worker(api_key):
            return

        self._logger.debug("Creating worker object", api_key=api_key)
        worker = Worker(state=WorkerState.OFFLINE)
        self._set_worker(api_key, worker)

    def _update_state(
        self,
        api_key: str,
        state: WorkerState,
        job_id: typing.Optional[uuid.UUID] = None,
        error: typing.Optional[str] = None,
    ) -> None:
        """Updates Worker object status.

        Args:
            api_key: API key associated with the ASIC worker.
            state: New worker state.
            job_id: Currently processed job id.
            error: Error message.

        Throws:
            KeyError if the worker object does not exist.
        """
        worker = self.get_worker(api_key)

        if worker.state == state:
            self._logger.debug(
                "Worker already in given state",
                api_key=api_key,
                state=state.name,
            )
            return

        self._logger.debug(
            "Updating worker state",
            error=error,
            job_id=str(job_id),
            state=state.name,
            api_key=api_key,
        )

        worker.state = state
        worker.error = error
        worker.job_id = job_id
        if worker.state == WorkerState.PROCESSING_JOB:
            worker.job_timestamp = int(time.time())

        self._set_worker(api_key, worker)

    def _set_worker(self, api_key: str, worker: Worker) -> None:
        """Sets api_key -> worker in the Worker connection.

        Args:
            api_key: API key associated with the ASIC worker.
            worker: Worker object to set.
        """
        value = encode(WorkerInternalSchema, worker)
        self._connections[RedisInstances.WORKERS].set(api_key, value)
