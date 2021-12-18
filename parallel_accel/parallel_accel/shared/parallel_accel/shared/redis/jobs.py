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
"""This module defines class for managing Jobs related objects in the Redis
store."""
import enum
import os

from ..schemas import Job, JobResult
from .base import BaseRedisStore, KeyNotFoundError, RedisInstances


class JobNotFoundError(KeyNotFoundError):
    """Job object not exist error."""

    def __init__(self, job_id: str) -> None:
        """Creates JobNotFoundError class instance.

        Args:
            job_id: Requested job id.
        """
        super().__init__(job_id, Job.__name__)


class JobResultNotFoundError(KeyNotFoundError):
    """JobResult object not exist error."""

    def __init__(self, job_id: str) -> None:
        """Creates JobResultNotFoundError class instance.

        Args:
            job_id: Requested job id.
        """
        super().__init__(job_id, JobResult.__name__)


class JobsRedisStore(BaseRedisStore):
    """Jobs store manager.

    This is base class for managing job related data in the Redis store. It is
    meant to be subclassed and provide service specific functionality.
    """

    class KeyType(enum.Enum):
        """Key types in JOBS store.

        Each simulation job has two unique keys in the JOBS Redis store:
            - CONTEXT key contains simulation job data (i.e. acyclic_graph, param
              resolvers)
            - STATUS key is used by simulator application to track current
              state of simulation job.
            - QUEUE key is used for pushing new `schemas.JobResults` objects and
              streaming job status server side events.
        """

        CONTEXT = "context"
        STATUS = "status"
        QUEUE = "queue"

        def key_for(self, job_id: str) -> str:
            """Gets key for given job id.

            Args:
                job_id: Unique job id.

            Returns:
                Formatted key.
            """
            return f"{job_id}.{self.value}"

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
        super().__init__(
            [RedisInstances.JOBS, RedisInstances.JOBS_IDS], host, port
        )

    def has_job(self, job_id: str) -> bool:
        """Checks whether there is a `schemas.Job` object matching given id.

        Args:
            job_id: Unique job id.

        Returns:
            True if there is job context object.
        """
        return self._has_key(
            JobsRedisStore.KeyType.CONTEXT.key_for(job_id), RedisInstances.JOBS
        )

    def has_job_results(self, job_id: str) -> bool:
        """Checks whether there is a `schemas.JobResults` object matching given
        id.

        Args:
            job_id: Unique job id.

        Returns:
            True if there is computation results object.
        """
        return self._has_key(
            JobsRedisStore.KeyType.STATUS.key_for(job_id),
            RedisInstances.JOBS,
        )
