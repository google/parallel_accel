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
"""This module defines server specific JobsRedisStore implementation."""
import datetime
from typing import Dict
import uuid
import redis

# pylint: disable=wrong-import-order
from parallel_accel.shared import logger, schemas, redis as shared_redis

# pylint: enable=wrong-import-order

from . import helpers


class CreateJobError(redis.exceptions.RedisError):
    """Create a new job error."""


class PushJobIdError(CreateJobError):
    """Push job to jobs queue error."""

    def __init__(self, job_id: str) -> None:
        """Creates PushJobIdError class instance.

        Args:
            job_id: New job id.
        """
        super().__init__(f"Failed to push job {job_id} to the jobs queue")


class SetJobContextError(CreateJobError):
    """Set new job context error."""

    def __init__(
        self, job_id: str, key_type: shared_redis.JobsRedisStore.KeyType
    ) -> None:
        """Creates SetJobContextError class instance.

        Args:
            job_id: New job id.
            key_type: Type of key that failed to be created.
        """
        super().__init__(
            f"Failed to create {key_type.name.upper()} key for {job_id}"
        )


class JobsManager(shared_redis.JobsRedisStore):
    """Jobs store manager.

    This is the server side class for communication with the Redis store, which
    decouples the server from the simulator. It provides methods for creating
    new jobs and retrieving simulation results.
    """

    _KEY_EXPIRE_TIME = datetime.timedelta(days=7)

    def create_job(
        self,
        api_key: str,
        job_type: schemas.JobType,
        context: Dict,
    ) -> str:
        """Creates a new job job data and puts them in the Redis store.

        Args:
            api_key: API key.
            job_type: Submitted job type
            context: Job specific context.

        Returns:
            Unique job id.
        """
        job_id = str(uuid.uuid4())

        logger.context.bind(job_id=job_id, job_type=job_type.name)

        self._logger.debug("Trying to save new job context")

        job = schemas.Job(
            api_key=api_key, context=context, id=job_id, type=job_type
        )
        data = schemas.encode(schemas.JobSchema, job)
        if not self._connections[shared_redis.RedisInstances.JOBS].set(
            JobsManager.KeyType.CONTEXT.key_for(job_id),
            data,
            self._KEY_EXPIRE_TIME,
        ):
            self._logger.error("Failed to create job context")
            raise SetJobContextError(job_id, JobsManager.KeyType.CONTEXT)

        result = schemas.JobResult(job_id, schemas.JobStatus.NOT_STARTED)
        data = schemas.encode(schemas.JobResultSchema, result)
        if not self._connections[shared_redis.RedisInstances.JOBS].set(
            JobsManager.KeyType.STATUS.key_for(job_id),
            data,
            self._KEY_EXPIRE_TIME,
        ):
            self._logger.error("Failed to create job status")
            raise SetJobContextError(job_id, JobsManager.KeyType.STATUS)

        if not self._connections[shared_redis.RedisInstances.JOBS].rpush(
            JobsManager.KeyType.QUEUE.key_for(job_id), data
        ):
            self._logger.error("Failed to create job status queue")
            raise SetJobContextError(job_id, JobsManager.KeyType.QUEUE)

        self._connections[shared_redis.RedisInstances.JOBS].expire(
            JobsManager.KeyType.QUEUE.key_for(job_id), self._KEY_EXPIRE_TIME
        )

        if not self._connections[shared_redis.RedisInstances.JOBS_IDS].rpush(
            api_key, job_id
        ):
            self._logger.error(
                "Failed to push job to the jobs queue", api_key=api_key
            )
            raise PushJobIdError(job_id)

        self._connections[shared_redis.RedisInstances.JOBS_IDS].expire(
            api_key, self._KEY_EXPIRE_TIME
        )

        self._logger.info("Created a new job")

        logger.context.unbind("job_id")

        return job_id

    def flush_job_queue(self, api_key: str) -> None:
        """Flushes job queue.

        Args:
            api_key: API key associated with the queue.
        """
        logger.context.bind(api_key=api_key)

        if not self.has_jobs_queue(api_key):
            self._logger.warning(
                "No job queue is associated with given API key"
            )
            logger.context.unbind("api_key")
            return

        size = self._connections[shared_redis.RedisInstances.JOBS_IDS].llen(
            api_key
        )
        jobs = self._connections[shared_redis.RedisInstances.JOBS_IDS].lrange(
            api_key, 0, size
        )
        self._logger.debug(f"Found {len(jobs)} pending jobs")

        self._logger.debug("Flushing job queue")
        self._connections[shared_redis.RedisInstances.JOBS_IDS].delete(api_key)

        for job in jobs:
            for key_type in JobsManager.KeyType:
                self._connections[shared_redis.RedisInstances.JOBS].delete(
                    key_type.key_for(job)
                )

        self._logger.debug("Done removing pending jobs")

        logger.context.unbind("api_key")

    def is_same_api_key(self, job_id: str, api_key: str) -> bool:
        """Checks if input API key is matching stored job's API key.

        Args:
            job_id: Unique job id.
            api_key: Input API key.

        Returns:
            True if the key matches, false otherwise.

        Throws:
            JobNotFoundError if no object exists for given id.
        """
        job = self._get_job_context(job_id)
        return job.is_same_api_key(api_key)

    def is_same_job_type(self, job_id: str, job_type: schemas.JobType) -> bool:
        """Checks if input job type is matching stored job type.

        Args:
            job_id: Unique job id.
            job_type: Input job type.

        Returns:
            True if the type matches, false otherwise.

        Throws:
            JobNotFoundError if no object exists for given id.
        """
        job = self._get_job_context(job_id)
        return job.type == job_type

    def get_job_queue(self, api_key: str) -> schemas.JobsQueue:
        """Gets jobs queue.

        Args:
            api_key: API key associated with the JobsQueue.

        Returns:
            JobsQueue object.
        """
        logger.context.bind(api_key=api_key)

        self._logger.debug("Looking for matching jobs queue")

        if not self.has_jobs_queue(api_key):
            self._logger.warning(
                "No job queue is associated with given API key"
            )
            queue = []
        else:
            size = self._connections[shared_redis.RedisInstances.JOBS_IDS].llen(
                api_key
            )
            queue = self._connections[
                shared_redis.RedisInstances.JOBS_IDS
            ].lrange(api_key, 0, size)

        self._logger.debug(f"Number of pending jobs: {len(queue)}")

        logger.context.unbind("api_key")

        return schemas.JobsQueue(queue)

    def get_job_status(self, job_id: str) -> schemas.JobStatus:
        """Gets job status.

        Args:
            job_id: Unique job id.

        Returns:
            Current job status.

        Throws:
            JobResultNotFoundError if no object exists for given id.
        """
        logger.context.bind(job_id=job_id)

        self._logger.debug("Looking for a matching JobResult object")

        if not self.has_job_results(job_id):
            self._logger.error("No matching JobResults objects for given id")
            raise shared_redis.JobResultNotFoundError(job_id)

        value = self._connections[shared_redis.RedisInstances.JOBS].get(
            self.KeyType.STATUS.key_for(job_id)
        )
        result: schemas.JobResult = schemas.decode(
            schemas.JobResultSchema, value
        )

        self._logger.debug("Found JobResult object", status=result.status.name)

        logger.context.unbind("job_id")

        return result.status

    def get_job_type(self, job_id: str) -> schemas.JobType:
        """Gets job type.

        Args:
            job_id: Unique job id.

        Returns:
            Job type.

        Throws:
            JobNotFoundError if no object exists for given id.
        """
        return self._get_job_context(job_id).type

    def has_jobs_queue(self, api_key: str) -> bool:
        """Checks whether there is a jobs queue associated with given API key.

        Args:
            api_key: Input API key.

        Returns:
            True if the jobs queue exists.
        """
        return self._has_key(api_key, shared_redis.RedisInstances.JOBS_IDS)

    def has_pending_job(self, api_key: str, job_id: uuid.UUID) -> bool:
        """Checks whether there is a pending job object in the jobs queue.

        Args:
            api_key: API key associated with the jobs queue.
            job_id: Unique job id to be found.

        Returns:
            True if found input job id.
        """
        if not self.has_jobs_queue(api_key):
            return False

        queue = self.get_job_queue(api_key)
        return str(job_id) in queue.ids

    async def subscribe_job_status(
        self, job_id: str, deadline: datetime.timedelta
    ) -> None:
        """Subscribes to job status changes.

        This function polls Redis store for job status changes and yields
        JobResult objects. If the job has not finished within the given
        deadline, the generator yields None and quits.

        Args:
            job_id: Requested task id.
            deadline: Time when the function should stop polling and return.

        Yields:
            JobResult object or None if the polling timed out.

        Throws:
            JobNotFoundError if no job found for given id.
        """
        logger.context.bind(job_id=job_id)

        if not self.has_job(job_id):
            raise shared_redis.jobs.JobNotFoundError(job_id)

        key = JobsManager.KeyType.QUEUE.key_for(job_id)
        status = schemas.JobStatus.NOT_STARTED
        timeout = False

        self._logger.debug("Subscribing to job status changes")
        while (
            status not in (schemas.JobStatus.COMPLETE, schemas.JobStatus.ERROR)
            and not timeout
        ):
            pool_timeout = helpers.compute_blpop_timeout(deadline)
            value = await helpers.blpop(
                self._connections[shared_redis.RedisInstances.JOBS],
                key,
                pool_timeout,
            )

            if value:
                result: schemas.JobResult = schemas.decode(
                    schemas.JobResultSchema, value
                )
                status = result.status
            else:
                self._logger.debug("BLPOP command timed out")
                result = None
                timeout = True

            yield result

        if not timeout:
            self._logger.debug(
                f"Removing {JobsManager.KeyType.CONTEXT.name} key"
            )
            self._connections[shared_redis.RedisInstances.JOBS].delete(
                JobsManager.KeyType.CONTEXT.key_for(job_id)
            )

        logger.context.unbind("job_id")

    def _get_job_context(self, job_id: str) -> schemas.Job:
        """Fetches stored job context.

        Args:
            job_id: Unique job id.

        Returns:
            Job object.

        Throws:
            JobNotFoundError if no object exists for given id.
        """
        logger.context.bind(job_id=job_id)
        self._logger.debug("Looking for a matching Job object")

        if not self.has_job(job_id):
            self._logger.error("No matching Job objects for given id")
            raise shared_redis.JobNotFoundError(job_id)

        self._logger.debug("Found matching Job object")
        logger.context.unbind("job_id")

        value = self._connections[shared_redis.RedisInstances.JOBS].get(
            JobsManager.KeyType.CONTEXT.key_for(job_id)
        )
        return schemas.decode(schemas.JobSchema, value)
