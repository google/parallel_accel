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
"""Jobs Manager for the ASIC Simulator."""
import typing
from typing import Any
import uuid
import marshmallow

from parallel_accel.shared.redis import (
    JobNotFoundError,
    JobResultNotFoundError,
    JobsRedisStore,
    RedisInstances,
)
from parallel_accel.shared.schemas import (
    ExpectationJobSchema,
    Job,
    JobProgress,
    JobResultSchema,
    JobSchema,
    JobStatus,
    JobType,
    SampleJobSchema,
    encode,
    decode,
)


class UnsupportedJobTypeError(KeyError):
    """Unsupported JobType error."""

    def __init__(self, job_type: str) -> None:
        """Creates UnsupportedJobTypeError class instance.

        Args:
            job_type: Requested job type.
        """
        super().__init__(f"Unsupported JobType: {job_type}")


class SimJobsManager(JobsRedisStore):
    """Simulation Jobs Store Manager.

    This is the Simulator side class for managing serialization and interaction
    with the Redis Stores. It provides methods for getting jobs, updating their
    status and returning results of processing a job.

    All schema level serialization is handled by the methods in this class.
    LinearAlgebra level serialization not defined is is expected to be handled
    before being passed to this class's methods.
    """

    def get_next_job(self, api_key: str) -> typing.Optional[Job]:
        """Get the next job from the store and deserialize
        This method will block until a job is available. The job will not be
        removed from the job queue.

        Args:
            api_key: api_key to query for waiting jobs.

        Returns:
            Next job to waiting to be processed by api_key.
        """
        context_schemas = {
            JobType.SAMPLE: SampleJobSchema,
            JobType.EXPECTATION: ExpectationJobSchema,
        }

        r_list = self._connections[RedisInstances.JOBS_IDS].blpop(
            api_key, timeout=5
        )
        if not r_list:
            return None

        _, job_id = r_list
        self._connections[RedisInstances.JOBS_IDS].lpush(api_key, job_id)
        if not self.has_job(job_id):
            raise JobNotFoundError(job_id)

        job_raw = self._connections[RedisInstances.JOBS].get(
            SimJobsManager.KeyType.CONTEXT.key_for(job_id)
        )
        job = decode(JobSchema, job_raw)
        return decode(context_schemas[job.type], job_raw)

    def clear_next_job(self, api_key: str) -> None:
        """Clear the next job_id from the job queue and remove the corresponding
        entry in JOBS.

        Args:
            api_key: api_key to clear from.
        """
        job_id = self._connections[RedisInstances.JOBS_IDS].lpop(api_key)
        if not job_id:
            return

        key = SimJobsManager.KeyType.STATUS.key_for(job_id)
        if self._has_key(key, RedisInstances.JOBS):
            self._connections[RedisInstances.JOBS].delete(key)

    def set_job_complete(
        self,
        job_id: uuid.UUID,
        result: typing.Any,
        schema: marshmallow.Schema,
    ) -> None:
        """Submit results and set a job as complete.

        Args:
            job_id: The non-serial job_id
            result: A pre-serialized job result
            schema: Schema to be used for encoding job results.
        """
        context = {"result": result, "status": JobStatus.COMPLETE}
        self._update_job(job_id=job_id, context=context, encode_schema=schema)

    def set_job_error(self, job_id: uuid.UUID, message: str) -> None:
        """Set a job as having an error status.

        Args:
            job_id: JobId to set to error status.
            message: Detailed error message.
        """
        context = {"error_message": message, "status": JobStatus.ERROR}
        self._update_job(job_id=job_id, context=context)

    def set_job_in_progress(self, job_id: uuid.UUID) -> None:
        """Set a job as in progress.

        Args:
            job_id: JobId to set to error status.
        """
        context = {"progress": JobProgress(), "status": JobStatus.IN_PROGRESS}
        self._update_job(job_id=job_id, context=context)

    def update_job_progress(
        self, job_id: uuid.UUID, progress: JobProgress
    ) -> None:
        """Update the JobProgress field of a job.

        Args:
            job_id: JobId to set to error status.
            progress: JobProgress object representing current progress.
        """
        context = {"progress": progress}
        self._update_job(job_id=job_id, context=context)

    def _update_job(
        self,
        job_id: uuid.UUID,
        context: typing.Dict[str, Any],
        encode_schema: marshmallow.Schema = JobResultSchema,
    ) -> None:
        """Updates the fields of a job result with the field supplied in a
        context dict. An empty context dict results in an unchanged job.

        Args:
            job_id: Target job id.
            context: Dict object with new JobResult properties.
            encode_schema: Schema to be used for encoding job results.

        Throws:
            JobResultNotFoundError if no matching job is present in the object
            store.
        """

        job_id = str(job_id)

        if not self.has_job_results(job_id):
            raise JobResultNotFoundError(job_id)

        result_raw = self._connections[RedisInstances.JOBS].get(
            SimJobsManager.KeyType.STATUS.key_for(job_id)
        )
        result = decode(JobResultSchema, result_raw)
        for key, value in context.items():
            setattr(result, key, value)

        result_raw = encode(encode_schema, result)
        for handler, key_type in (
            (
                self._connections[RedisInstances.JOBS].set,
                SimJobsManager.KeyType.STATUS,
            ),
            (
                self._connections[RedisInstances.JOBS].rpush,
                SimJobsManager.KeyType.QUEUE,
            ),
        ):
            handler(key_type.key_for(job_id), result_raw)
