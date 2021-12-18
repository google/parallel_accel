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
"""ParallelAccel marsobj_fnllow schemas that are being used for exchanging data between API
service and ASIC workers.
"""
import dataclasses
import re
from typing import Any, Union
import uuid
import marshmallow_dataclass
import marshmallow_enum

from parallel_accel.shared import utils
from . import external


@dataclasses.dataclass
class Job:
    """Start a new job request data.

    Properties:
        api_key: SHA1 encoded API key
        context: Job specific context.
        id: Unique job id.
        type: Job type.
    """

    api_key: str
    id: uuid.UUID  # pylint: disable=invalid-name
    type: external.JobType = dataclasses.field(
        metadata={
            "marshmallow_field": marshmallow_enum.EnumField(
                external.JobType, by_value=True
            )
        }
    )
    context: Any

    def __post_init__(self) -> None:
        """See base class documentation."""
        if not re.match(r"^[a-fA-F0-9]{40}$", self.api_key):
            # Hash input API key using SHA1
            self.api_key = utils.sha1(self.api_key)

    def is_same_api_key(self, api_key: str) -> bool:
        """Checks if the API keys are matching.

        Args:
            api_key: Input API key.

        Returns:
            True if key matches, false otherwise.
        """
        return self.api_key == utils.sha1(api_key)


@dataclasses.dataclass
class ExpectationJob(Job):
    """Start a new expectation job request data.

    Properties:
        context: Job specific context.
        id: Unique job id.
        type: Job type.
    """

    context: Union[
        external.ExpectationBatchJobContext,
        external.ExpectationJobContext,
        external.ExpectationSweepJobContext,
    ]


@dataclasses.dataclass
class SampleJob(Job):
    """Start a new sample job request data.

    Properties:
        context: Job specific context.
        id: Unique job id.
        type: Job type.
    """

    context: Union[
        external.SampleBatchJobContext,
        external.SampleJobContext,
        external.SampleSweepJobContext,
    ]


@dataclasses.dataclass
class NoisyExpectationJob(Job):
    """Start a new noisy expectation job request data.

    Properties:
        context: Job specific context.
        id: Unique job id.
        type: Job type.
    """

    context: external.NoisyExpectationJobContext


@dataclasses.dataclass
class Worker(external.Worker):
    """Current status of the ASIC worker.

    Properties:
        error: Error details.
        job_id: Currently processed job id.
        job_timestamp: Last job timestamp.
        state: Worker state.
    """

    job_timestamp: int = dataclasses.field(default=0)


(
    ExpectationJobSchema,
    JobSchema,
    NoisyExpectationJobSchema,
    SampleJobSchema,
    WorkerSchema
) = tuple(
    marshmallow_dataclass.class_schema(x)()
    for x in (
        ExpectationJob,
        Job,
        NoisyExpectationJob,
        SampleJob,
        Worker
    )
)
