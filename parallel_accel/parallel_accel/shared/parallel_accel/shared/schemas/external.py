# Copyright 2021 The ParallelAccel Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This module provides types definitions."""

import dataclasses
import enum
import json
import time
from typing import Any, Dict, List, Optional, Union
import uuid
import linear_algebra
import marshmallow
import marshmallow_dataclass
import marshmallow_enum


#####################################
# Utility functions                 #
#####################################


def decode(
    schema: marshmallow.Schema, data: str, **kwargs
) -> dataclasses.dataclass:
    """Decodes input string using provided schema.

    Args:
        schema: Schema to be used for deserialization.
        data: JSON-encoded data to be deserialized.
        **kwargs: Extra keyworded arguments to be passed to
            `marshmallow.Schemas.loads` method.

    Returns:
        Deserialized `dataclasses.dataclass` object.
    """
    return schema.loads(data, **kwargs)


def encode(
    schema: marshmallow.Schema, data: dataclasses.dataclass, **kwargs
) -> str:
    """Encodes input data using provided schema.

    Args:
        schema: Schema to be used for serialization.
        data: Dataclass object to be serialized.
        **kwargs: Extra keyworded arguments to be passed to
            `marshmallow.Schemas.dumps` method.

    Returns:
        JSON-encoded serialized data.
    """
    return schema.dumps(data, separators=(",", ":"), **kwargs)


#####################################
# Types aliases                     #
#####################################


OperatorsType = List[linear_algebra.ops.ProbBasisAxisSum]


#####################################
# marshmallow helpers               #
#####################################

_SerializedLinearAlgebraObject = Dict[str, Any]
_SerializedProbBasisAxisSums = List[List[Dict[str, Any]]]

# `linear_algebra` offers only functions to dump and load objects from the JSON encoded
# string, and does not support builtin dict objects. When we call json.dumps()
# over already JSON encoded string, all quotation marks and brackets are
# prefixed with the backslash. Instead, we can convert JSON object to the dict
# type and reduce serialized object size.


def _deserialize_linear_algebra_object(data: _SerializedLinearAlgebraObject) -> Any:
    """Deserializes linear_algebra object from dict type.

    Since `linear_algebra` does not provide function to load objects from builtin dict
    objects, we need some workaround here: first we dump the dict object into
    JSON encoded string, then parse them into `linear_algebra` object.

    Args:
        data: Dict encoded linear_algebra object.

    Returns:
        Deserialized linear_algebra object.
    """
    return linear_algebra.read_json(json_text=json.dumps(data))


def _serialize_linear_algebra_object(obj: Any) -> _SerializedLinearAlgebraObject:
    """Serializes linear_algebra object to dict type.

    Since `linear_algebra` does not provide function to dump objects into builtin dict
    objects, we need some workaround here: first we dump the `linear_algebra` object into
    JSON encoded string, then parsing them into dict object.

    Args:
        data: linear_algebra object to be encoded.

    Returns:
        Serialized linear_algebra object.
    """
    return json.loads(linear_algebra.to_json(obj))


class _LinearAlgebraField(marshmallow.fields.Field):
    """`marshmallow.fields.Field` that serializes and deserializes `linear_algebra` type
    object."""

    def _serialize(
        self, value: Any, *_args, **_kwargs
    ) -> _SerializedLinearAlgebraObject:
        """See base class documentation."""
        return _serialize_linear_algebra_object(value)

    def _deserialize(
        self, value: _SerializedLinearAlgebraObject, *_args, **_kwargs
    ) -> Any:
        """See base class documentation."""
        try:
            return _deserialize_linear_algebra_object(value)
        except json.JSONDecodeError as ex:
            raise marshmallow.ValidationError("Not a JSON object") from ex


class _OperatorsField(marshmallow.fields.Field):
    """`marshmallow.fields.Field` that serializes and deserializes
    `linear_algebra.ProbBasisAxisSum` operators."""

    def _serialize(
        self, value: OperatorsType, _attr, _obj, **kwargs
    ) -> _SerializedProbBasisAxisSums:
        """See base class documentation."""
        if not isinstance(value, list):
            value = [value]

        return [[_serialize_linear_algebra_object(term) for term in op] for op in value]

    def _deserialize(
        self, value: _SerializedProbBasisAxisSums, _attr, _obj, **kwargs
    ) -> OperatorsType:
        """See base class documentation."""
        try:
            return [
                sum([_deserialize_linear_algebra_object(term) for term in op])
                for op in value
            ]
        except json.JSONDecodeError as ex:
            raise marshmallow.ValidationError("Not a JSON object") from ex


Graph = marshmallow_dataclass.NewType(
    "Graph", linear_algebra.Graph, field=_LinearAlgebraField
)
Operators = marshmallow_dataclass.NewType(
    "Operators", OperatorsType, field=_OperatorsField
)
ParamResolver = marshmallow_dataclass.NewType(
    "ParamResolver", linear_algebra.ParamResolver, field=_LinearAlgebraField
)
Result = marshmallow_dataclass.NewType("Result", linear_algebra.Result, field=_LinearAlgebraField)
Sweepable = marshmallow_dataclass.NewType(
    "Sweepable", linear_algebra.study.Sweepable, field=_LinearAlgebraField
)


#####################################
# Server side events                #
#####################################


@dataclasses.dataclass
class ServerSideEvent:
    """Base class for server side event.

    Both `event` and `timestamp` fields are auto-populated if using default
    values:

    - `event` is set to the class name
    - `timestamp` is set to the current time

    Attributes:
        id: Event unique id.
        data: Event payload.
        event: Event name.
        timestamp: Event timestamp (in UNIX seconds).
    """

    id: uuid.UUID  # pylint: disable=invalid-name
    data: Any
    event: str = dataclasses.field(default="")
    timestamp: int = dataclasses.field(default=0)

    def __post_init__(self) -> None:
        if self.event == "":
            self.event = self.__class__.__name__

        if self.timestamp == 0:
            self.timestamp = int(time.time())


@dataclasses.dataclass
class StreamTimeoutEvent(ServerSideEvent):
    """Server side event that indicates the stream connection reached the
    maximum timeout (10 minutes)."""

    data: Optional[Any] = dataclasses.field(default=None)


#####################################
# API relevant types                #
#####################################


@dataclasses.dataclass
class APIError:
    """API error response.

    Attributes:
        code: HTTP error code.
        message: Error details.
    """

    code: int
    message: str


#####################################
# Jobs relevant types               #
#####################################


@dataclasses.dataclass
class BatchJobContext:
    """Simulation batch job context.

    Attributes:
        acyclic_graphs (List[linear_algebra.Graph]): List of acyclic_graphs to be run as a batch.
        params (List[linear_algebra.study.Sweepable]): List of parameters to be used
            with acyclic_graphs, same size as list of acyclic_graphs.
    """

    acyclic_graphs: List[Graph]
    params: List[Sweepable]

    def __post_init__(self) -> None:
        if len(self.acyclic_graphs) != len(self.params):
            raise ValueError(
                "Number of sweeps parameters has to match number of acyclic_graphs"
            )


@dataclasses.dataclass
class JobContext:
    """Simulation job context.

    Attributes:
        acyclic_graph (linear_algebra.Graph): Graph to be run.
        param_resolver (linear_algebra.ParamResolver): ParamResolver to be used with the
            acyclic_graph.
    """

    acyclic_graph: Graph
    param_resolver: ParamResolver


@dataclasses.dataclass
class SweepJobContext:
    """Simulation sweep job context.

    Attributes:
        acyclic_graph (linear_algebra.Graph): Graph to be run.
        params (linear_algebra.study.Sweepable): Parameters to be used with the
            acyclic_graph.
    """

    acyclic_graph: Graph
    params: Sweepable


class JobStatus(enum.IntEnum):
    """Current job status.

    Attributes:
        NOT_STARTED: The job was added to the queue.
        IN_PROGRESS: The job is being processed.
        COMPLETE: Simulation has been completed successfully.
        ERROR: Simulation has failed.
    """

    NOT_STARTED = 0
    IN_PROGRESS = 1
    COMPLETE = 2
    ERROR = 3


@dataclasses.dataclass
class JobProgress:
    """Job computation progress.

    Attributes:
        current: Number of completed work units.
        total: Total number of work units.
    """

    completed: int = dataclasses.field(default=0)
    total: int = dataclasses.field(default=1)

    def __post_init__(self) -> None:
        if self.completed < 0:
            raise ValueError("Current work unit cannot be less than zero")

        if self.total < 1:
            raise ValueError("Total number of work units cannot be less than 1")

        if self.completed > self.total:
            raise ValueError(
                "Current work unit cannot be greater than total work units"
            )


@dataclasses.dataclass
class JobResult:
    """Simulation job result.

    Attributes:
        id: Unique job id.
        status: Current job status.
        error_message: Optional error message explaining why the computation
            failed, only set if the `status` is
            :attr:`parallel_accel.client.schemas.JobStatus.ERROR`.
        progress: Optional computation progress, only set if the `status` is
            :attr:`parallel_accel.client.schemas.JobStatus.IN_PROGRESS`.
        result: Optional simulation job result, only set if the `status` is
            :attr:`parallel_accel.client.schemas.JobStatus.COMPLETE`.
    """

    id: uuid.UUID  # pylint: disable=invalid-name
    status: JobStatus = dataclasses.field(
        metadata={
            "marshmallow_field": marshmallow_enum.EnumField(
                JobStatus, by_value=True
            )
        }
    )
    error_message: Optional[str] = dataclasses.field(default=None)
    progress: Optional[JobProgress] = dataclasses.field(default=None)
    result: Optional[Any] = dataclasses.field(default=None)

    def __post_init__(self) -> None:
        if self.status == JobStatus.IN_PROGRESS and self.progress is None:
            raise ValueError("Missing job progress")

        if self.status == JobStatus.ERROR:
            if not self.error_message:
                raise ValueError("Missing error messsage")
            if self.result:
                raise ValueError("Failed job cannot have result field")

        if self.status == JobStatus.COMPLETE:
            if not self.result:
                raise ValueError("Missing job result")
            if self.error_message:
                raise ValueError(
                    "Completed job cannot have error_message field"
                )
            if (
                self.progress is not None
                and self.progress.total != self.progress.completed
            ):
                raise ValueError("Not all work units are marked as completed")


@dataclasses.dataclass
class JobStatusEvent(ServerSideEvent):
    """Job status changed event.

    Attributes:
        data: Simulation job result.
    """

    data: JobResult


@dataclasses.dataclass
class JobSubmitted:
    """Submitted job.

    Attributes:
        id: Unique job id.
    """

    id: uuid.UUID  # pylint: disable=invalid-name


#####################################
# Expectation job relevant types    #
#####################################


@dataclasses.dataclass
class ExpectationBatchJobContext(BatchJobContext):
    """Expectation values batch job context.

    Attributes:
        operators (List[List[linear_algebra.ops.ProbBasisAxisSum]]): List of list of
            `linear_algebra.ops.ProbBasisAxisSum` operators, same size as list of acyclic_graphs.
    """

    operators: List[Operators]

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.operators) != len(self.acyclic_graphs):
            raise ValueError(
                "Number of operators has to match number of acyclic_graphs"
            )


@dataclasses.dataclass
class ExpectationBatchJobResult(JobResult):
    """Expectation values batch job result.

    Attributes:
        result: List of expectation values list, same size as number of
            acyclic_graphs. Each element has the outer size of input sweep parameters
            and the inner size of input operators size.
    """

    result: Optional[List[List[List[float]]]] = dataclasses.field(default=None)


@dataclasses.dataclass
class ExpectationJobContext(JobContext):
    """Expectation values job context.

    Attributes:
        operators (linear_algebra.ops.ProbBasisAxisSum): List of `linear_algebra.ops.ProbBasisAxisSum` operators.
    """

    operators: Operators


@dataclasses.dataclass
class ExpectationJobResult(JobResult):
    """Expectation values job result.

    Attributes:
        result: List of floats, same size as input operators size.
    """

    result: Optional[List[float]] = dataclasses.field(default=None)


@dataclasses.dataclass
class ExpectationSweepJobContext(SweepJobContext):
    """Expectation values sweep job context.

    Attributes:
        operators (List[linear_algebra.ops.ProbBasisAxisSum]): List of `linear_algebra.ops.ProbBasisAxisSum`
            operators, same size as list of acyclic_graphs.
    """

    operators: Operators


@dataclasses.dataclass
class ExpectationSweepJobResult(JobResult):
    """Expectation values sweep job result.

    Attributes:
        result: List of expectation values list. The outer size is the same as
            input sweep size, the inner size is the same size as input operators
            size.
    """

    result: Optional[List[List[float]]] = dataclasses.field(default=None)


@dataclasses.dataclass
class ExpectationJobStatusEvent(JobStatusEvent):
    """Expectation job status changed event.

    Attributes:
        data: Expectation job result.
    """

    data: Union[
        ExpectationJobResult,
        ExpectationBatchJobResult,
        ExpectationSweepJobResult,
    ]


########################################
# Noisy expectation job relevant types #
########################################


@dataclasses.dataclass
class NoisyExpectationJobContext(ExpectationJobContext):
    """Noisy expectation job context.

    Attributes:
        num_samples: Number of times the operators will run. Can be specified as
            a single value or list of same size as input operators.
    """

    # We cannot set default field value for Union type
    num_samples: Union[int, List[int]]

    def __post_init__(self) -> None:
        if isinstance(self.num_samples, list) and (
            len(self.num_samples) != len(self.operators)
        ):
            raise ValueError(
                "Number of num_samples has to match number of operators"
            )


@dataclasses.dataclass
class NoisyExpectationJobResult(ExpectationJobResult):
    """Noisy expectation job result."""


@dataclasses.dataclass
class NoisyExpectationJobStatusEvent(JobStatusEvent):
    """Noisy expecation job status changed event.

    Attributes:
        data: Noisy expecation job result.
    """

    data: NoisyExpectationJobResult


#####################################
# Sample job relevant types         #
#####################################


@dataclasses.dataclass
class SampleBatchJobContext(BatchJobContext):
    """Sample batch job context.

    Attributes:
        repetitions: Number of times the acyclic_graphs will run. Can be specified as
            a single value or list of same size as input acyclic_graphs.
    """

    class RepetitionsValidator(
        marshmallow.validate.Validator
    ):  # pylint: disable=too-few-public-methods
        """A Helper class for validating repetitions field value."""

        def __call__(
            self, value: Union[int, List[int]]
        ) -> Union[int, List[int]]:
            if isinstance(value, list) and not all(x > 0 for x in value):
                raise marshmallow.ValidationError(
                    "All elements must be greater than or equal to 1"
                )

            if isinstance(value, int) and not value > 0:
                raise marshmallow.ValidationError(
                    "Must be greater than or equal to 1"
                )

            return value

    # We cannot set default field value for Union type
    repetitions: Union[int, List[int]] = dataclasses.field(
        metadata={"validate": RepetitionsValidator()}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.repetitions, list) and (
            len(self.repetitions) != len(self.acyclic_graphs)
        ):
            raise ValueError(
                "Number of repetitions has to match number of acyclic_graphs"
            )


@dataclasses.dataclass
class SampleBatchJobResult(JobResult):
    """Sample batch job result.

    Attributes:
        result (Optional[List[List[linear_algebra.Result]]]): Output from running the
            acyclic_graph.
    """

    result: Optional[List[List[Result]]] = dataclasses.field(default=None)


@dataclasses.dataclass
class SampleJobContext(JobContext):
    """Sample job context.

    Attributes:
        repetitions: Number of times the acyclic_graph will run.
    """

    repetitions: int = dataclasses.field(
        default=1, metadata={"validate": marshmallow.validate.Range(min=1)}
    )


@dataclasses.dataclass
class SampleJobResult(JobResult):
    """Sample job result.

    Attributes:
        result: Output from running the acyclic_graph.
    """

    result: Optional[Result] = dataclasses.field(default=None)


@dataclasses.dataclass
class SampleSweepJobContext(SweepJobContext):
    """Sample sweep job context.

    Attributes:
        repetitions: Number of times the acyclic_graph will run.
    """

    repetitions: int = dataclasses.field(
        default=1, metadata={"validate": marshmallow.validate.Range(min=1)}
    )


@dataclasses.dataclass
class SampleSweepJobResult(JobResult):
    """Sample sweep job result.

    Attributes:
        result: Output from running the acyclic_graph.
    """

    result: Optional[List[Result]] = dataclasses.field(default=None)


@dataclasses.dataclass
class SampleJobStatusEvent(JobStatusEvent):
    """Sample job status changed event.

    Attributes:
        data: Sample job result.
    """

    data: Union[SampleJobResult, SampleBatchJobResult, SampleSweepJobResult]


#####################################
# Jobs queue relevant types         #
#####################################


class JobType(enum.IntEnum):
    """Simulation job type.

    Attributes:
        SAMPLE: Sampling.
        EXPECTATION: Expectation values.
        NOISY_EXPECTATION: Noisy expectation values.
    """

    SAMPLE = 0
    EXPECTATION = 1
    NOISY_EXPECTATION = 2


@dataclasses.dataclass
class JobsQueue:
    """Current status of jobs queue.

    Attributes:
        ids: List of pending jobs ids.
    """

    ids: List[uuid.UUID] = dataclasses.field(default_factory=[])


@dataclasses.dataclass
class PendingJob:
    """Queued job details.

    Attributes:
        id: Unique job id.
        status: Current job status.
        type: Job type.
    """

    id: uuid.UUID  # pylint: disable=invalid-name
    status: JobStatus = dataclasses.field(
        metadata={
            "marshmallow_field": marshmallow_enum.EnumField(
                JobStatus,
                by_value=True,
            )
        }
    )
    type: JobType = dataclasses.field(
        metadata={
            "marshmallow_field": marshmallow_enum.EnumField(
                JobType, by_value=True
            )
        }
    )

    def __post_init__(self) -> None:
        if self.status in (JobStatus.COMPLETE, JobStatus.ERROR):
            raise ValueError(
                f"PendingJob cannot have {self.status.name} status"
            )


#####################################
# Tasks relevant types              #
#####################################


class TaskState(enum.IntEnum):
    """Current task state.

    Attributes:
        PENDING: Task is scheduled for execution.
        RUNNING: Task is running.
        DONE: Task is finished.
    """

    PENDING = 0
    RUNNING = 1
    DONE = 2


@dataclasses.dataclass
class TaskStatus:
    """Current task status.

    Attributes:
        state: Current task state.
        error: Optional error message explaining why the task failed, only set
            if the state is :attr:`parallel_accel.client.schemas.TaskState.DONE` and the
            `success` flag is False.
        success: Optional flag indicating whether task finished successfully,
            only set if the task state is
            :attr:`parallel_accel.client.schemas.TaskState.DONE`.
    """

    state: TaskState = dataclasses.field(
        metadata={
            "marshmallow_field": marshmallow_enum.EnumField(
                TaskState, by_value=True
            )
        }
    )
    error: Optional[str] = dataclasses.field(default=None)
    success: Optional[bool] = dataclasses.field(default=None)

    def __post_init__(self) -> None:
        """See base class documentation."""
        if self.state != TaskState.DONE and (
            (self.error is not None) or (self.success is not None)
        ):
            field = "error" if self.error is not None else "success"
            raise ValueError(f"Unfinished task cannot have {field} field.")


@dataclasses.dataclass
class TaskSubmitted:
    """Submitted task.

    Attributes:
        id: Unique task id.
    """

    id: uuid.UUID  # pylint: disable=invalid-name


@dataclasses.dataclass
class TaskStatusEvent(ServerSideEvent):
    """Task status changed event.

    Attributes:
        data: Task status.
    """

    data: TaskStatus


#####################################
# Worker relevant types             #
#####################################


class WorkerState(enum.IntEnum):
    """ASIC worker state.

    Attributes:
        BOOTING: Worker is booting.
        ERROR: Worker encountered an error.
        IDLE: Worker is idling.
        OFFLINE: Worker is offline.
        PROCESSING_JOB: Worker is processing a job.
        SHUTTING_DOWN: Worker is shutting down.
    """

    OFFLINE = 0
    BOOTING = 1
    SHUTTING_DOWN = 2
    IDLE = 3
    PROCESSING_JOB = 4
    ERROR = 5


@dataclasses.dataclass
class Worker:
    """Current status of the ASIC worker.

    Attributes:
        state: Current worker state.
        error: Optional error message explaining problem with the worker, only
            set when the `state` is
            :attr:`parallel_accel.client.schemas.WorkerState.ERROR`.
        job_id: Currently processed job id, only set when the `state` is
            :obj:`parallel_accel.client.schemas.WorkerState.PROCESSING_JOB`.
    """

    state: WorkerState = dataclasses.field(
        metadata={
            "marshmallow_field": marshmallow_enum.EnumField(
                WorkerState, by_value=True
            )
        }
    )
    error: Optional[str] = dataclasses.field(default=None)
    job_id: Optional[uuid.UUID] = dataclasses.field(default=None)

    def __post_init__(self) -> None:
        """See base class documentation."""
        if (
            self.state
            not in (
                WorkerState.PROCESSING_JOB,
                WorkerState.ERROR,
            )
            and ((self.error is not None) or (self.job_id is not None))
        ):
            raise ValueError(
                "Cannot have extra properties for the worker status "
                f"{self.state.name}"
            )

        if self.state == WorkerState.ERROR:
            if not self.error:
                raise ValueError("Missing error messsage")
            if self.job_id:
                raise ValueError("Cannot have job_id field for the ERROR state")

        if self.state == WorkerState.PROCESSING_JOB:
            if not self.job_id:
                raise ValueError("Missing job id")
            if self.error:
                raise ValueError("Cannot have error field for the IDLE state")


#####################################
# marshmallow schemas               #
#####################################


class _SSERenderer:
    """A helper class for serializing and deserializing objects to server side
    events message format.

    The server side event message is UTF-8 text data separated by a pair of
    newline characters.
    """

    @staticmethod
    def dumps(obj: Dict[str, Any], *_args, **_kwargs) -> str:
        r"""Encodes input object into text string.

        Args:
            obj: Object to be serialized.

        Returns:
            Text string in format:
                {key}: {value}\n
                ...
                \n
        """
        result = ""
        for key in ("event", "id", "timestamp", "data"):
            value = obj.get(key, None)
            if not value:
                continue

            if key == "data":
                value = json.dumps(value, separators=(",", ":"))

            result += f"{key}: {value}\n"

        result += "\n"
        return result

    @staticmethod
    def loads(  # pylint: disable=invalid-name
        s: str, *_args, **_kwargs
    ) -> Dict[str, Any]:
        """Decodes input text string into dict object.

        Args:
            s: Text string to be decoded.

        Returns:
            Dict object.
        """
        obj = {}
        for line in s.split("\n"):
            line = line.strip()
            if not line:
                continue

            key, value = line.split(": ")
            if key == "data":
                value = json.loads(value)

            obj[key] = value

        return obj


class _BaseSchema(marshmallow.Schema):
    """Base `marshmallow.schema.Schema` for ParallelAccel related schemas.

    This is a helper schema that provides custom `marsobj_fnllow.post_dump` method,
    that excludes all None fields from the final serialization result.
    """

    @marshmallow.post_dump
    def remove_empty_fields(  # pylint: disable=no-self-use
        self, data: Dict, **_kwargs
    ) -> Dict[str, Any]:
        """Removes all None fields from the input data.

        Args:
            data: Input data dictionary object.

        Returns:
            Filtered dictionary object.
        """
        return {k: v for k, v in data.items() if v is not None}


class _SSEBaseSchema(_BaseSchema):
    """Base `marshmallow.schema.Schema` for ParallelAccel service server side events."""

    class Meta:  # pylint: disable=too-few-public-methods
        """Metadata passed to the `marshmallow.schemas.Schema` constructor."""

        render_module = _SSERenderer


(
    APIErrorSchema,
    ExpectationBatchJobContextSchema,
    ExpectationBatchJobResultSchema,
    ExpectationJobContextSchema,
    ExpectationJobResultSchema,
    ExpectationJobStatusEventSchema,
    ExpectationSweepJobContextSchema,
    ExpectationSweepJobResultSchema,
    JobProgressSchema,
    JobResultSchema,
    JobStatusEventSchema,
    JobSubmittedSchema,
    JobsQueueSchema,
    NoisyExpectationJobContextSchema,
    NoisyExpectationJobResultSchema,
    NoisyExpectationJobStatusEventSchema,
    PendingJobSchema,
    SampleBatchJobContextSchema,
    SampleBatchJobResultSchema,
    SampleJobContextSchema,
    SampleJobResultSchema,
    SampleJobStatusEventSchema,
    SampleSweepJobContextSchema,
    SampleSweepJobResultSchema,
    ServerSideEventSchema,
    StreamTimeoutEventSchema,
    TaskStatusEventSchema,
    TaskStatusSchema,
    TaskSubmittedSchema,
    WorkerSchema,
) = tuple(
    marshmallow_dataclass.class_schema(x, base_schema=y)()
    for x, y in (
        (APIError, None),
        (ExpectationBatchJobContext, None),
        (ExpectationBatchJobResult, _BaseSchema),
        (ExpectationJobContext, None),
        (ExpectationJobResult, _BaseSchema),
        (ExpectationJobStatusEvent, _SSEBaseSchema),
        (ExpectationSweepJobContext, None),
        (ExpectationSweepJobResult, _BaseSchema),
        (JobProgress, None),
        (JobResult, _BaseSchema),
        (JobStatusEvent, _SSEBaseSchema),
        (JobSubmitted, None),
        (JobsQueue, None),
        (NoisyExpectationJobContext, None),
        (NoisyExpectationJobResult, _BaseSchema),
        (NoisyExpectationJobStatusEvent, _SSEBaseSchema),
        (PendingJob, None),
        (SampleBatchJobContext, None),
        (SampleBatchJobResult, _BaseSchema),
        (SampleJobContext, None),
        (SampleJobResult, _BaseSchema),
        (SampleJobStatusEvent, _SSEBaseSchema),
        (SampleSweepJobContext, None),
        (SampleSweepJobResult, _BaseSchema),
        (ServerSideEvent, _SSEBaseSchema),
        (StreamTimeoutEvent, _SSEBaseSchema),
        (TaskStatusEvent, _SSEBaseSchema),
        (TaskStatus, _BaseSchema),
        (TaskSubmitted, None),
        (Worker, _BaseSchema),
    )
)
