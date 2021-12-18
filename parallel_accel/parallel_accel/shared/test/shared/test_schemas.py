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
"""Unit test for schema module"""
import dataclasses
import unittest
import uuid
import linear_algebra
import marshmallow

from parallel_accel.shared import schemas


class TestSchema(unittest.TestCase):  # pylint: disable=too-many-public-methods
    """Tests shared schema"""

    def test_api_error(self) -> None:
        """Tests APIError schema"""
        data = schemas.APIError(code=500, message="error message")
        schema = schemas.APIErrorSchema

        # Run test
        self._verify(schema, data)

    def test_expectation_batch_job_context(self) -> None:
        """Tests ExpectationBatchJobContext schema"""
        discretes = linear_algebra.LinearSpace.range(2)
        data = schemas.ExpectationBatchJobContext(
            acyclic_graphs=[linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1)))],
            params=[linear_algebra.ParamResolver(None)],
            operators=[
                [
                    linear_algebra.flip_x_axis(discretes[0]) + linear_algebra.flip_y_axis(discretes[1]),
                    linear_algebra.flip_y_axis(discretes[0]) + linear_algebra.flip_x_axis(discretes[1]),
                ]
            ],
        )
        schema = schemas.ExpectationBatchJobContextSchema

        # Run test
        self._verify(schema, data)

        # Test case: invalid number of params
        with self.assertRaises(ValueError):
            schemas.ExpectationBatchJobContext(
                acyclic_graphs=[linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1)))],
                params=[],
                operators=[
                    [
                        linear_algebra.flip_x_axis(discretes[0]) + linear_algebra.flip_y_axis(discretes[1]),
                        linear_algebra.flip_y_axis(discretes[0]) + linear_algebra.flip_x_axis(discretes[1]),
                    ]
                ],
            )

        # Test case: invalid number of operators
        with self.assertRaises(ValueError):
            schemas.ExpectationBatchJobContext(
                acyclic_graphs=[linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1)))],
                params=[linear_algebra.ParamResolver(None)],
                operators=[],
            )

    def test_expectations_batch_job_result(self) -> None:
        """Tests ExpectationBatchJobResult schema"""
        data = schemas.ExpectationBatchJobResult(
            id=uuid.uuid4(),
            status=schemas.JobStatus.COMPLETE,
            result=[[[1.0, 2.0]]],
        )
        schema = schemas.ExpectationBatchJobResultSchema

        # Run test
        self._verify(schema, data)

    def test_expectation_job(self) -> None:
        """Tests ExpectationJob schema"""
        api_key = "api-key"
        acyclic_graph = linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1)))
        param_resolver = linear_algebra.ParamResolver(None)

        discretes = linear_algebra.LinearSpace.range(2)
        operators = [
            linear_algebra.flip_x_axis(discretes[0]) + linear_algebra.flip_y_axis(discretes[1]),
            linear_algebra.flip_y_axis(discretes[0]) + linear_algebra.flip_x_axis(discretes[1]),
        ]

        schema = schemas.ExpectationJobSchema

        # Test case: ExpectationBatchJobContext
        context = schemas.ExpectationBatchJobContext(
            acyclic_graphs=[acyclic_graph], params=[param_resolver], operators=[operators]
        )
        data = schemas.ExpectationJob(
            api_key=api_key,
            context=context,
            id=uuid.uuid4(),
            type=schemas.JobType.EXPECTATION,
        )

        self._verify(schema, data)

        # Test case: ExpectationJobContext
        context = schemas.ExpectationJobContext(
            acyclic_graph=acyclic_graph, param_resolver=param_resolver, operators=operators
        )
        data = schemas.ExpectationJob(
            api_key=api_key,
            context=context,
            id=uuid.uuid4(),
            type=schemas.JobType.EXPECTATION,
        )

        self._verify(schema, data)

        # Test case: ExpectationSweepJobContext
        context = schemas.ExpectationSweepJobContext(
            acyclic_graph=acyclic_graph, params=param_resolver, operators=operators
        )
        data = schemas.ExpectationJob(
            api_key=api_key,
            context=context,
            id=uuid.uuid4(),
            type=schemas.JobType.EXPECTATION,
        )

        self._verify(schema, data)

        # Test is_same_api_key method behavior
        self.assertTrue(data.is_same_api_key(api_key))
        self.assertFalse(data.is_same_api_key(api_key + api_key))

    def test_expectation_job_context(self) -> None:
        """Tests ExpectationJobContext schema"""
        discretes = linear_algebra.LinearSpace.range(2)
        data = schemas.ExpectationJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
            operators=[
                linear_algebra.flip_x_axis(discretes[0]) + linear_algebra.flip_y_axis(discretes[1]),
                linear_algebra.flip_y_axis(discretes[0]) + linear_algebra.flip_x_axis(discretes[1]),
            ],
        )
        schema = schemas.ExpectationJobContextSchema

        # Run test
        self._verify(schema, data)

    def test_expectations_job_result(self) -> None:
        """Tests ExpectationJobResult schema"""
        data = schemas.ExpectationJobResult(
            id=uuid.uuid4(),
            status=schemas.JobStatus.COMPLETE,
            result=[1.0, 2.0],
        )
        schema = schemas.ExpectationJobResultSchema

        # Run test
        self._verify(schema, data)

    def test_expectation_sweep_job_context(self) -> None:
        """Tests ExpectationSweepJobContext schema"""
        discretes = linear_algebra.LinearSpace.range(2)
        data = schemas.ExpectationSweepJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            params=linear_algebra.ParamResolver(None),
            operators=[
                linear_algebra.flip_x_axis(discretes[0]) + linear_algebra.flip_y_axis(discretes[1]),
                linear_algebra.flip_y_axis(discretes[0]) + linear_algebra.flip_x_axis(discretes[1]),
            ],
        )
        schema = schemas.ExpectationSweepJobContextSchema

        # Run test
        self._verify(schema, data)

    def test_expectations_sweep_job_result(self) -> None:
        """Tests ExpectationSweepJobResult schema"""
        data = schemas.ExpectationSweepJobResult(
            id=uuid.uuid4(),
            status=schemas.JobStatus.COMPLETE,
            result=[[1.0, 2.0]],
        )
        schema = schemas.ExpectationSweepJobResultSchema

        # Run test
        self._verify(schema, data)

    def test_job_progress(self) -> None:
        """Tests JobProgress schema"""
        schema = schemas.JobProgressSchema

        # Test case: default values
        data = schemas.JobProgress()

        self._verify(schema, data)

        # Test case: invalid current work unit
        with self.assertRaises(ValueError):
            data = schemas.JobProgress(completed=-1)

        # Test case: invalid total work units
        with self.assertRaises(ValueError):
            data = schemas.JobProgress(total=0)

        # Test case: current work unit greated than total units
        with self.assertRaises(ValueError):
            data = schemas.JobProgress(completed=1, total=0)

    def test_job_result(self) -> None:
        """Tests JobResults schema"""
        job_id = uuid.uuid4()
        schema = schemas.JobResultSchema

        # Test case: serialize data - job completed
        data = schemas.JobResult(
            id=job_id,
            result="example",
            status=schemas.JobStatus.COMPLETE,
        )

        self._verify(schema, data)

        dump = schemas.encode(schema, data)
        self.assertTrue("error_message" not in dump)

        # Test case: serialize data - job not started
        data = schemas.JobResult(
            id=job_id,
            status=schemas.JobStatus.NOT_STARTED,
        )

        self._verify(schema, data)

        dump = schemas.encode(schema, data)
        self.assertTrue("error_message" not in dump)
        self.assertTrue("result" not in dump)

        # Test case: serialize data - job failed
        data = schemas.JobResult(
            error_message="Some error",
            id=job_id,
            status=schemas.JobStatus.ERROR,
        )

        self._verify(schema, data)

        dump = schemas.encode(schema, data)
        self.assertTrue("result" not in dump)

        # Test case: missing error message
        with self.assertRaises(ValueError):
            schemas.JobResult(id=job_id, status=schemas.JobStatus.ERROR)

        # Test case: result with failed job
        with self.assertRaises(ValueError):
            schemas.JobResult(
                error_message="Error",
                id=job_id,
                result="result",
                status=schemas.JobStatus.ERROR,
            )

        # Test case: missing job result
        with self.assertRaises(ValueError):
            schemas.JobResult(id=job_id, status=schemas.JobStatus.COMPLETE)

        # Test case: error_message with completed job
        with self.assertRaises(ValueError):
            schemas.JobResult(
                error_message="Error",
                id=job_id,
                result="result",
                status=schemas.JobStatus.ERROR,
            )

    def test_job_status_event(self) -> None:
        """Tests JobStatusEvent schema"""
        job_id = uuid.uuid4()
        payload = schemas.JobResult(
            id=job_id, status=schemas.JobStatus.NOT_STARTED
        )
        data = schemas.JobStatusEvent(id=uuid.uuid4(), data=payload)
        schema = schemas.JobStatusEventSchema

        # Run test
        self._verify(schema, data)


    def test_noisy_expectation_job(self) -> None:
        """Tests NoisyExpectationJob schema"""
        api_key = "api-key"
        acyclic_graph = linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1)))
        param_resolver = linear_algebra.ParamResolver(None)

        discretes = linear_algebra.LinearSpace.range(2)
        operators = [
            linear_algebra.flip_x_axis(discretes[0]) + linear_algebra.flip_y_axis(discretes[1]),
            linear_algebra.flip_y_axis(discretes[0]) + linear_algebra.flip_x_axis(discretes[1]),
        ]
        num_samples = 100

        schema = schemas.NoisyExpectationJobSchema

        # Test case: NoisyExpectationJobContext with int num_samples
        context = schemas.NoisyExpectationJobContext(
            acyclic_graph=acyclic_graph, param_resolver=param_resolver,
            operators=operators, num_samples=num_samples
        )
        data = schemas.NoisyExpectationJob(
            api_key=api_key,
            context=context,
            id=uuid.uuid4(),
            type=schemas.JobType.NOISY_EXPECTATION,
        )

        self._verify(schema, data)

        # Test case: NoisyExpectationJobContext with List[int] num_samples.
        # This num_samples account for each operator.
        context = schemas.NoisyExpectationJobContext(
            acyclic_graph=acyclic_graph, param_resolver=param_resolver,
            operators=operators, num_samples=[num_samples for _ in operators]
        )
        data = schemas.NoisyExpectationJob(
            api_key=api_key,
            context=context,
            id=uuid.uuid4(),
            type=schemas.JobType.NOISY_EXPECTATION,
        )

        self._verify(schema, data)

        # Test is_same_api_key method behavior
        self.assertTrue(data.is_same_api_key(api_key))
        self.assertFalse(data.is_same_api_key(api_key + api_key))

    def test_noisy_expectation_job_context(self) -> None:
        """Tests NoisyExpectationJobContext schema"""
        discretes = linear_algebra.LinearSpace.range(2)
        data = schemas.NoisyExpectationJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
            operators=[
                linear_algebra.flip_x_axis(discretes[0]) + linear_algebra.flip_y_axis(discretes[1]),
                linear_algebra.flip_y_axis(discretes[0]) + linear_algebra.flip_x_axis(discretes[1]),
            ],
            num_samples=100,
        )
        schema = schemas.NoisyExpectationJobContextSchema

        # Run test
        self._verify(schema, data)

        data = schemas.NoisyExpectationJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
            operators=[
                linear_algebra.flip_x_axis(discretes[0]) + linear_algebra.flip_y_axis(discretes[1]),
                linear_algebra.flip_y_axis(discretes[0]) + linear_algebra.flip_x_axis(discretes[1]),
            ],
            num_samples=[100, 100],  # we have 2 operators.
        )

        # Run test
        self._verify(schema, data)

    def test_noisy_expectations_job_result(self) -> None:
        """Tests NoisyExpectationJobResult schema"""
        data = schemas.NoisyExpectationJobResult(
            id=uuid.uuid4(),
            status=schemas.JobStatus.COMPLETE,
            result=[1.0, 2.0],
        )
        schema = schemas.NoisyExpectationJobResultSchema

        # Run test
        self._verify(schema, data)


    def test_pending_job(self) -> None:
        """Tests PendingJob schema"""
        data = schemas.PendingJob(
            id=uuid.uuid4(),
            status=schemas.JobStatus.IN_PROGRESS,
            type=schemas.JobType.EXPECTATION,
        )
        schema = schemas.PendingJobSchema

        # Run test
        self._verify(schema, data)

        # Test case: invalid job status
        for status in (schemas.JobStatus.COMPLETE, schemas.JobStatus.ERROR):
            with self.assertRaises(ValueError):
                schemas.PendingJob(
                    id=uuid.uuid4(), status=status, type=schemas.JobType.SAMPLE
                )

    def test_sample_batch_job_context(self) -> None:
        """Tests SampleBatchJobContext schema"""
        data = schemas.SampleBatchJobContext(
            acyclic_graphs=[linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1)))],
            params=[linear_algebra.ParamResolver(None)],
            repetitions=1,
        )
        schema = schemas.SampleBatchJobContextSchema

        # Run test
        self._verify(schema, data)
        self.assertEqual(data.repetitions, 1)

        # Test case: invalid number of params
        with self.assertRaises(ValueError):
            schemas.SampleBatchJobContext(
                acyclic_graphs=[linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1)))],
                params=[],
                repetitions=1,
            )

        # Test case: invalid number of repetitions
        with self.assertRaises(ValueError):
            data = schemas.SampleBatchJobContext(
                acyclic_graphs=[linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1)))],
                params=[linear_algebra.ParamResolver(None)],
                repetitions=[],
            )

    def test_samples_batch_job_result(self) -> None:
        """Tests SampleBatchJobResult schema"""
        data = schemas.SampleBatchJobResult(
            id=uuid.uuid4(),
            status=schemas.JobStatus.COMPLETE,
            result=[
                [
                    linear_algebra.Result(
                        params=linear_algebra.ParamResolver(None),
                        observations={},
                    )
                ]
            ],
        )
        schema = schemas.SampleBatchJobResultSchema

        # Run test
        self._verify(schema, data)

    def test_sample_job(self) -> None:
        """Tests SampleJob schema"""
        api_key = "api_key"
        acyclic_graph = linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1)))
        param_resolver = linear_algebra.ParamResolver(None)

        schema = schemas.SampleJobSchema

        # Test case: SampleBatchJobContext
        context = schemas.SampleBatchJobContext(
            acyclic_graphs=[acyclic_graph],
            params=[param_resolver],
            repetitions=1,
        )
        data = schemas.SampleJob(
            api_key=api_key,
            context=context,
            id=uuid.uuid4(),
            type=schemas.JobType.SAMPLE,
        )

        self._verify(schema, data)

        # Test case: SampleJobContext
        context = schemas.SampleJobContext(
            acyclic_graph=acyclic_graph, param_resolver=param_resolver
        )
        data = schemas.SampleJob(
            api_key=api_key,
            context=context,
            id=uuid.uuid4(),
            type=schemas.JobType.SAMPLE,
        )

        self._verify(schema, data)

        # Test case: SampleSweepJobContext
        context = schemas.SampleSweepJobContext(
            acyclic_graph=acyclic_graph, params=param_resolver
        )
        data = schemas.SampleJob(
            api_key=api_key,
            context=context,
            id=uuid.uuid4(),
            type=schemas.JobType.SAMPLE,
        )

        self._verify(schema, data)

    def test_sample_job_context(self) -> None:
        """Tests SampleJobContext schema"""
        data = schemas.SampleJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            param_resolver=linear_algebra.ParamResolver(None),
        )
        schema = schemas.SampleJobContextSchema

        # Run test
        self._verify(schema, data)

    def test_sample_job_result(self) -> None:
        """Tests SampleJobResult schema"""
        data = schemas.SampleJobResult(
            id=uuid.uuid4(),
            status=schemas.JobStatus.COMPLETE,
            result=linear_algebra.Result(
                params=linear_algebra.ParamResolver(None),
                observations={},
            ),
        )
        schema = schemas.SampleJobResultSchema

        # Run test
        self._verify(schema, data)

    def test_sample_sweep_job_context(self) -> None:
        """Tests SampleSweepJobContext schema"""
        data = schemas.SampleSweepJobContext(
            acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
            params=linear_algebra.ParamResolver(None),
        )
        schema = schemas.SampleSweepJobContextSchema

        # Run test
        self._verify(schema, data)

    def test_sample_sweep_job_result(self) -> None:
        """Tests SampleSweepJobResult schema"""
        data = schemas.SampleSweepJobResult(
            id=uuid.uuid4(),
            status=schemas.JobStatus.COMPLETE,
            result=[
                linear_algebra.Result(
                    params=linear_algebra.ParamResolver(None),
                    observations={},
                )
            ],
        )
        schema = schemas.SampleSweepJobResultSchema

        # Run test
        self._verify(schema, data)

    def test_stream_timeout_event(self) -> None:
        """Tests StreamTimeoutEvent schema"""
        data = schemas.StreamTimeoutEvent(id=uuid.uuid4())
        schema = schemas.StreamTimeoutEventSchema

        # Run test
        self._verify(schema, data)

    def test_task_status(self) -> None:
        """Tests TaskStatus schema"""
        data = schemas.TaskStatus(
            state=schemas.TaskState.DONE,
            error="error message",
            success=False,
        )
        schema = schemas.TaskStatusSchema

        # Run test
        self._verify(schema, data)

        # Test case: extra field with PENDING state
        for kwargs in ({"error": "error message"}, {"success": True}):
            with self.assertRaises(ValueError):
                kwargs["state"] = schemas.TaskState.PENDING
                data = schemas.TaskStatus(**kwargs)

    def test_task_status_event(self) -> None:
        """Tests TaskStatusEvent schema"""
        payload = schemas.TaskStatus(state=schemas.TaskState.PENDING)
        data = schemas.TaskStatusEvent(id=uuid.uuid4(), data=payload)
        schema = schemas.TaskStatusEventSchema

        # Run test
        self._verify(schema, data)

    def test_task_submitted(self) -> None:
        """Tests TaskSubmitted schema"""
        data = schemas.TaskSubmitted(id=uuid.uuid4())
        schema = schemas.TaskSubmittedSchema

        # Run test
        self._verify(schema, data)

    def test_worker(self) -> None:
        """Tests Worker schema"""
        job_id = uuid.uuid4()
        schema = schemas.WorkerSchema

        # Test case: serialize data - PROCESSING_JOB state
        data = schemas.Worker(
            job_id=job_id,
            state=schemas.WorkerState.PROCESSING_JOB,
        )

        self._verify(schema, data)

        dump = schemas.encode(schema, data)
        self.assertTrue("error" not in dump)

        # Test case: serialize data - ERROR state
        data = schemas.Worker(
            error="Some error",
            state=schemas.WorkerState.ERROR,
        )

        self._verify(schema, data)

        dump = schemas.encode(schema, data)
        self.assertTrue("job_id" not in dump)

        # Test case: serialize data - IDLE state
        data = schemas.Worker(state=schemas.WorkerState.IDLE)

        self._verify(schema, data)

        dump = schemas.encode(schema, data)
        for key in ("error", "result"):
            self.assertTrue(key not in dump)

        # Test case: missing error message for ERROR state
        with self.assertRaises(ValueError):
            schemas.Worker(state=schemas.WorkerState.ERROR)

        # Test case: job_id with ERROR state
        with self.assertRaises(ValueError):
            schemas.Worker(
                error="error message",
                job_id=job_id,
                state=schemas.WorkerState.ERROR,
            )

        # Test case: missing job id for PROCESSING_JOB state
        with self.assertRaises(ValueError):
            schemas.Worker(state=schemas.WorkerState.PROCESSING_JOB)

        # Test case: error_message with PROCESSING_JOB state
        with self.assertRaises(ValueError):
            schemas.Worker(
                error="error message",
                job_id=job_id,
                state=schemas.WorkerState.ERROR,
            )

        # Test case: extra parameters for IDLE state
        with self.assertRaises(ValueError):
            schemas.Worker(
                error="error message",
                job_id=job_id,
                state=schemas.WorkerState.IDLE,
            )

    def test_worker_internal(self) -> None:
        """Tests WorkerInternal schema"""
        schema = schemas.WorkerInternalSchema

        # Test case: serialize data - default state
        data = schemas.WorkerInternal(state=schemas.WorkerState.IDLE)

        self.assertEqual(data.job_timestamp, 0)
        self._verify(schema, data)

    def _verify(
        self, schema: marshmallow.Schema, data: dataclasses.dataclass
    ) -> None:
        """Verifies whether serialized data can be decoded back to the same object.

        Args:
            schema: Schema to be used for encoding and decoding
            data: Data to be encoded
        """
        dump = schemas.encode(schema, data)
        self.assertEqual(schemas.decode(schema, dump), data)
