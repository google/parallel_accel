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
swagger: "2.0"
info:
  title: "${title_prefix}ParallelAccel Remote Simulation API Specification"
  description: |
    This document defines ParallelAccel-Client public REST API.

    For questions contact the ParallelAccel Team.
  contact:
    name: ParallelAccel Team
  version: 0.2.5

host: ${hostname}
REDACTED:
  deadline: 620.0
REDACTED:
- name: ${hostname}
  target: ${ip_address}

basePath: /api/v1
consumes:
  - application/json
produces:
  - application/json
schemes:
  - https

securityDefinitions:
  ApiKeyHeader:
    type: apiKey
    in: header
    name: X-API-Key
security:
  - ApiKeyHeader: []

paths:
  # START [Jobs endpoints]
  # START [Expectation endpoints]
  /jobs/exp/submit:
    post:
      description: Submits expectation job.
      operationId: submitExpectationJob
      tags:
        - Expectation
        - Jobs
      consumes:
        - application/json
      produces:
        - application/json
      parameters:
        - in: body
          name: context
          schema:
            $ref: "#/definitions/ExpectationJobContext"
      responses:
        201:
          $ref: "#/responses/JobSubmitted"
        default:
          $ref: "#/responses/APIError"
  /jobs/exp/batch/submit:
    post:
      description: Submits expectation batch job.
      operationId: submitExpectationBatchJob
      tags:
        - Expectation
        - Jobs
      consumes:
        - application/json
      produces:
        - application/json
      parameters:
        - in: body
          name: context
          schema:
            $ref: "#/definitions/ExpectationBatchJobContext"
      responses:
        201:
          $ref: "#/responses/JobSubmitted"
        default:
          $ref: "#/responses/APIError"
  /jobs/exp/sweep/submit:
    post:
      description: Submits expectation sweep job.
      operationId: submitExpectationSweepJob
      tags:
        - Expectation
        - Jobs
      consumes:
        - application/json
      produces:
        - application/json
      parameters:
        - in: body
          name: context
          schema:
            $ref: "#/definitions/ExpectationSweepJobContext"
      responses:
        201:
          $ref: "#/responses/JobSubmitted"
        default:
          $ref: "#/responses/APIError"
  /jobs/exp/{id}/stream:
    get:
      description: Gets expectation job status event stream.
      operationId: getExpJobResults
      tags:
        - Expectation
        - Jobs
      parameters:
        - $ref: "#/parameters/UUID"
      produces:
        - text/event-stream
      responses:
        200:
          description: Expectation job status event stream.
          schema:
            $ref: "#/definitions/ExpectationJobStatusEvent"
        default:
          $ref: "#/responses/APIError"
  # END [Expectation endpoints]
  # START [Sample endpoints]
  /jobs/sample/batch/submit:
    post:
      description: Submits sample batch job.
      operationId: submitSampleBatchJob
      tags:
        - Samples
        - Jobs
      consumes:
        - application/json
      produces:
        - application/json
      parameters:
        - in: body
          name: context
          schema:
            $ref: "#/definitions/SampleBatchJobContext"
      responses:
        201:
          $ref: "#/responses/JobSubmitted"
        default:
          $ref: "#/responses/APIError"
  /jobs/sample/submit:
    post:
      description: Submits sample job.
      operationId: submitSampleJob
      tags:
        - Samples
        - Jobs
      consumes:
        - application/json
      produces:
        - application/json
      parameters:
        - in: body
          name: context
          schema:
            $ref: "#/definitions/SampleJobContext"
      responses:
        201:
          $ref: "#/responses/JobSubmitted"
        default:
          $ref: "#/responses/APIError"
  /jobs/sample/sweep/submit:
    post:
      description: Submits sample sweep job.
      operationId: submitSampleSweepJob
      tags:
        - Samples
        - Jobs
      consumes:
        - application/json
      produces:
        - application/json
      parameters:
        - in: body
          name: context
          schema:
            $ref: "#/definitions/SampleSweepJobContext"
      responses:
        201:
          $ref: "#/responses/JobSubmitted"
        default:
          $ref: "#/responses/APIError"
  /jobs/sample/{id}/stream:
    get:
      tags:
        - Samples
        - Jobs
      description: Gets sample job status event stream.
      operationId: getSampleJobResults
      parameters:
        - $ref: "#/parameters/UUID"
      produces:
        - text/event-stream
      responses:
        200:
          description: Sample job status event stream.
          schema:
            $ref: "#/definitions/SampleJobStatusEvent"
        default:
          $ref: "#/responses/APIError"
  # END [Sample endpoints]
  # START [Jobs queue]
  /jobs/queue:
    get:
      tags:
        - Jobs Queue
      description: Gets jobs queue
      operationId: getJobsQueue
      produces:
        - application/json
      responses:
        200:
          description: Array of pending jobs ids
          schema:
            type: array
            items:
              $ref: "#/definitions/UUID"
        default:
          $ref: "#/responses/APIError"
    delete:
      tags:
        - Jobs Queue
      description: Flushes jobs queue
      operationId: flushJobsQueue
      produces:
        - application/json
      responses:
        200:
          description: Flush job queue requested.
          schema:
            $ref: "#/definitions/TaskSubmitted"
        default:
          $ref: "#/responses/APIError"
  /jobs/queue/{id}:
    get:
      description: Gets pending job details
      operationId: getQueuedJob
      tags:
        - Jobs Queue
      parameters:
        - $ref: "#/parameters/UUID"
      produces:
        - application/json
      responses:
        200:
          description: Pending job object
          schema:
            $ref: "#/definitions/PendingJob"
        default:
          $ref: "#/responses/APIError"
  # END [Jobs queue]
  # END [Jobs endpoints]

  # START [Tasks endpoints]
  /tasks/{id}/stream:
    get:
      description: Task status event stream
      operationId: taskEventStream
      tags:
        - Tasks
      parameters:
        - $ref: "#/parameters/UUID"
      produces:
        - text/event-stream
      responses:
        200:
          description: Task status event stream
          schema:
            $ref: "#/definitions/TaskStatusEvent"
        default:
          $ref: "#/responses/APIError"
  # END [Tasks endpoints]

  # START [Worker endpoints]
  /worker/{command}:
    post:
      tags:
        - Worker
      description: Triggers worker command.
      operationId: triggerWorkerCommand
      parameters:
        - in: path
          name: command
          description: |
            Command to be exectued:
            * **restart** - restarts the worker
            * **start** - starts the worker
            * **stop** - stops the worker
          required: true
          type: string
          pattern: ^(restart)|(start)|(stop)$
      produces:
        - application/json
      responses:
        201:
          description: Worker command submitted.
          schema:
            $ref: "#/definitions/TaskSubmitted"
        default:
          $ref: "#/responses/APIError"
  /worker/status:
    get:
      tags:
        - Worker
      description: Gets worker status.
      operationId: getWorkerStatus
      produces:
        - application/json
      responses:
        200:
          description: Worker status.
          schema:
            $ref: "#/definitions/WorkerStatus"
        default:
          $ref: "#/responses/APIError"
  # END [Worker endpoints]

parameters:
  UUID:
    in: path
    name: id
    description: Object unique id
    required: true
    type: string
    pattern: ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[4][0-9a-fA-F]{3}-[89AaBb][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$

responses:
  # Generic error response
  APIError:
    description: API error
    schema:
      $ref: "#/definitions/APIError"
    examples:
      application/json:
        code: 403
        message: "Invalid or missing API token"

  # Jobs specific responses
  JobSubmitted:
    description: Job submitted
    schema:
      $ref: "#/definitions/JobSubmitted"

  # Task requested
  TaskSubmitted:
    description: Asynchronous task requested
    schema:
      $ref: "#/definitions/TaskSubmitted"

definitions:
  # START [Common API types]
  ErrorMessage:
    description: Detailed error message.
    type: string
    example: Unknown error occured.
  UUID:
    type: string
    pattern: ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[4][0-9a-fA-F]{3}-[89AaBb][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$
    example: c6d89bba-0a4f-472b-aa69-6c7e9efbb4a6

  APIError:
    description: API error message.
    type: object
    properties:
      code:
        type: number
        description: HTTP error code
      message:
        $ref: "#/definitions/ErrorMessage"
  ServerEvent:
    description: Server Side Event.
    type: array
    format: chunked
    items:
      type: object
      format: text
      properties:
        id:
          $ref: "#/definitions/UUID"
        event:
          type: string
          description: Event type.
        timestamp:
          type: number
          description: Event timestamp in UNIX seconds.
  # END [Common API types]

  # START [Jobs relevant types]
  # START [LinearAlgebra relevant types]
  Graph:
    description: |
      JSON-encoded [linear_algebra.acyclic_graphs.Graph](
      reference_doc/python/linear_algebra/acyclic_graphs/Graph) to be
      run.
    type: string
  Operators:
    description: |
      JSON-encoded list of [linear_algebra.ops.ProbBasisAxisSum](
      reference_doc/python/linear_algebra/ops/ProbBasisAxisSum)
      operators.
    type: string
  ParamResolver:
    description: |
      JSON-encoded [linear_algebra.study.ParamResolver](
        reference_doc/python/linear_algebra/study/ParamResolver) to
        be used with the acyclic_graph.
    type: string
  Repetitions:
    description: The number of times to sample.
    type: number
  Result:
    description: |
      A JSON-encoded [linear_algebra.study.Result](
      reference_doc/python/linear_algebra/study/Result)
      object containing the output from running the acyclic_graph.
    type: string
  Sweepable:
    description: |
      JSON-encoded [linear_algebra.study.Sweepable](
        reference_doc/python/linear_algebra/study/Sweepable) to
        be used with the acyclic_graph.
    type: string
  # END [LinearAlgebra relevant types]
  JobId:
    description: Unique job id.
    allOf:
      - $ref: "#/definitions/UUID"
  JobStatus:
    description: |
      Current job status:
      * JOB_STATUS_NOT_STARTED = 0
      * JOB_STATUS_IN_PROGRESS = 1
      * JOB_STATUS_COMPLETE = 2
      * JOB_STATUS_ERROR = 3
    type: integer
    enum: [0, 1, 2, 3]
    example: 2
  JobType:
    description: |
      Job type:
      * JOB_TYPE_SAMPLE = 0
      * JOB_TYPE_EXPECTATION = 1
    type: integer
    enum: [0, 1]
    example: 1

  BatchJobContext:
    description: Submit new batch job context
    type: object
    properties:
      acyclic_graphs:
        description: |
          JSON-encoded list of [linear_algebra.acyclic_graphs.Graph](
          reference_doc/python/linear_algebra/acyclic_graphs/Graph) to be
          run.
        type: array
        items:
          $ref: "#/definitions/Graph"
      params:
        description: |
          JSON-encoded [linear_algebra.study.Sweepable](
          reference_doc/python/linear_algebra/study/Sweepable) to
          be used with the acyclic_graph. Same size as the list of acyclic_graphs.
        type: array
        items:
          $ref: "#/definitions/Sweepable"
    required:
      - acyclic_graphs
      - params
  JobContext:
    description: Submit new job context
    type: object
    properties:
      acyclic_graph:
        $ref: "#/definitions/Graph"
      param_resolver:
        $ref: "#/definitions/ParamResolver"
    required:
      - acyclic_graph
      - param_resolver
  SweepJobContext:
    description: Submit new sweep job context
    type: object
    properties:
      acyclic_graph:
        $ref: "#/definitions/Graph"
      params:
        $ref: "#/definitions/Sweepable"
    required:
      - acyclic_graph
      - params
  JobProgress:
    description: Simulation job computation progress
    type: object
    properties:
      completed:
        description: Number of completed work units
        type: integer
        example: 1
      total:
        description: Total number of work units
        type: integer
        example: 2
  JobResult:
    description: Simulation job result
    type: object
    properties:
      error_message:
        $ref: "#/definitions/ErrorMessage"
      id:
        $ref: "#/definitions/JobId"
      progress:
        $ref: "#/definitions/JobProgress"
      status:
        $ref: "#/definitions/JobStatus"
  JobSubmitted:
    description: Submit job response type.
    type: object
    properties:
      id:
        $ref: "#/definitions/JobId"
  PendingJob:
    description: Queued job object
    type: object
    properties:
      id:
        $ref: "#/definitions/JobId"
      status:
        $ref: "#/definitions/JobStatus"
      type:
        $ref: "#/definitions/JobType"
  # END [Jobs relevant types]

  # START [Expectation endpoints types]
  ExpectationJobContext:
    description: New expectation job context.
    allOf:
      - $ref: "#/definitions/JobContext"
    properties:
      operators:
        $ref: "#/definitions/Operators"
    required:
      - operators
  ExpectationBatchJobContext:
    description: New expectation batch job context.
    allOf:
      - $ref: "#/definitions/BatchJobContext"
    properties:
      operators:
        type: array
        items:
          $ref: "#/definitions/Operators"
    required:
      - operators
  ExpectationSweepJobContext:
    description: New expectation sweep job context.
    allOf:
      - $ref: "#/definitions/SweepJobContext"
    properties:
      operators:
        $ref: "#/definitions/Operators"
    required:
      - operators
  ExpectationJobResult:
    description:  Result of the expectation job type.
    allOf:
      - $ref: "#/definitions/JobResult"
    properties:
      result:
        description:  |
          A list of expectation values, with the value at index n corresponding
          to observables[n] from the input.
        type: array
        items:
          type: number
        example: [1.23456]
  ExpectationBatchJobResult:
    description:  Result of the expectation batch job type.
    allOf:
      - $ref: "#/definitions/JobResult"
    properties:
      result:
        description:  |
            A list of expectation-value lists. The outer index determines the
            acyclic_graph, the midle index determines the sweep, and the inner index
            determines the observable. For instance, results[1][2][3] would
            select the fourth observable measured in the third sweep in the
            second acyclic_graph.
        type: array
        items:
          type: array
          items:
            type: array
            items:
              type: number
        example: [[[1.23456]]]
  ExpectationSweepJobResult:
    description:  Result of the expectation sweep job type.
    allOf:
      - $ref: "#/definitions/JobResult"
    properties:
      result:
        description:  |
            A list of expectation-value lists. The outer index determines the
            sweep, and the inner index determines the observable. For instance,
            results[1][3] would select the fourth observable measured in the
            second sweep.
        type: array
        items:
          type: array
          items:
            type: number
        example: [[1.23456]]
  ExpectationJobStatusEvent:
    description: |
      Expectation job status event.

      The data field depends on the submitted expectation job type and could be
      one of the following types:
        - ExpectationJobResult
        - ExpectationBatchJobResult
        - ExpectationSweepJobResult
    allOf:
      - $ref: "#/definitions/ServerEvent"
    items:
      properties:
        event:
          default: ExpectationJobStatusEvent
        data:
          $ref: "#/definitions/ExpectationJobResult"
  # END [Expectation endpoints types]

  # START [Sample endpoints types]
  SampleJobContext:
    description: New sample job context.
    allOf:
      - $ref: "#/definitions/JobContext"
    properties:
      repetitions:
        $ref: "#/definitions/Repetitions"
  SampleBatchJobContext:
    description: New sample batch job context.
    allOf:
      - $ref: "#/definitions/BatchJobContext"
    properties:
      repetitions:
        description: |
          Number of acyclic_graph repetitions to run. Can be specified as a single
          value to use for all runs, or as a list of values, one for each
          acyclic_graph.
        type: array
        items:
          type: number
    required:
      - operators
  SampleSweepJobContext:
    description: New sample sweep job context.
    allOf:
      - $ref: "#/definitions/SweepJobContext"
    properties:
      repetitions:
        $ref: "#/definitions/Repetitions"
  SampleJobResult:
    description: Result of the sample job type.
    allOf:
      - $ref: "#/definitions/JobResult"
    properties:
      result:
        $ref: "#/definitions/Result"
  SampleBatchJobResult:
    description: Result of the sample batch job type.
    allOf:
      - $ref: "#/definitions/JobResult"
    properties:
      result:
        description: |
          A list of lists of TrialResults. The outer list corresponds to the
          acyclic_graphs, while each inner list contains the TrialResults for the
          corresponding acyclic_graph, in the order imposed by the associated
          parameter sweep.
        type: array
        items:
          type: array
          items:
            $ref: "#/definitions/Result"
  SampleSweepJobResult:
    description: Result of the sample sweep job type.
    allOf:
      - $ref: "#/definitions/JobResult"
    properties:
      result:
        description: |
          Result list for this run, one for each possible parameter resolver.
        type: array
        items:
          $ref: "#/definitions/Result"
  SampleJobStatusEvent:
    description: |
      Sample job status event.

      The data field depends on the submitted sample job type and could be
      one of the following types:
        - SampleJobResult
        - SampleBatchJobResult
        - SampleSweepJobResult
    allOf:
      - $ref: "#/definitions/ServerEvent"
    items:
      properties:
        event:
          default: SampleJobStatusEvent
        data:
          $ref: "#/definitions/SampleJobResult"
  # END [Sample endpoints types]

  # START [Tasks relevant types]
  TaskState:
    description: |
      Current task state:
      * TASK_STATE_PENDING = 0
      * TASK_STATE_RUNNING = 1
      * TASK_STATE_DONE = 2
    type: integer
    enum: [0, 1, 2]
    example: 2
  TaskStatus:
    description: Asynchronous task status.
    type: object
    properties:
      error:
        $ref: "#/definitions/ErrorMessage"
      state:
        $ref: "#/definitions/TaskState"
      success:
        description: Indicates if task completed successfully.
        type: boolean
        example: false
  TaskStatusEvent:
    allOf:
      - $ref: "#/definitions/ServerEvent"
    items:
      properties:
        event:
          default: TaskStatusEvent
        data:
          $ref: "#/definitions/TaskStatus"
  TaskSubmitted:
    description: Request asynchronous task response.
    type: object
    properties:
      id:
        allOf:
          - $ref: "#/definitions/UUID"
        description: Unique task id.
  # END [Tasks relevant types]

  # START [Worker endpoints types]
  WorkerState:
    description: |
      Current worker state:
      * WORKER_STATE_OFFLINE = 0
      * WORKER_STATE_SHUTTING_DOWN = 1
      * WORKER_STATE_IDLE = 2
      * WORKER_STATE_BOOTING = 3
      * WORKER_STATE_PROCESSING = 4
      * WORKER_STATE_ERROR = 5
    type: integer
    enum: [0, 1, 2, 3, 4, 5]
    example: 2
  WorkerStatus:
    description: Current worker status.
    type: object
    properties:
      error:
        $ref: "#/definitions/ErrorMessage"
      job_id:
        $ref: "#/definitions/JobId"
      state:
        $ref: "#/definitions/WorkerState"
  # END [Worker endpoints types]
