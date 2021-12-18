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
"""Top Level ASIC Simulator Runner."""
import os
import signal
import traceback
import functools
from types import FrameType
from typing import Any, Callable

from parallel_accel.shared import logger, schemas
from parallel_accel.shared.schemas import (
    ExpectationBatchJobContext,
    ExpectationBatchJobResultSchema,
    ExpectationJobContext,
    ExpectationJobResultSchema,
    ExpectationSweepJobContext,
    ExpectationSweepJobResultSchema,
    JobProgress,
    SampleBatchJobContext,
    SampleBatchJobResultSchema,
    SampleJobContext,
    SampleJobResultSchema,
    SampleSweepJobContext,
    SampleSweepJobResultSchema,
)
from parallel_accel.shared.redis import workers

from asic_la import asic_simulator
import context_validator
import sim_jobs_manager

log = logger.get_logger(__name__)


class ASICSimRunner:
    """Class responsible for claiming and running Jobs with a simulator.

    This class instantiates a simulator, listens for jobs with a JobManager,
    and dispatches jobs to a simulator.
    """

    def __init__(
        self,
        api_key: str,
        jobs_manager: sim_jobs_manager.SimJobsManager,
        worker_manager: workers.WorkersRedisStore,
        simulator: asic_simulator.ASICSimulator,
    ) -> None:
        """Creates ASICSimRunner class instance.

        Args:
            api_key: API key to be used for polling new jobs.
            jobs_manager: Reference to SimJobsManager object.
            worker_manager: Reference to WorkersRedisStore object.
            simulator: Reference to ASICSimulator object.
        """
        self._api_key = api_key
        self._jobs_manager = jobs_manager
        self._worker_manager = worker_manager
        self._sim = simulator

        self._running = True

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self.signal_handler)

    def signal_handler(self, signum: int, _frame: FrameType) -> None:
        """Handles signals.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        log.warning("Received a new signal", signal=signum)

        self._worker_manager.set_shutting_down(self._api_key)
        self._running = False

    def _dispatch_job(self, job: schemas.Job) -> Any:
        """Dispatch a fully deserialized job to the simulator.

        Args:
            job: Job to be sent to the simulator.

        Returns:
            Simulation result.
        """

        def progress_callback(completed: int, total: int) -> None:
            self._jobs_manager.update_job_progress(
                job.id, JobProgress(completed, total)
            )

        context_validator.validate(job.context)

        if job.type == schemas.JobType.SAMPLE:
            result = self._dispatch_sample_job(job.context, progress_callback)
        elif job.type == schemas.JobType.EXPECTATION:
            result = self._dispatch_expectation_job(
                job.context, progress_callback
            )

        return result

    @functools.singledispatchmethod
    def _dispatch_sample_job(self, *args):
        raise NotImplementedError()

    @functools.singledispatchmethod
    def _dispatch_expectation_job(self, *args):
        raise NotImplementedError()

    @_dispatch_sample_job.register
    def _(self, context: schemas.SampleJobContext, progress_callback: Callable):
        result = self._sim.compute_samples(
            context.acyclic_graph,
            context.param_resolver,
            context.repetitions,
            progress_callback,
        )
        return result

    @_dispatch_sample_job.register
    def _(
        self,
        context: schemas.SampleSweepJobContext,
        progress_callback: Callable,
    ):
        result = self._sim.compute_samples_sweep(
            context.acyclic_graph,
            context.params,
            context.repetitions,
            progress_callback,
        )
        return result

    @_dispatch_sample_job.register
    def _(
        self,
        context: schemas.SampleBatchJobContext,
        progress_callback: Callable,
    ):
        result = self._sim.compute_samples_batch(
            context.acyclic_graphs,
            context.params,
            context.repetitions,
            progress_callback,
        )
        return result

    @_dispatch_expectation_job.register
    def _(
        self,
        context: schemas.ExpectationJobContext,
        progress_callback: Callable,
    ):
        result = self._sim.compute_expectations(
            context.acyclic_graph,
            context.operators,
            context.param_resolver,
            progress_callback,
        )
        result = result.tolist()
        return result

    @_dispatch_expectation_job.register
    def _(
        self,
        context: schemas.ExpectationSweepJobContext,
        progress_callback: Callable,
    ):
        result = self._sim.compute_expectations_sweep(
            context.acyclic_graph,
            context.operators,
            context.params,
            progress_callback,
        )
        result = [r.tolist() for r in result]
        return result

    @_dispatch_expectation_job.register
    def _(
        self,
        context: schemas.ExpectationBatchJobContext,
        progress_callback: Callable,
    ):
        result = self._sim.compute_expectations_batch(
            context.acyclic_graphs,
            context.operators,
            context.params,
            progress_callback,
        )
        result = [[r.tolist() for r in inner] for inner in result]
        return result

    def _process_job(self, job: schemas.Job) -> None:
        """Dispatches job to the simulator, updates its status and returning
        results to job manager on completion.

        Args:
            job: Job to be processed.
        """
        schema_map = {
            SampleJobContext: SampleJobResultSchema,
            SampleSweepJobContext: SampleSweepJobResultSchema,
            SampleBatchJobContext: SampleBatchJobResultSchema,
            ExpectationJobContext: ExpectationJobResultSchema,
            ExpectationSweepJobContext: ExpectationSweepJobResultSchema,
            ExpectationBatchJobContext: ExpectationBatchJobResultSchema,
        }

        try:
            result = self._dispatch_job(job)
            self._jobs_manager.set_job_complete(
                job.id, result, schema_map[type(job.context)]
            )
            log.info("job finished")
        except (context_validator.ValidationError, Exception) as error:
            message = "Simulator internal error occured"
            if isinstance(error, context_validator.ValidationError):
                message = str(error)

            self._jobs_manager.set_job_error(job.id, message)
            log.error("job failed", exc_info=error)

    def run(self) -> None:
        """Starts main loop.

        Polls Redis store for any new jobs and schedules them for execution,
        one at the time.
        """
        log.info("Listening for jobs", api_key=self._api_key)
        self._worker_manager.set_idle(self._api_key)

        while self._running:
            try:
                job = self._jobs_manager.get_next_job(self._api_key)
                if not job:
                    continue

                logger.context.bind(job_id=str(job.id))

                log.info("Job claimed", job_type=job.type.name)

                self._worker_manager.set_processing_job(self._api_key, job.id)
                self._jobs_manager.set_job_in_progress(job.id)

                self._process_job(job)

                self._jobs_manager.clear_next_job(self._api_key)

                self._worker_manager.set_idle(self._api_key)

                logger.context.unbind("job_id")
            except Exception as error:  # pylint: disable=broad-except
                self._running = False

                log.error(
                    "Unexepected simulator exception", exception=repr(error)
                )
                traceback.print_exc()
                self._worker_manager.set_error(
                    self._api_key, "Internal Simulator Error"
                )

        log.info("Quitting...")


def main() -> None:
    """Script entry point."""
    api_key = os.environ.get("API_KEY", None)
    if not api_key:
        raise RuntimeError("API_KEY environment variable was not set")

    jobs_manager = sim_jobs_manager.SimJobsManager()
    worker_manager = workers.WorkersRedisStore()
    simulator = asic_simulator.ASICSimulator()

    runner = ASICSimRunner(api_key, jobs_manager, worker_manager, simulator)
    runner.run()


if __name__ == "__main__":
    main()
