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
# pylint: disable=c-extension-no-member

"""This module defines application structure

Instead of manually creating objects and managing their instances, we are using
dependency_injector framework to declare application structure and dependencies
between the components.
"""
import dependency_injector.containers
import dependency_injector.providers

from parallel_accel.shared import redis as shared_redis

from . import app, blueprints, redis, tasks, worker_manager


class ApplicationContainer(
    dependency_injector.containers.DeclarativeContainer
):  # pylint: disable=too-few-public-methods
    """Declares application structure."""

    jobs_manager = dependency_injector.providers.Singleton(
        redis.JobsManager,
    )
    tasks_store = dependency_injector.providers.Singleton(redis.TasksRedisStore)
    workers_store = dependency_injector.providers.Singleton(
        shared_redis.WorkersRedisStore
    )

    tasks_manager = dependency_injector.providers.Singleton(tasks.TasksManager)
    worker_manager = dependency_injector.providers.Singleton(
        worker_manager.ASICWorkerManager, workers_store
    )

    sanic_app = dependency_injector.providers.Factory(
        app.Application,
        blueprints=dependency_injector.providers.List(
            dependency_injector.providers.Factory(
                blueprints.ExpectationBlueprint, jobs_manager, workers_store
            ),
            dependency_injector.providers.Factory(
                blueprints.JobsQueueBlueprint, jobs_manager, tasks_manager
            ),
            dependency_injector.providers.Factory(
                blueprints.SampleBlueprint, jobs_manager, workers_store
            ),
            dependency_injector.providers.Factory(
                blueprints.TasksBlueprint, tasks_store
            ),
            dependency_injector.providers.Factory(
                blueprints.WorkerBlueprint,
                workers_store,
                tasks_manager,
                worker_manager,
            ),
        ),
    )
