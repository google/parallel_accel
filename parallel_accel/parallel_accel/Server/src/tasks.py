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
"""This module defines asynchronous tasks manager."""
import functools
import typing
import uuid
import sanic

from parallel_accel.shared import logger
from . import redis


class TasksManager:
    """Schedules asynchronous tasks."""

    class _wrap_target_func:  # pylint: disable=invalid-name
        """A helper decorator for post_coro and post_function methods. It
        applies functools.partial() on the input coroutine/function argument.
        """

        def __init__(self, handler: typing.Callable) -> None:
            """Creates _wrap_target_func class instance.

            Args:
                handler: Instance method to be decorated.
            """
            self.handler = handler

        def __get__(self, instance: "TasksManager", *args) -> functools.partial:
            return functools.partial(self.__call__, instance)

        def __call__(
            self,
            instance: "TasksManager",
            func: typing.Callable,
            *args,
            **kwargs,
        ) -> typing.Callable:
            if not isinstance(func, functools.partial):
                func = functools.partial(func, *args, **kwargs)
            return self.handler(instance, func)

    class AsyncTask:
        """A helper class that executes input coroutine and reports progress to
        the Redis store."""

        _store: redis.TasksRedisStore = None

        def __init__(self, coro: functools.partial) -> None:
            """Creates AsyncTask class instance.

            Args:
                coro: partial coroutine function.
            """
            self._coro = coro
            self._id = uuid.uuid4()
            self._logger = logger.get_logger(
                self.__class__.__name__, task_id=str(self._id)
            )

            self._store.create_task(self._id)

        @property
        def id(self) -> uuid.UUID:  # pylint: disable=invalid-name
            """Unique task id."""
            return self._id

        async def __call__(self) -> None:
            self._logger.debug("Task started")

            try:
                logger.context.clear()
                logger.context.bind(task_id=str(self._id))

                self._store.set_running(self._id)

                self._logger.debug("Calling coroutine function")
                await self._coro()
                self._logger.debug("Coroutine done")

                success, error = True, None
            except Exception as err:  # pylint: disable=broad-except
                self._logger.error("Coroutine failed", exc_info=err)

                # TODO: Figure out how to pass more detailed, user friendly
                # error messages.
                success, error = False, repr(err)
            finally:
                self._store.set_completed(self._id, success, error)

                logger.context.clear()

            self._logger.debug("Task done", success=success, error=error)

        @staticmethod
        def initialize(store: redis.TasksRedisStore) -> None:
            """Sets TasksRedisStore instance.

            Args:
                store: TasksRedisStore instance.
            """
            TasksManager.AsyncTask._store = (
                store  # pylint: disable=protected-access
            )

    def __init__(self) -> None:
        """Creates TasksManager class instance."""
        self._app: sanic.Sanic = None
        self._logger = logger.get_logger(self.__class__.__name__)

    def initialize(
        self,
        app: sanic.Sanic,
        store: redis.TasksRedisStore,
    ) -> None:
        """Sets Sanic application.

        Args:
            app: Instance of Sanic application.
            store: Instance of TasksRedisStore.
        """
        self._app = app
        TasksManager.AsyncTask.initialize(store)

    @_wrap_target_func
    def post_coro(self, coro: typing.Coroutine, *_args, **_kwargs) -> uuid.UUID:
        """Posts coroutine to the event loop.

        Args:
            coro: Coroutine to be posted.

        Returns:
            Unique task id.
        """
        task = TasksManager.AsyncTask(coro)
        self._logger.debug("Task created", task_id=str(task.id))
        self._app.add_task(task())
        return task.id

    @_wrap_target_func
    def post_function(
        self,
        func: typing.Callable,
        *_args,
        **_kwargs,
    ) -> uuid.UUID:
        """Posts synchronous function to the event loop.

        Args:
            func: Function to be posted.

        Returns:
            Unique task id.
        """

        async def wrap() -> typing.Coroutine:
            """Runs synchronous IO blocking function in the background thread.

            The Sanic application provides method to add asynchronous background
            tasks to its event loop. In order to use it with the synchronous
            functions we need to schedule it manually to specific executor.

            Returns:
                Decorated function.
            """
            loop = sanic.app.get_event_loop()
            result = await loop.run_in_executor(None, func)
            return result

        return self.post_coro(wrap)
