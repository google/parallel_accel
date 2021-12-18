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
"""Logging module for the ParallelAccel project.

This is a front-end for the ParallelAccel logger. It should be used by every component
within this project. Refer to the README.md for how to use this module.
"""
import os
import structlog

from . import utils


class LoggerContextManager:
    """A helper class that manages logger local context variables."""

    @staticmethod
    def bind(**kwargs) -> None:
        """Binds logger local context variables."""
        structlog.threadlocal.bind_threadlocal(**kwargs)

    @staticmethod
    def clear() -> None:
        """Clears logger local context variables."""
        structlog.threadlocal.clear_threadlocal()

    @staticmethod
    def unbind(*args) -> None:
        """Unbinds logger local context variables."""
        structlog.threadlocal.unbind_threadlocal(*args)


class LoggerProcessors:  # pylint: disable=too-few-public-methods
    """A helper class that wraps custom logging processors."""

    @staticmethod
    def api_key_hasher(_, __, event_dict: dict) -> dict:
        """Custom processor that hashes API key using SHA1 algorithm.

        Args:
            event_dict: Dictionary of log event context.

        Returns:
            Dictionary of log event context.
        """

        api_key = event_dict.get("api_key", None)
        if api_key:
            event_dict["api_key"] = utils.sha1(api_key)

        return event_dict


class LoggerProvider:
    """Configures and provides logger.

    Properties:
        context: Reference to LoggerContextManager class instance.
    """

    context = LoggerContextManager()

    @staticmethod
    def setup(run_in_gke: bool = False) -> None:
        """Creates Logger class instance."""
        processors = [
            structlog.dev.set_exc_info,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.threadlocal.merge_threadlocal,
            LoggerProcessors.api_key_hasher,
        ]

        if run_in_gke:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.extend(
                [
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.dev.ConsoleRenderer(),
                ]
            )

        structlog.configure(
            cache_logger_on_first_use=False,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            processors=processors,  # type: ignore
            wrapper_class=structlog.stdlib.BoundLogger,
        )

    @staticmethod
    def get_logger(name, **kwargs) -> structlog.BoundLogger:
        """Gets Logger object.

        Args:
            name: Logger name.

        Returns:
            Configured Logger object.
        """
        if not structlog.is_configured():
            run_in_gke = any(x.startswith("KUBERNETES_") for x in os.environ)
            LoggerProvider.setup(run_in_gke)

        return structlog.get_logger(name, component=name, **kwargs)
