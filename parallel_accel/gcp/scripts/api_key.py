#!/bin/python3
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
"""An utility script for managing ParallelAccel client API keys"""

import argparse
import enum
import json
import logging
import subprocess
import sys
import typing


class ServiceEnvironment(enum.Enum):
    """Supported service environments."""

    DEVELOPMENT = "parallel_accel-dev:symplectic-x99"
    STAGING = "parallel_accel:symplectic-x99"
    PRODUCTION = "parallel_accel:sandbox-at-alphabet-parallel_accel-prod"

    @property
    def project(self) -> str:
        """Google Cloud project for the service environment."""
        _, project = self.value.split(":")
        return project

    @property
    def service_name(self) -> str:
        """Cloud Endpoints full service name."""
        service, project = self.value.split(":")
REDACTED"

    @property
    def service(self) -> str:
        """Cloud Endpoints service prefix."""
        service, _ = self.value.split(":")
        return service


class LabelAction(argparse.Action):  # pylint: disable=too-few-public-methods
    """A helper class for converting input label argument into API key display
    name."""

    ARG_NAME = "label"

    def __call__(
        self,
        _: argparse.ArgumentParser,
        args: argparse.Namespace,
        value: str,
        __: typing.Optional[str] = None,
    ) -> None:
        """See base class documentation."""
        setattr(args, self.ARG_NAME, f"{args.service.service}-{value}")


class ServiceAction(argparse.Action):  # pylint: disable=too-few-public-methods
    """A helper class for converting input environment argument into
    ServiceEnvironment enum."""

    ARG_NAME = "service"

    def __call__(
        self,
        _: argparse.ArgumentParser,
        args: argparse.Namespace,
        value: str,
        __: typing.Optional[str] = None,
    ) -> None:
        """See base class documentation."""
        setattr(args, self.ARG_NAME, ServiceEnvironment[value.upper()])


class APIKeysManager:
    """Manages ParallelAccel Client API keys."""

    def __init__(self, env: ServiceEnvironment, silent: bool = True) -> None:
        """Creates APIKeysManager class instance.

        Args:
            environment: Target service environment.
            silent: Surpress logger output.
        """
        self._env = env
        self._keys = {}

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.disabled = silent

        self._logger.debug("Fetching API keys")
        stdout, _ = self._execute("list")
        for key in json.loads(stdout):
            self._keys[key["displayName"]] = key

    def create(self, label: str) -> typing.Tuple[str, str]:
        """Creates API keys.

        Args:
            labels: List of API keys display labels.

        Returns:
            Tuple object of the display name and the API key.

        Throws:
            CalledProcessError if failed to run the command.
        """
        self._logger.debug("Looking for %s API key", label)

        if label in self._keys:
            self._logger.debug("Key already exists, updating access scope...")
            command = "update"
            args = [self._keys[label]["name"]]
        else:
            self._logger.debug("Key does not exist, creating...")
            command = "create"
            args = [f"--display-name={label}"]

        args.append(f"--api-target=service={self._env.service_name}")

        _, stderr = self._execute(command, *args)
        stderr = stderr.decode()

        result = json.loads(stderr[stderr.find("{") :])
        return (label, result["keyString"])

    def delete(self, label: str) -> None:
        """Deletes the API key.

        Args:
            label: API key display name.

        Returns:
            List of matched API keys.

        Throws:
            CalledProcessError if failed to remove the API key.
        """
        self._logger.debug("Looking for %s API key", label)

        if label not in self._keys:
            self._logger.debug("Key does not exist")
            return

        self._logger.debug("Removing %s API key", label)
        self._execute("delete", self._keys[label]["name"])
        del self._keys[label]

    def get(self, label: str) -> typing.Optional[typing.Tuple[str, str]]:
        """Gets API key for given label.

        Args:
            label: API key display name.
            secret: Get API key string.

        Returns:
            List of matched API keys.
        """
        self._logger.debug("Looking for %s API key", label)

        if label not in self._keys:
            self._logger.debug("Key does not exist")
            return None

        self._logger.debug("Key exists, fetching credentials...")
        stdout, _ = self._execute("get-key-string", self._keys[label]["name"])
        credentials = json.loads(stdout)
        return label, credentials["keyString"]

    def _execute(self, command: str, *args, **kwargs) -> bytes:
        """Runs gcloud command.

        Executes `gcloud alpha services api-keys {command}` shell command,
        captures the output and returns parsed result.

        Args:
            command: Input command.

        Returns:
            Raw command output.

        Throws:
            CalledProcessError if the command failed.
        """
        result = subprocess.run(
            [
                "gcloud",
                "alpha",
                "services",
                "api-keys",
                command,
                *args,
                "--format=json",
                f"--project={self._env.project}",
            ],
            check=True,
            capture_output=True,
            **kwargs,
        )
        return result.stdout, result.stderr


def main() -> None:
    """Script entry point."""
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        "ParallelAccel Client API key management tool",
        description="An utility script for API keys management",
    )
    parser.add_argument(
        "command",
        help="command to be invoked",
        choices=["create", "delete", "get"],
    )
    parser.add_argument(
        f"--{ServiceAction.ARG_NAME}",
        help="Cloud Endpoints service to be provisioned with the API key.",
        choices=[x.name.lower() for x in ServiceEnvironment],
        required=True,
        action=ServiceAction,
    )
    parser.add_argument(
        "--label",
        help="API key display label",
        required=True,
        action=LabelAction,
    )
    parser.add_argument(
        "-s",
        "--silent",
        help="Silent mode",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    manager = APIKeysManager(getattr(args, ServiceAction.ARG_NAME), args.silent)
    result = getattr(manager, args.command)(args.label)
    if result:
        print(result)


if __name__ == "__main__":
    main()
