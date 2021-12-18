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
"""This module defines ASIC workers management components."""
import asyncio
import datetime
import enum
import functools
import multiprocessing
import multiprocessing.pool
import time
import typing
import google.auth
import google.auth.compute_engine
import google.cloud.container
import kubernetes
import kubernetes.client
import kubernetes.watch
import sanic

from parallel_accel.shared import logger, redis, schemas, utils


class WorkerCommand(enum.IntEnum):
    """Supported worker commands."""

    RESTART = enum.auto()
    START = enum.auto()
    STOP = enum.auto()

    @property
    def asic_cluster_event(self) -> str:
        """Expected Kubernetes asic_cluster event."""
        events_map = {
            WorkerCommand.RESTART: "ADDED",
            WorkerCommand.START: "ADDED",
            WorkerCommand.STOP: "DELETED",
        }

        return events_map[self]

    @property
    def replicas_count(self) -> int:
        """Deployment replicas count.

        Returns:
            Number of replicas count.

        Throws:
            KeyError for RESTART command.
        """
        replicas_map = {
            WorkerCommand.START: 1,
            WorkerCommand.STOP: 0,
        }

        return replicas_map[self]


class ASICWorkerManager:
    """A helper class for managing ASIC workers."""

    _DEFAULT_API_KWARGS = {"async_req": True, "namespace": "default"}

    _MAXIMUM_IDLING_TIME = datetime.timedelta(hours=3).total_seconds()

    def __init__(self, workers_store: redis.WorkersRedisStore) -> None:
        """Creates ASICWorkerManager class instance.

        Args:
            workers_store: Reference to WorkersRedisStore class instance.
        """
        self._app: typing.Optional[sanic.Sanic] = None

        self._credentials: google.auth.credentials.Credentials = None
        self._workers_store = workers_store

        self._logger = logger.get_logger(self.__class__.__name__)

    async def handle_command(
        self,
        api_key: str,
        command: WorkerCommand,
    ) -> None:
        """Handles worker command.

        Args:
            api_key: API key associated with the worker.
            command: Worker command.
        """
        await self._refresh_credentials()

        logger.context.bind(api_key=api_key)

        hashed_key = utils.sha1(api_key)
        kwargs = {}

        if command in (WorkerCommand.START, WorkerCommand.STOP):
            kwargs["replicas"] = command.replicas_count
            handler = self._handle_start_stop_command
        else:
            handler = self._handle_restart_command

        self._logger.debug("Handling ASIC worker command", command=command.name)
        await handler(hashed_key, **kwargs)

        await self._loop.run_in_executor(
            None, self._wait_for_asic_cluster_event, api_key, command.asic_cluster_event
        )

        if command in (WorkerCommand.START, WorkerCommand.RESTART):
            await self._loop.run_in_executor(
                None, self._wait_for_asic_cluster_readiness, api_key
            )

        logger.context.unbind("api_key")

    async def initialize(
        self,
        project: str,
        cluster: str,
        app: typing.Optional[sanic.Sanic] = None,
    ) -> None:
        """Initializes ASICWorkerManager.

        Args:
            project: Name of the GCP project.
            cluster: Name of the GKE cluster.
            app: Instance of Sanic application.
        """
        self._logger.debug("Initializing...")
        self._app = app

        # Get Google API credentials
        func = functools.partial(
            google.auth.default,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self._credentials, _ = await self._loop.run_in_executor(None, func)
        await self._refresh_credentials()

        # Access GKE cluster
        self._logger.debug(f"Accessing the {cluster} cluster")
        request = google.cloud.container.GetClusterRequest()
        request.name = (
            f"projects/{project}/locations/us-central1/clusters/{cluster}"
        )

        manager = google.cloud.container.ClusterManagerAsyncClient(
            credentials=self._credentials
        )
        cluster = await manager.get_cluster(request)

        # Configure Kubernetes client credentials
        config = kubernetes.client.Configuration.get_default_copy()
        config.host = f"https://{cluster.endpoint}:443"
        config.verify_ssl = False

        self._logger.debug("Setting Kubernetes client default credentials")
        kubernetes.client.Configuration.set_default(config)

        self._logger.info("Done initialization")

    async def stop_idling_workers(self) -> None:
        """Stops any idling workers."""
        self._logger.debug("Looking for idling workers")

        count = 0
        api_keys = self._workers_store.get_workers_ids()
        for api_key in api_keys:
            worker = self._workers_store.get_worker(api_key)
            if (
                worker.state == schemas.WorkerState.IDLE
                and (int(time.time()) - worker.job_timestamp)
                > self._MAXIMUM_IDLING_TIME
            ):
                self._logger.debug(
                    f"Sending {WorkerCommand.STOP.name} command to the worker",
                    api_key=api_key,
                    last_job_timestamp=worker.job_timestamp,
                )
                await self.handle_command(api_key, WorkerCommand.STOP)
                count += 1

        self._logger.debug(f"Stopped {count} of {len(api_keys)} workers")

    @property
    def _loop(self) -> asyncio.AbstractEventLoop:
        """Currently running event loop."""
        if self._app is not None:
            return self._app.loop

        return asyncio.get_event_loop()

    async def _refresh_credentials(self) -> None:
        """Checks whether Google API credentials have expired and refreshes them
        if needed."""
        if not self._credentials.expired and self._credentials.valid:
            return

        self._logger.debug("Refreshing Google Credentials")

        await self._loop.run_in_executor(
            None,
            self._credentials.refresh,
            google.auth.transport.requests.Request(),
        )

        config = kubernetes.client.Configuration.get_default_copy()
        config.api_key = {"authorization": f"Bearer {self._credentials.token}"}

        self._logger.debug("Updating Kubernetes client credentials")
        kubernetes.client.Configuration.set_default(config)

    async def _handle_start_stop_command(self, key: str, replicas: int) -> None:
        """Handles start/stop command.

        Args:
            key: SHA1 encoded API key.
            replicas: Target number of deployment replicas count.
        """
        client = kubernetes.client.AppsV1Api()

        self._logger.debug(
            "Looking for deployments matching given API key",
        )
        deployments = await self._get_resources_names(
            client.list_namespaced_deployment, key
        )
        self._logger.debug(
            f"Found {len(deployments)} matching deployments",
        )

        kwargs = dict(self._DEFAULT_API_KWARGS)
        kwargs["body"] = {
            "spec": {
                "replicas": replicas,
            }
        }
        for deployment in deployments:
            self._logger.info(
                "Scaling ASIC worker deployment",
                deployment=deployment,
                replicas=kwargs["body"]["spec"]["replicas"],
            )
            thread = client.patch_namespaced_deployment_scale(
                deployment,
                **kwargs,
            )
            await self._wait_for_thread(thread)

        self._logger.debug("Done scaling deployments")

    async def _handle_restart_command(self, key: str) -> None:
        """Handles restart command.

        Args:
            key: SHA1 encoded API key.
        """
        client = kubernetes.client.CoreV1Api()

        self._logger.debug("Looking for asic_clusters matching given API key")
        asic_clusters = await self._get_resources_names(client.list_namespaced_asic_cluster, key)
        self._logger.debug(f"Found {len(asic_clusters)} matching asic_clusters")

        kwargs = dict(self._DEFAULT_API_KWARGS)
        for asic_cluster in asic_clusters:
            self._logger.info(f'Deleting "{asic_cluster}" asic_cluster', asic_cluster=asic_cluster)
            thread = client.delete_namespaced_asic_cluster(asic_cluster, **kwargs)
            await self._wait_for_thread(thread)

        self._logger.debug("Done restarting asic_clusters")

    async def _get_resources_names(
        self, func: typing.Callable, resource_id: typing.Optional[str] = None
    ) -> typing.List[str]:
        """Gets Kubernetes resources names.

        Args:
            func: API function to be called.
            resource_id: Optional "id" label filter.

        Returns:
            List of resources names.
        """
        kwargs = dict(self._DEFAULT_API_KWARGS)
        if resource_id:
            kwargs["label_selector"] = f"id={resource_id}"

        thread = func(**kwargs)
        await self._wait_for_thread(thread)

        resource = thread.get()
        return [x.metadata.name for x in resource.items]

    async def _wait_for_thread(
        self,
        thread: multiprocessing.pool.AsyncResult,
    ) -> None:
        """Waits until thread is ready.

        Args:
            thread: AsyncResult object.
        """
        self._logger.debug("Waiting for the AsyncResult to complete")
        await self._loop.run_in_executor(None, thread.wait)

    def _wait_for_asic_cluster_event(self, api_key: str, target_event: str) -> None:
        """Waits for asic_cluster to be created or removed.

        Args:
            api_key: API key associated with the ASIC workers.
            target_event: Target asic_cluster event ("ADDED" or "DELETED").
        """

        def handler(event: dict, watch: kubernetes.watch.Watch) -> None:
            """Callback function to be triggered on a new event.

            Listens for asic_cluster state change event ("ADDED" or "DELETED") and
            updates ASIC worker state in the Redis store.

            Args:
                event: Received event.
                watch: Watch object.
            """
            if event["type"] == "ADDED" == target_event:
                self._workers_store.set_booting(api_key)
            elif event["type"] == "DELETED" == target_event:
                self._workers_store.set_offline(api_key)

            if event["type"] == target_event:
                watch.stop()

        hashed_key = utils.sha1(api_key)
        self._logger.debug(f"Waiting for asic_cluster to be {target_event.lower()}")

        self._watch_asic_cluster_events(hashed_key, handler, 60)

        self._logger.debug(f"ASICCluster {target_event.lower()}")

    def _wait_for_asic_cluster_readiness(self, api_key: str) -> None:
        """Waits for asic_cluster to be ready.

        Args:
            api_key: API key associated with the ASIC workers.
        """

        def handler(event: dict, watch: kubernetes.watch.Watch) -> None:
            """Callback function to be triggered on a new event.

            Checks whether asic_cluster is in "Running" phase and stops watch when asic_cluster
            is ready.

            Args:
                event: Received event.
                watch: Watch object.
            """
            if event["object"].status.phase == "Running":
                watch.stop()

        hashed_key = utils.sha1(api_key)
        self._logger.debug("Waiting for asic_cluster to be ready")

        self._watch_asic_cluster_events(hashed_key, handler)

        self._logger.debug("ASICCluster ready")

    def _watch_asic_cluster_events(
        self,
        hashed_key: str,
        handler: typing.Callable[[dict, kubernetes.watch.Watch], None],
        timeout: typing.Optional[int] = None,
    ) -> None:
        """Listens to events emitted by asic_clusters associated with given API. Every
        time a new event is received, the handler callback function is called.

        Args:
            hashed_key: SHA1 encoded API key associated with the ASIC worker.
            handler: Function to be called after receiving a new event.
            timeout: Maximum number of seconds to watch for incoming events. If
                not specified, the watch will run indefinitely.
        """
        watch = kubernetes.watch.Watch()
        client = kubernetes.client.CoreV1Api()

        kwargs = {
            "func": client.list_namespaced_asic_cluster,
            "label_selector": f"id={hashed_key}",
            "namespace": self._DEFAULT_API_KWARGS["namespace"],
        }
        if timeout is not None:
            kwargs["timeout_seconds"] = timeout

        self._logger.debug("Watching for asic_cluster events")

        for event in watch.stream(**kwargs):
            self._logger.debug("Received a new asic_cluster event", type=event["type"])
            handler(event, watch)

        self._logger.debug("Done watching for asic_cluster events")
