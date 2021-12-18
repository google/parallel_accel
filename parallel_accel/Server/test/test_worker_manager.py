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
# pylint: disable=protected-access

"""Unit test for worker_manager module"""
import asyncio
import hashlib
import importlib
import multiprocessing
import os
import secrets
import time
import typing
import unittest
import unittest.mock
import aiounittest
import google.auth
import google.cloud.container
import kubernetes
import sanic

from parallel_accel.shared import redis, schemas

from src import worker_manager


class TestASICWorkerManager(aiounittest.AsyncTestCase):
    """Tests ASICWorkerManager class behavior."""

    API_KEY: str = secrets.token_hex(16)
    API_KEY_HASH: str = None

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        # Compute API key hash
        hasher = hashlib.sha1()
        hasher.update(cls.API_KEY.encode())
        cls.API_KEY_HASH = hasher.hexdigest()

        # Patch imports
        cls.patchers = []

        cls.mocked_appsv1api = unittest.mock.Mock(
            spec=kubernetes.client.AppsV1Api
        )
        cls.mocked_appsv1api.return_value = cls.mocked_appsv1api
        patcher = unittest.mock.patch(
            "kubernetes.client.AppsV1Api", cls.mocked_appsv1api
        )
        cls.patchers.append(patcher)

        cls.mocked_corev1api = unittest.mock.Mock(
            spec=kubernetes.client.CoreV1Api
        )
        cls.mocked_corev1api.return_value = cls.mocked_corev1api
        patcher = unittest.mock.patch(
            "kubernetes.client.CoreV1Api", cls.mocked_corev1api
        )
        cls.patchers.append(patcher)

        cls.mocked_watch = unittest.mock.Mock(spec=kubernetes.watch.Watch)
        cls.mocked_watch.return_value = cls.mocked_watch
        patcher = unittest.mock.patch(
            "kubernetes.watch.Watch", cls.mocked_watch
        )
        cls.patchers.append(patcher)

        for patcher in cls.patchers:
            patcher.start()

        cls.mocked_event_loop = unittest.mock.Mock(
            spec=asyncio.AbstractEventLoop
        )
        cls.mocked_event_loop.run_in_executor = unittest.mock.AsyncMock()

        cls.mocked_redis_store = unittest.mock.Mock(
            spec=redis.WorkersRedisStore
        )

        os.environ["GKE_CLUSTER"] = "test-cluster"
        os.environ["GCP_PROJECT"] = "test-project"

        importlib.reload(worker_manager)

        cls.mocked_sanic_app = unittest.mock.Mock(spec=sanic.Sanic)
        cls.mocked_sanic_app.loop = cls.mocked_event_loop

        cls.manager = worker_manager.ASICWorkerManager(cls.mocked_redis_store)
        cls.manager._app = cls.mocked_sanic_app

    @classmethod
    def tearDownClass(cls) -> None:
        """See base class documentation."""
        del os.environ["GKE_CLUSTER"]

        for patcher in cls.patchers:
            patcher.stop()

    def tearDown(self) -> None:
        """See base class documentation."""
        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

    async def test_handler_start_command(self) -> None:
        """Tests START worker command."""
        # Test setup
        meta = kubernetes.client.V1ObjectMeta(name="test-deployment-1")
        deployment = kubernetes.client.V1Deployment(metadata=meta)
        deployment_list = kubernetes.client.V1DeploymentList(items=[deployment])

        list_namespaced_deployment_thread = unittest.mock.Mock(
            spec=multiprocessing.pool.AsyncResult
        )
        list_namespaced_deployment_thread.get.return_value = deployment_list
        self.mocked_appsv1api.list_namespaced_deployment.return_value = (
            list_namespaced_deployment_thread
        )

        patch_namespaced_deployment_scale_thread = unittest.mock.Mock(
            spec=multiprocessing.pool.AsyncResult
        )
        self.mocked_appsv1api.patch_namespaced_deployment_scale.return_value = (
            patch_namespaced_deployment_scale_thread
        )

        self.mocked_watch.stream.return_value = [{"type": "ADDED"}]

        self.manager._credentials = unittest.mock.MagicMock()
        self.manager._credentials.expired = False
        self.manager._credentials.valid = True

        # Run test
        await self.manager.handle_command(
            self.API_KEY, worker_manager.WorkerCommand.START
        )

        # Verification
        kwargs = {
            "async_req": True,
            "namespace": "default",
            "label_selector": f"id={self.API_KEY_HASH}",
        }
        self.mocked_appsv1api.list_namespaced_deployment.assert_called_once_with(
            **kwargs
        )

        kwargs = {
            "async_req": True,
            "namespace": "default",
            "body": {"spec": {"replicas": 1}},
        }
        self.mocked_appsv1api.patch_namespaced_deployment_scale.assert_called_once_with(
            meta.name, **kwargs
        )

        self._verify_run_in_executor(
            [
                (
                    (
                        None,
                        list_namespaced_deployment_thread.wait,
                    ),
                ),
                (
                    (
                        None,
                        patch_namespaced_deployment_scale_thread.wait,
                    ),
                ),
                (
                    (
                        None,
                        self.manager._wait_for_asic_cluster_event,
                        self.API_KEY,
                        "ADDED",
                    ),
                ),
                (
                    (
                        None,
                        self.manager._wait_for_asic_cluster_readiness,
                        self.API_KEY,
                    ),
                ),
            ]
        )

    async def test_handler_stop_command(self) -> None:
        """Tests STOP worker command."""
        # Test setup
        meta = kubernetes.client.V1ObjectMeta(name="test-deployment-1")
        deployment = kubernetes.client.V1Deployment(metadata=meta)
        deployment_list = kubernetes.client.V1DeploymentList(items=[deployment])

        list_namespaced_deployment_thread = unittest.mock.Mock(
            spec=multiprocessing.pool.AsyncResult
        )
        list_namespaced_deployment_thread.get.return_value = deployment_list
        self.mocked_appsv1api.list_namespaced_deployment.return_value = (
            list_namespaced_deployment_thread
        )

        patch_namespaced_deployment_scale_thread = unittest.mock.Mock(
            spec=multiprocessing.pool.AsyncResult
        )
        self.mocked_appsv1api.patch_namespaced_deployment_scale.return_value = (
            patch_namespaced_deployment_scale_thread
        )

        self.mocked_watch.stream.return_value = [
            {"type": x for x in ("ADDED", "DELETED")}
        ]

        self.manager._credentials = unittest.mock.MagicMock()
        self.manager._credentials.expired = False

        # Run test
        await self.manager.handle_command(
            self.API_KEY, worker_manager.WorkerCommand.STOP
        )

        # Verification
        kwargs = {
            "async_req": True,
            "namespace": "default",
            "label_selector": f"id={self.API_KEY_HASH}",
        }
        self.mocked_appsv1api.list_namespaced_deployment.assert_called_once_with(
            **kwargs
        )

        kwargs = {
            "async_req": True,
            "namespace": "default",
            "body": {"spec": {"replicas": 0}},
        }
        self.mocked_appsv1api.patch_namespaced_deployment_scale.assert_called_once_with(
            meta.name, **kwargs
        )

        self._verify_run_in_executor(
            [
                (
                    (
                        None,
                        list_namespaced_deployment_thread.wait,
                    ),
                ),
                (
                    (
                        None,
                        patch_namespaced_deployment_scale_thread.wait,
                    ),
                ),
                (
                    (
                        None,
                        self.manager._wait_for_asic_cluster_event,
                        self.API_KEY,
                        "DELETED",
                    ),
                ),
            ]
        )

    async def test_handler_restart_command(self) -> None:
        """Tests RESTART worker command."""
        # Test setup
        meta = kubernetes.client.V1ObjectMeta(name="test-asic_cluster-1")
        asic_cluster = kubernetes.client.V1ASICCluster(metadata=meta)
        asic_cluster_list = kubernetes.client.V1ASICClusterList(items=[asic_cluster])

        list_namespaced_asic_cluster_thread = unittest.mock.Mock(
            spec=multiprocessing.pool.AsyncResult
        )
        list_namespaced_asic_cluster_thread.get.return_value = asic_cluster_list
        self.mocked_corev1api.list_namespaced_asic_cluster.return_value = (
            list_namespaced_asic_cluster_thread
        )

        delete_namespaced_asic_cluster_thread = unittest.mock.Mock(
            spec=multiprocessing.pool.AsyncResult
        )
        self.mocked_corev1api.delete_namespaced_asic_cluster.return_value = (
            delete_namespaced_asic_cluster_thread
        )

        self.manager._credentials = unittest.mock.MagicMock()
        self.manager._credentials.expired = False

        # Run test
        await self.manager.handle_command(
            self.API_KEY, worker_manager.WorkerCommand.RESTART
        )

        # Verification
        kwargs = {
            "async_req": True,
            "namespace": "default",
            "label_selector": f"id={self.API_KEY_HASH}",
        }
        self.mocked_corev1api.list_namespaced_asic_cluster.assert_called_once_with(
            **kwargs
        )

        kwargs = {"async_req": True, "namespace": "default"}
        self.mocked_corev1api.delete_namespaced_asic_cluster.assert_called_once_with(
            meta.name, **kwargs
        )

        self._verify_run_in_executor(
            [
                (
                    (
                        None,
                        list_namespaced_asic_cluster_thread.wait,
                    ),
                ),
                (
                    (
                        None,
                        delete_namespaced_asic_cluster_thread.wait,
                    ),
                ),
                (
                    (
                        None,
                        self.manager._wait_for_asic_cluster_event,
                        self.API_KEY,
                        "ADDED",
                    ),
                ),
                (
                    (
                        None,
                        self.manager._wait_for_asic_cluster_readiness,
                        self.API_KEY,
                    ),
                ),
            ]
        )

    @unittest.mock.patch("kubernetes.client.Configuration")
    @unittest.mock.patch("google.cloud.container.ClusterManagerAsyncClient")
    @unittest.mock.patch("google.cloud.container.GetClusterRequest")
    @unittest.mock.patch("google.auth.transport.requests.Request")
    @unittest.mock.patch("google.auth.default")
    async def test_initialize(
        self,
        _mocked_default: unittest.mock.Mock,
        mocked_request: unittest.mock.Mock,
        mocked_get_cluster_request: unittest.mock.Mock,
        mocked_cluster_manager: unittest.mock.Mock,
        mocked_configuration: unittest.mock.Mock,
    ) -> None:
        """Tests initialize method behavior."""
        cluster_name = "cluster-name"
        gcp_project = "gcp-project"

        mocked_request.return_value = mocked_request

        mocked_credentials = unittest.mock.Mock(
            spec=google.auth.compute_engine.Credentials
        )
        mocked_credentials.token = "API-TOKEN"
        mocked_credentials.expired = False
        mocked_credentials.valid = False
        self.mocked_event_loop.run_in_executor.return_value = (
            mocked_credentials,
            None,
        )

        mocked_get_cluster_request.return_value = mocked_get_cluster_request

        mocked_cluster = unittest.mock.Mock(spec=google.cloud.container.Cluster)
        mocked_cluster.endpoint = "1.2.3.4"
        mocked_cluster_manager.return_value = mocked_cluster_manager
        mocked_cluster_manager.get_cluster = unittest.mock.AsyncMock(
            return_value=mocked_cluster
        )

        mocked_configuration.get_default_copy.return_value = (
            mocked_configuration
        )

        # Run test
        await self.manager.initialize(
            gcp_project, cluster_name, self.mocked_sanic_app
        )

        # Verification
        self.assertEqual(self.mocked_event_loop.run_in_executor.call_count, 2)

        mocked_get_cluster_request.assert_called_once_with()
        self.assertEqual(
            mocked_get_cluster_request.name,
            f"projects/{gcp_project}/locations/us-central1/clusters/{cluster_name}",
        )

        mocked_cluster_manager.assert_called_once_with(
            credentials=mocked_credentials,
        )
        mocked_cluster_manager.get_cluster.assert_called_once_with(
            mocked_get_cluster_request
        )

        self.assertEqual(mocked_configuration.get_default_copy.call_count, 2)
        self.assertEqual(mocked_configuration.set_default.call_count, 2)
        self.assertEqual(
            mocked_configuration.api_key,
            {"authorization": f"Bearer {mocked_credentials.token}"},
        )
        self.assertEqual(
            mocked_configuration.host, f"https://{mocked_cluster.endpoint}:443"
        )
        self.assertFalse(mocked_configuration.verify_ssl)

    async def test_stop_idling_workers(self) -> None:
        """Tests stop_idling_workers method behavior."""
        # Test setup
        self.mocked_redis_store.get_workers_ids.return_value = [self.API_KEY]
        self.mocked_redis_store.get_worker.return_value = (
            schemas.WorkerInternal(schemas.WorkerState.IDLE)
        )

        meta = kubernetes.client.V1ObjectMeta(name="test-deployment-1")
        deployment = kubernetes.client.V1Deployment(metadata=meta)
        deployment_list = kubernetes.client.V1DeploymentList(items=[deployment])

        list_namespaced_deployment_thread = unittest.mock.Mock(
            spec=multiprocessing.pool.AsyncResult
        )
        list_namespaced_deployment_thread.get.return_value = deployment_list
        self.mocked_appsv1api.list_namespaced_deployment.return_value = (
            list_namespaced_deployment_thread
        )

        patch_namespaced_deployment_scale_thread = unittest.mock.Mock(
            spec=multiprocessing.pool.AsyncResult
        )
        self.mocked_appsv1api.patch_namespaced_deployment_scale.return_value = (
            patch_namespaced_deployment_scale_thread
        )

        self.mocked_watch.stream.return_value = [
            {"type": x for x in ("ADDED", "DELETED")}
        ]

        self.manager._credentials = unittest.mock.MagicMock()
        self.manager._credentials.expired = False

        # Run test
        await self.manager.stop_idling_workers()

        # Verification
        self.mocked_redis_store.get_workers_ids.assert_called_once()
        self.mocked_redis_store.get_worker.assert_called_once_with(self.API_KEY)

        kwargs = {
            "async_req": True,
            "namespace": "default",
            "label_selector": f"id={self.API_KEY_HASH}",
        }
        self.mocked_appsv1api.list_namespaced_deployment.assert_called_once_with(
            **kwargs
        )

        kwargs = {
            "async_req": True,
            "namespace": "default",
            "body": {"spec": {"replicas": 0}},
        }
        self.mocked_appsv1api.patch_namespaced_deployment_scale.assert_called_once_with(
            meta.name, **kwargs
        )

        self._verify_run_in_executor(
            [
                (
                    (
                        None,
                        list_namespaced_deployment_thread.wait,
                    ),
                ),
                (
                    (
                        None,
                        patch_namespaced_deployment_scale_thread.wait,
                    ),
                ),
                (
                    (
                        None,
                        self.manager._wait_for_asic_cluster_event,
                        self.API_KEY,
                        "DELETED",
                    ),
                ),
            ]
        )

    async def test_stop_idling_workers_no_matches(self) -> None:
        """Tests stop_idling_workers method behavior: no worker is idling"""
        self.mocked_redis_store.get_workers_ids.return_value = [
            self.API_KEY
        ] * 2
        self.mocked_redis_store.get_worker.side_effect = [
            schemas.WorkerInternal(
                schemas.WorkerState.IDLE, job_timestamp=int(time.time())
            ),
            schemas.WorkerInternal(schemas.WorkerState.OFFLINE),
        ]

        # Run test
        await self.manager.stop_idling_workers()

        # Verification
        self.mocked_redis_store.get_workers_ids.assert_called_once()
        call_args_list = [((self.API_KEY,),)] * 2
        self.assertEqual(
            self.mocked_redis_store.get_worker.call_args_list, call_args_list
        )

        self.mocked_appsv1api.list_namespaced_deployment.assert_not_called()

    def test_wait_for_asic_cluster_events_added(self) -> None:
        """Tests _wait_for_asic_cluster_events method behavior: listen for ADDED asic_clusters"""
        # Test setup
        self.mocked_watch.stream.return_value = [{"type": "ADDED"}]

        # Run test
        self.manager._wait_for_asic_cluster_event(self.API_KEY, "ADDED")

        # Verification
        self._verify_watch(timeout_seconds=60)
        self._verify_redis_store("set_booting")

    def test_wait_for_asic_cluster_events_deleted(self) -> None:
        """Tests _wait_for_asic_cluster_events method behavior: listen for DELETED asic_clusters"""
        # Test setup
        self.mocked_watch.stream.return_value = [
            {"type": x for x in ("ADDED", "DELETED")}
        ]

        # Run test
        self.manager._wait_for_asic_cluster_event(self.API_KEY, "DELETED")

        # Verification
        self._verify_watch(timeout_seconds=60)
        self._verify_redis_store("set_offline")

    def test_wait_for_asic_cluster_readiness(self) -> None:
        """Tests _wait_for_asic_cluster_readiness method behavior"""
        # Test setup
        mocked_object = unittest.mock.Mock()
        mocked_object.status = unittest.mock.Mock()
        mocked_object.status.phase = "Running"

        self.mocked_watch.stream.return_value = [
            {"object": mocked_object, "type": "MODIFIED"}
        ]

        # Run test
        self.manager._wait_for_asic_cluster_readiness(self.API_KEY)

        # Verification
        self._verify_watch()

    def _verify_run_in_executor(
        self, call_args_list: typing.List[unittest.mock._Call]
    ) -> None:
        """Verifies calls to the mocked_event_loop.run_in_executor mock.

        Args:
            funcs: List of mocked AsyncResult object that were passed to the
            ASICWorkerManager._wait_for_thread method.
        """
        self.assertEqual(
            self.mocked_event_loop.run_in_executor.call_args_list,
            call_args_list,
        )

    def _verify_redis_store(self, func: str) -> None:
        """Verifies calls to mocked_redis_store mock.

        Args:
            func: Expected called mocked method.
        """
        getattr(self.mocked_redis_store, func).assert_called_once_with(
            self.API_KEY
        )

    def _verify_watch(self, **kwargs) -> None:
        """Verifies calls to the mocked_watch mock."""
        self.mocked_watch.assert_called_once_with()
        self.mocked_watch.stream.assert_called_once_with(
            func=self.mocked_corev1api.list_namespaced_asic_cluster,
            label_selector=f"id={self.API_KEY_HASH}",
            namespace="default",
            **kwargs,
        )
        self.mocked_watch.stop.assert_called_once_with()
