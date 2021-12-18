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
"""Server application entry point"""
import asyncio
import os
import sanic

from src import containers  # pylint: disable=import-error


async def initialize(
    app: sanic.Sanic,  # pylint: disable=redefined-outer-name
    _: asyncio.AbstractEventLoop,
) -> None:
    """Initializes application components.

    This is the one time setup function that should be called right after Sanic
    main loop has started. If this function throw any exception, the entire
    application will fail to start.

    More about Sanic lifecycle hooks:
        https://sanicframework.org/en/guide/basics/listeners.html#listeners

    Args:
        app: Instance of Sanic application.
        loop: Currently running event loop.
    """
    store = container.tasks_store()

    tasks_manager = container.tasks_manager()
    tasks_manager.initialize(app, store)

    if int(os.environ.get("DISABLE_GKE_ACCESS", 0)) == 1:
        sanic.log.logger.warning("Disabling GKE Access...")
        return

    asic_worker_manager = container.worker_manager()
    await asic_worker_manager.initialize(
        os.environ["GCP_PROJECT"], os.environ["GKE_CLUSTER"], app
    )


container = containers.ApplicationContainer()
app = container.sanic_app()
app.ctx.container = container
app.register_listener(initialize, "main_process_start")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True, access_log=True)
