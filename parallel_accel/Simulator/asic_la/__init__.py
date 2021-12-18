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
import os
import time

from jax.config import config
from parallel_accel.shared import logger

# NOTE: This is necessary in the current GKE Setup
#       * Sleeping circumvents a GKE crash loop
#       * configuration sets up ASIC Backend on GCP using a previous driver
#         as the nightly driver has caused issues
#       * A better solution should be created in the future

ASIC_VERSION = "asic_driver0.1-dev20200911"
# Alternate drivers
# asic_driver0.1-dev20200320
# asic_driver2.3.0.dev20200620

log = logger.get_logger(__name__)
if "ASIC_NAME" in os.environ:
    if "PARALLELACCEL_FAST_BOOT" not in os.environ:
        log.debug("Waiting for asic to boot...")
        time.sleep(120)  # wait for asic to boot up
    else:
        log.warning("Fast Boot flag found, not waiting for asic.")

    if not "SET_ASIC_DRIVER" in os.environ:
        from cloud_asic_client import Client

        log.debug(
            "Configuring asic client",
            asic_name=os.environ.get("ASIC_NAME"),
            asic_version=ASIC_VERSION,
        )
        c = Client(os.environ.get("ASIC_NAME").split("/")[1])
        c.configure_asic_version(ASIC_VERSION)
        os.environ["SET_ASIC_DRIVER"] = "1"

    asic_ip = os.environ.get("KUBE_GOOGLE_CLOUD_ASIC_ENDPOINTS")

    if asic_ip is not None:
        log.debug("Configuring jax_backend", asic_ip=asic_ip)
        config.update("jax_xla_backend", "asic_driver")
        config.update("jax_backend_target", asic_ip)

if "PARALLELACCEL_JAX_VERBOSE" in os.environ:
    log.debug("Setting jax_log_compiles to verbose")
    config.update("jax_log_compiles", 1)
    # Log jit cache activity
