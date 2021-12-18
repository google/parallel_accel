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
import jax

# Using the CPU of the host VM for jitted computations
# alongside ASIC jitted functions
# in the cloud ASIC setup is currently not possible
# in JAX because of a bug (https://github.com/google/jax/issues/5638).
# This module impolements a temporary workaround until the issue
# is resolved on the JAX side.
# TODO : fix this once the JAX bug is fixed.

# we set the default jax config to whatever it is by default
jax_config_settings = {
    "jax_backend_target": jax.config.FLAGS.jax_backend_target,
    "jax_xla_backend": jax.config.FLAGS.jax_xla_backend,
    "jax_platform_name": jax.config.FLAGS.jax_platform_name,
}
MAX_CACHE_SIZE = 256
JAX_PREPRO_BACKEND = None


def set_jax_config_to_cpu():
    """
    Helper function to configure JAX to using CPU.
    This functions stores the current `jax.config`
    configuration in `jax_config_settings` and
    resets jax.config to
    ```
    jax.config.update("jax_backend_target", 'local')
    jax.config.update("jax_xla_backend", 'xla')
    jax.config.update("jax_platform_name", 'cpu')
    ```
    """
    jax_config_settings[
        "jax_backend_target"
    ] = jax.config.FLAGS.jax_backend_target
    jax_config_settings["jax_xla_backend"] = jax.config.FLAGS.jax_xla_backend
    jax_config_settings[
        "jax_platform_name"
    ] = jax.config.FLAGS.jax_platform_name

    jax.lib.xla_bridge.get_backend.cache_clear()
    jax.config.update("jax_backend_target", "local")
    jax.config.update("jax_xla_backend", "xla")
    jax.config.update("jax_platform_name", "cpu")


def reset_jax_to_former_config():
    """
    Helper function to restore the configuration
    prior to the last call to `set_jax_config_to_cpu`.
    """
    jax.lib.xla_bridge.get_backend.cache_clear()
    jax.config.update(
        "jax_backend_target", jax_config_settings["jax_backend_target"]
    )
    jax.config.update("jax_xla_backend", jax_config_settings["jax_xla_backend"])
    jax.config.update(
        "jax_platform_name", jax_config_settings["jax_platform_name"]
    )
