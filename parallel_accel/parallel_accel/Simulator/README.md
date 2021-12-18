# ParAcc-Simulator
Backend ASIC simulator for simulating symplectic acyclic_graphs

# asic_sim_runner.py
Entry point for the Simulator on the ParallelAccel service. This class instantiates the
simulator and manages connections to other parallel_accel service components.

# asic_la/asic_simulator.py
ASIC simulation code

## Environment Variables
Disable startup wait time:
`PARALLELACCEL_FAST_BOOT=1`

Enable experimental all to all behaviour:
`PARALLELACCEL_ENABLE_EXPERIMENTAL_ALL_TO_ALL=1`

Turn on JAX Compilation Logging:
`PARALLELACCEL_JAX_VERBOSE=1`

Specify ASIC Accelerator IP to configure gcp_cluster_vm XLA backend:
`KUBE_GOOGLE_CLOUD_ASIC_ENDPOINTS=...`

Set ASIC Name to configure ASIC backend driver version:
`ASIC_NAME=...`
