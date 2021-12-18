# ParallelAccel gcp_cluster_vm Dev Scripts

The following scripts are provided for quick development and benchmarking
on a gcp_cluster_vm setup. These scripts provide the following functionality:
* Quickly spin up a VM and paired ASIC
* Setup ENV vars for parallel_accel to find the paired ASIC
* Continuously sync the ParAcc package to the VM so text editing can be done
locally
* Spin down the VM / ASIC pair

## Usage

`./create_dev.sh username`

This created a VM / ASIC pair with the names:

* parallel_accel-dev-vm-username
* parallel_accel-dev-asic-username

And sets the `KUBE_GOOGLE_CLOUD_ASIC_ENDPOINTS` env var on the VM.

`./run_sync_dev.sh username`

This continuously watches the local ParAcc repo and syncs changes to the VM.
This script assumes ssh keys for `username` are included in VM metadata and 
the local `~/.ssh/google_compute_engine` file.

`./ssh_dev.sh username`

Establishes an SSH connection to the VM. Makes the same assumptions about SSH
keys.

`./delete_dev.sh`

Deletes the VM / ASIC pair.

