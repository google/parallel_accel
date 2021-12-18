# ASIC worker overlays

This directory contains scripts for creating overlays for ASIC workers.

## Setup

Before running the script call [bootstrap.sh](./scripts/bootstrap.sh) shell
script that setup virtual environment and install necessary dependencies.

## Create API key

Before creating ASIC overlay file, you need to have the corresponding API key.
You can use the [api_key.py](../../../scripts/api_key.py) script to generate one
using following shell command:

```bash
cd ../../../scripts
api_key.py --service={development|production} --label=LABEL create
```

where:

- **SERVICE** is the service environment.
- **LABEL** is the key custom display name.

## Usage

The script accepts two input parameters:

- **api_key** is the API key generated in the Cloud Console.
- **label** is the deployment suffix, same as the API key label.
- **image-version** is the gcr image tag to source from. The default version is
	latest.

Assume we would like to create overlay for the **example-key-owner** owner
using **example-api-key** API key. We will use the following shell command:

```bash
cd scripts/worker_overlay

# Generate single overlay file
worker_overlay.py --api-key=example-api-key --label=example-key-owner

# Generate multiple overlay files
worker_overlay.py \
    --api-key=example-api-key-1 \
    --label=example-key-label-1 \
    --api-key=example-api-key-2 \
    --label=example-key-label-2
```

The generated overlay will be emitted to the [workers](../../overlays/workers)
directory along with `kustomization.yaml` file.
