# ParAcc

Cloud based linear_algebra simulator on ASICs

## Installation

For local development and testing install:

* remote_linear_algebra - (python3 -m pip install -e <REPO_DIR>/ParAcc-Client)

* shared utils - (from ParAcc/shared pip install -e .)

Then requirements from

* ParAcc-Server

* ParAcc-Simulator

* ParAcc-Benchmarks

Note: Because of how local install paths are written pip install commands must
be run from each project sub-directory (ParAcc/ParAcc-Server) instead
of from the project root. This is due to how pip handles
[relative paths](https://github.com/pypa/pip/issues/6112)

## Docker

### Authorization

In order to communicate with the Google Cloud Registry you have to provide
credentials helpers using `gcloud` command. Follow the steps at the
REDACTED page.

### Local API-worker development

Since the CI/CD process is taking quite a long time (from merging change to
deploying new image to the GKE cluster), one can use local setup to speed up
testing and developing features.

The [docker-compose.yml](./docker-compose.yml) file provides a configuration
that builds API service and ASIC simulator images from local sources, starts them
and connect to local Redis database.

```shell
# Start docker-compose file
docker-compose up

# Rebuild the select image
# Note: The build argument is optional
docker-compose build \
  --build-arg BUILD_TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  {working_area-api|working_area-sim}

# Stop docker-compose
docker-compose down
```

## Pre-commit hook

The project is using [pre-commit](https://pre-commit.com/) framework for
managing Git pre-commit hooks. The relevant configuration is located in the
[.pre-commit-config.yaml](./.pre-commit-config.yaml) file. Run the shell
commands below to install the tool and repository specific hooks:

```shell
# Install the tool in the system
sudo apt install -y pre-commit

# Now install the hooks
pre-commit install
```
