# ParallelAccel Service end-to-end tests

## Setup

First, we need to setup and configure Python virtual environment. Open shell and
run following commands:

```bash
python3 -m venv env
source env/bin/activate
# Install ParallelAccel client package
pip install -e ../ParAcc-Client
# Install pytest
pip install pytest
```

### Local

To bring up local development environment we can use
[docker-compose.yml](../docker-compose.yml) file that spins up API
server, simulator and Redis instance. Use the following shell command to start
up local setup:

```bash
docker-compose -f ../docker-compose.yml up
```

### Remote

By default E2E tests are configured to run against local setup. In order to test
against ParallelAccel service, we need to modify client configuration in the
[test_config.py](test_config.py).

## Running tests

To run end-to-end tests, we will use pytest to execute test cases. Open shell
terminal and run following command:

```shell
# Run all tests
pytest
# Run specific test module
pytest <module_name>_test.py
```

### Test against local setup

Some tests are designed to be executed only against the remote service and will
fail when running locally. Those are resource expensive or GCP infrastructure
dependent tests. We can skip them by passing `USING_LOCAL_SETUP=1` to the pytest
command:

```shell
USING_LOCAL_SETUP=1 pytest
```
