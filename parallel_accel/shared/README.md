# ParAcc Utilities

Shared code for ParAcc backends.

## Logger

The [logger.py](./parallel_accel/shared/logger.py) module is a wrapper for the
[structlog](https://www.structlog.org/en/stable/) package. When making initial
call to `parallel_accel.shared.logger.get_logger` function, the structlog is configured
based on the environment:

- when running in local development setup, the log messages are pretty-printed
- when running in a GKE cluster, the logs messages are JSON formatted

```python
# Import logging module
from parallel_accel.shared import logger

# Now get the logger and use it the standard way
instance = logger.get_logger(__name__)
instance.debug("Hello!", who="World")

# When logging API key, pass it as the api_key parameter
instance.log(api_key="my-key")

# When logging exception, pass it as the exc_info parameter
try:
  raise Exception
except Exception as exc:
  instance.error(exc_info=exc)
```

The structlog logger provides an option to bind contextual parameters to the
log messages via `structlog.threadlocal` module. Once a parameter is bound,
it will be emitted to all subsequential log message until manually removed.
This feature is provided by `LoggerProvider.context` property:

```python
# Import logging module
from parallel_accel.shared import logger

# Now get the logger and use it the standard way
instance = logger.get_logger(__name__)

# Bind some parameters
logger.context.bind(foo="bar", foobar="baz")

# Log message as usual, it will include "foo=bar foobar=baz" parameters
instance.debug("Hello!")

# Unbind specific parameter
logger.context.unbind("foo")

# The log message will only contain "foobar=baz" parameter
instance.debug("Hello!")

# Clear all parameters
logger.context.clear()

# Now the log message won't contain any extra parameters
instance.debug("Hello!")
```

## Redis

The [redis](./parallel_accel/shared/redis) package contains RedisStore wrappers for
sharing data between ASIC workers and the API service:

- [BaseRedisStore](./parallel_accel/shared/redis/base.py:45) is based class for spawning
  connection to the Redis host.
- [JobsRedisStore](./parallel_accel/shared/redis/jobs.py:33) is based class for sending
  jobs related data between services. It is meant to be subclassed by specific
  service.
- [WorkerRedisStore](./parallel_accel/shared/redis/workers.py:39) handles fetching and
  updating current ASIC worker status.

## Schemas

The [schemas.py](./parallel_accel/shared/schemas.py) module contains reusable
[marshmallow](https://marshmallow.readthedocs.io/en/stable/index.html) schemas
for the data exchange between API service, ASIC worker and the end user client.

```python
import uuid
import linear_algebra

from parallel_accel.shared import schemas

# Create a SampleJob object
context = schemas.SampleJobContext(
    acyclic_graph=linear_algebra.Graph(linear_algebra.flip_z_axis(linear_algebra.GridSpace(1, 1))),
    param_resolver=linear_algebra.ParamResolver(None),
    repetitions=1,
)
job = schemas.SampleJob(
    context=context,
    id=uuid.uuid4(),
    type=schemas.JobType.SAMPLE,
)

# Serialize job object to JSON encoded string
encoded_job = schemas.encode(schemas.SampleJobSchema, job)

# Deserialize JSON encoded string to job
decoded_job = schemas.decode(schemas.SampleJobSchema, serialized_job)

# Decode only specific fields
partial_job = schemas.decode(
    schemas.SampleJobSchema, serialized_job, partial=('id', 'type')
)
```
