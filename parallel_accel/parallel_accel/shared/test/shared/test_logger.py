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
# pylint: disable=invalid-name

"""Unit test for shared.logger module"""
import json
import unittest
import unittest.mock
import structlog

from parallel_accel.shared import logger, utils


class TestLogger(unittest.TestCase):
    """Tests Logger class behavior."""

    def tearDown(self) -> None:
        """See base class documentation."""
        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

    def test_local_logger(self) -> None:
        """Tests logger behviaor for local environment"""
        # Set up
        api_key = "some_key"
        message = "Hello World"

        logger.setup(True)

        cf = structlog.testing.CapturingLoggerFactory()
        structlog.configure(logger_factory=cf)

        # Run test
        log = logger.get_logger(self.__class__.__name__)
        log.info(message, api_key=api_key)

        # Verification
        self.assertEqual(cf.logger.calls[0].method_name, "info")
        self.assertEqual(
            cf.logger.calls[0].args,
            (
                json.dumps(
                    {
                        "component": self.__class__.__name__,
                        "api_key": utils.sha1(api_key),
                        "event": message,
                    }
                ),
            ),
        )
