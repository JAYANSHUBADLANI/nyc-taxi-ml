"""Shared fixtures.

These tests never touch the real gold parquet, they run against small hand built
frames whose expected answers are worked out by hand. A single local Spark session
is shared across the whole run since starting a JVM per test would dominate the
runtime.
"""

from __future__ import annotations

import os
import time

# Pinned before the JVM starts. PySpark converts naive Python datetimes using the
# driver's system timezone while the session evaluates them in
# spark.sql.session.timeZone. On a machine set to anything other than UTC the two
# disagree and every timestamp in a fixture silently shifts, so both ends are
# pinned to UTC here and the transform is what gets measured, not the machine.
os.environ["TZ"] = "UTC"
time.tzset()

import pytest  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[2]")
        .appName("nyc-taxi-ml-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
