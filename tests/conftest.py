"""PyTest configuration module.

Defines useful fixtures, command line args.
"""

import gzip
import logging
import shutil
from pathlib import Path

import pandas as pd
import pytest

from etoolbox.utils.cloud import etb_cloud_init
from etoolbox.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@pytest.fixture
def df_dict() -> dict:
    """Dictionary of dfs."""
    return {
        "a": pd.DataFrame(
            [[0, 1], [2, 3]],
            columns=pd.MultiIndex.from_tuples([(0, "a"), (1, "b")]),
        ),
        "b": pd.Series([1, 2, 3, 4]),
    }


# @pytest.fixture
# def klass_w_slot(df_dict):
#     """Generic class that uses slots."""
#     obj = _KlassSlots()
#     obj.foo = df_dict["a"]
#     obj.tup = (1, 2)
#     obj.lis = (3, 4)
#     obj._dfs = df_dict
#     return obj
#
#
# @pytest.fixture
# def klass_wo_slot(df_dict):
#     """Generic class that does not use slots."""
#     obj = _TestKlass()
#     obj.foo = df_dict["a"]
#     obj._dfs = df_dict
#     return obj


@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Return the path to the top-level directory containing the tests.

    This might be useful if there's test data stored under the tests
    directory that
    you need to be able to access from elsewhere within the tests.

    Mostly this is meant as an example of a fixture.
    """
    return Path(__file__).parent


@pytest.fixture(scope="session")
def temp_dir(test_dir) -> Path:
    """Return the path to a temp directory that gets deleted on teardown."""
    out = test_dir / "temp"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(exist_ok=True)
    yield out
    shutil.rmtree(out)


@pytest.fixture(scope="session")
def pudl_test_cache(temp_dir):
    """Change PUDL cache path for testing."""
    import etoolbox.utils.pudl as pudl

    pudl.CACHE_PATH = temp_dir / "pudl_cache"


@pytest.fixture
def pudl_test_cache_for_ep(temp_dir):
    """Setup dummy pudl cache and config directories for testing."""
    import etoolbox.utils.pudl as pudl

    pudl.CACHE_PATH = temp_dir / "rmi.pudl.cache/aws"
    pudl.TOKEN_PATH = temp_dir / "rmi.pudl/.pudl-access-key.json"

    pudl.CACHE_PATH.mkdir(exist_ok=True, parents=True)
    pudl.TOKEN_PATH.parent.mkdir(exist_ok=True)
    pudl.TOKEN_PATH.touch(exist_ok=True)
    pudl.pl_read_pudl("core_eia__codes_balancing_authorities")
    (pudl.CACHE_PATH.parent / "cache").touch()


@pytest.fixture(scope="class")
def cloud_test_cache_persistent(temp_dir):
    """Setup dummy cloud cache and config directories for testing."""
    import etoolbox.utils.cloud as cloud

    original_paths = (
        cloud.CONFIG_PATH,
        cloud.ETB_AZURE_CONFIG_PATH,
        cloud.AZURE_BASE_CACHE_PATH,
    )

    cloud.CONFIG_PATH = temp_dir / "etb-cloud_persistent"
    cloud.AZURE_BASE_CACHE_PATH = temp_dir / "rmi.cloud.cache_persistent"
    cloud.ETB_AZURE_CONFIG_PATH = cloud.CONFIG_PATH / "etb_azure_config.json"

    cloud.AZURE_BASE_CACHE_PATH.mkdir(exist_ok=True, parents=True)
    cloud.CONFIG_PATH.mkdir(exist_ok=True, parents=True)
    cloud.read_config.cache_clear()
    yield

    shutil.rmtree(cloud.AZURE_BASE_CACHE_PATH, ignore_errors=True)
    shutil.rmtree(cloud.CONFIG_PATH, ignore_errors=True)
    cloud.CONFIG_PATH, cloud.ETB_AZURE_CONFIG_PATH, cloud.AZURE_BASE_CACHE_PATH = (
        original_paths
    )
    cloud.read_config.cache_clear()


@pytest.fixture
def cloud_test_cache(temp_dir):
    """Setup dummy cloud cache and config directories for testing."""
    import etoolbox.utils.cloud as cloud

    original_config = cloud.read_config()

    original_paths = (
        cloud.CONFIG_PATH,
        cloud.ETB_AZURE_CONFIG_PATH,
        cloud.AZURE_BASE_CACHE_PATH,
    )

    cloud.CONFIG_PATH = temp_dir / "etb-cloud"
    cloud.AZURE_BASE_CACHE_PATH = temp_dir / "rmi.cloud.cache"
    cloud.ETB_AZURE_CONFIG_PATH = cloud.CONFIG_PATH / "etb_azure_config.json"

    cloud.AZURE_BASE_CACHE_PATH.mkdir(exist_ok=True, parents=True)
    cloud.CONFIG_PATH.mkdir(exist_ok=True, parents=True)
    cloud.read_config.cache_clear()
    yield

    shutil.rmtree(cloud.AZURE_BASE_CACHE_PATH, ignore_errors=True)
    shutil.rmtree(cloud.CONFIG_PATH, ignore_errors=True)
    cloud.CONFIG_PATH, cloud.ETB_AZURE_CONFIG_PATH, cloud.AZURE_BASE_CACHE_PATH = (
        original_paths
    )
    cloud.read_config.cache_clear()


@pytest.fixture
def cloud_test_cache_w_files(temp_dir):
    """Setup dummy cloud cache and config files for testing."""
    import etoolbox.utils.cloud as cloud

    original_config = cloud.read_config()
    original_paths = (
        cloud.CONFIG_PATH,
        cloud.ETB_AZURE_CONFIG_PATH,
        cloud.AZURE_BASE_CACHE_PATH,
    )

    cloud.CONFIG_PATH = temp_dir / "etb-cloud"
    cloud.AZURE_BASE_CACHE_PATH = temp_dir / "rmi.cloud.cache"
    cloud.ETB_AZURE_CONFIG_PATH = cloud.CONFIG_PATH / "etb_azure_config.json"

    cloud.AZURE_BASE_CACHE_PATH.mkdir(exist_ok=True, parents=True)
    cloud.CONFIG_PATH.mkdir(exist_ok=True, parents=True)
    cloud.read_config.cache_clear()
    etb_cloud_init(original_config["account_name"], "use_azure_cli", False)

    fs = cloud.rmi_cloud_fs()
    _ = pd.read_parquet("az://patio-data/test_data.parquet", filesystem=fs)
    yield original_config["account_name"]

    shutil.rmtree(cloud.AZURE_BASE_CACHE_PATH, ignore_errors=True)
    shutil.rmtree(cloud.CONFIG_PATH, ignore_errors=True)
    cloud.CONFIG_PATH, cloud.ETB_AZURE_CONFIG_PATH, cloud.AZURE_BASE_CACHE_PATH = (
        original_paths
    )
    cloud.read_config.cache_clear()


@pytest.fixture(scope="session")
def gzip_test_data(temp_dir):
    """Download pudl sqlite for testing if we don't have a local one."""
    gzip_path = temp_dir / "test_file.txt.gz"
    content = b"What hath G-d wrought"
    with gzip.open(gzip_path, "wb") as f:
        f.write(content)
    return gzip_path, content


@pytest.fixture(scope="session")
def test_logger(temp_dir):
    """Set up logger for testing."""
    log_file = temp_dir / "logs/log.jsonl"
    setup_logging(filename=log_file)

    logger = logging.getLogger("etb_test")
    return logger, log_file
