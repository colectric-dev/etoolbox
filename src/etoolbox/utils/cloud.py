"""Tools for working with RMI's Azure storage."""

import base64
import logging
import os
import shutil
import subprocess
import warnings
from contextlib import nullcontext, suppress
from datetime import datetime
from functools import cache
from pathlib import Path

import click
import orjson as json
import pandas as pd
import polars as pl
import tomllib
import yaml
from fsspec import filesystem
from fsspec.implementations.cached import WholeFileCacheFileSystem
from platformdirs import user_cache_path, user_config_path

from etoolbox.datazip import DataZip
from etoolbox.utils.misc import all_logging_disabled

try:
    import tqdm  # noqa: F401
    from fsspec.callbacks import TqdmCallback as ProgressCallback

except (ImportError, ModuleNotFoundError):
    from fsspec.callbacks import DotPrinterCallback as ProgressCallback


CONFIG_PATH = user_config_path("rmi.cloud", ensure_exists=True)
AZURE_BASE_CACHE_PATH = user_cache_path("rmi.cloud", ensure_exists=True)
ETB_AZURE_CONFIG_PATH = CONFIG_PATH / "etb_azure_config.json"
ETB_AZURE_TOKEN_PATH = CONFIG_PATH / "etb_azure_token.txt"
ETB_AZURE_ACCOUNT_NAME_PATH = CONFIG_PATH / "etb_azure_account_name.txt"

logger = logging.getLogger("etoolbox")


@property
def AZURE_CACHE_PATH():  # noqa: N802
    """For compatibility."""
    warnings.warn("use read_config()['cache_path']", DeprecationWarning, stacklevel=2)
    return read_config()["cache_path"]


def convert_cloud_config(*, keep_old):
    """Convert config and cache to new setup / schema."""
    account_name = ETB_AZURE_ACCOUNT_NAME_PATH.read_text()
    cache_path = AZURE_BASE_CACHE_PATH / account_name
    if not cache_path.exists():
        cache_path.mkdir()
    old_cache = user_cache_path("rmi.cloud")
    if AZURE_BASE_CACHE_PATH.exists() and not keep_old:
        for file in old_cache.iterdir():
            if file.is_dir():
                continue
            shutil.move(file, cache_path)
    if (old_path := CONFIG_PATH / "rmicfezil_token.txt").exists():
        with open(old_path) as f:
            token = base64.b64encode(f.read().encode("utf-8"))
    else:
        token = ETB_AZURE_TOKEN_PATH.read_text()
    if isinstance(token, bytes):
        token = token.decode("utf8")
    out = {
        "active": account_name,
        account_name: {
            "account_name": account_name,
            "auth": {"sas_token": token},
            "cache_path": str(cache_path),
        },
    }
    with open(ETB_AZURE_CONFIG_PATH, "wb") as f:
        f.write(json.dumps(out, option=json.OPT_INDENT_2))
    if keep_old:
        return None
    for p in (old_path, ETB_AZURE_TOKEN_PATH, ETB_AZURE_ACCOUNT_NAME_PATH):
        with suppress(Exception):
            p.unlink()


@cache
def read_config() -> dict[str, str | dict[str, str | bool]]:
    """Read configs and return active config."""
    if all(
        os.environ.get(v) for v in ("ETB_AZURE_SAS_TOKEN", "ETB_AZURE_ACCOUNT_NAME")
    ):
        cache_path = AZURE_BASE_CACHE_PATH / os.environ.get("ETB_AZURE_ACCOUNT_NAME")
        if not cache_path.exists():
            cache_path.mkdir()
        config = {
            "account_name": os.environ.get("ETB_AZURE_ACCOUNT_NAME"),
            "cache_path": cache_path,
        }
        token = os.environ.get("ETB_AZURE_SAS_TOKEN")
        if all(part in token.casefold() for part in ("azure", "cli")):
            config["auth"] = {"anon": False}
            config["auth_pl"] = {"use_azure_cli": "True"}
        else:
            config["auth_pl"] = config["auth"] = {"sas_token": token}
        return config
    if not ETB_AZURE_CONFIG_PATH.exists():
        try:
            convert_cloud_config(keep_old=True)
        except Exception as exc:
            raise RuntimeError(
                "new config does not exist and unable to convert existing config, "
                "run `etb cloud init`"
            ) from exc
    with open(ETB_AZURE_CONFIG_PATH) as f:
        configs = json.loads(f.read())
    config = configs[os.environ.get("ETB_AZURE_ACTIVE_ACCOUNT", configs["active"])]
    if (tok := config["auth"].get("sas_token", None)) is not None:
        config["auth_pl"] = config["auth"] = {
            "sas_token": base64.b64decode(tok).decode("utf-8")
        }
    else:
        config["auth_pl"] = {"use_azure_cli": "True"}
    return config


def cloud_clean(*, dry: bool = False, all_: bool = False):
    """Cleanup cache and config directories."""
    config = read_config()
    info = cache_info()
    size = info["size"].sum() * 1e-6
    click.echo(
        f"Will delete the following items using {size:,.0f} MB at "
        f"{config['cache_path']}"
    )
    click.echo(info[["size", "time"]])
    if not dry:
        shutil.rmtree(config["cache_path"], ignore_errors=True)
    if all_:
        click.echo(f"deleting config {CONFIG_PATH}")
        if not dry:
            shutil.rmtree(CONFIG_PATH, ignore_errors=True)


# @lru_cache
# def read_token() -> str:
#     """Read SAS token from disk or environment variable."""
#
#     with open(ETB_AZURE_CONFIG_PATH) as f:
#         config = json.loads(f.read())
#
#     if ETB_AZURE_TOKEN_PATH.exists():
#         return base64.b64decode(ETB_AZURE_TOKEN_PATH.read_text()).decode("utf-8")
#     if (token := os.environ.get("ETB_AZURE_SAS_TOKEN")) is not None:
#         return token
#     if (old_path := CONFIG_PATH / "rmicfezil_token.txt").exists():
#         with open(old_path) as f:
#             token = f.read()
#         with open(ETB_AZURE_TOKEN_PATH, "wb") as f:
#             f.write(base64.b64encode(token.encode("utf-8")))
#         old_path.unlink()
#         return read_token()
#     raise ValueError(
#         "No SAS Token found, either run `etb cloud init` or set "
#         "ETB_AZURE_SAS_TOKEN environment variable."
#     )


# @lru_cache
# def read_account_name() -> str:
#     """Read SAS token from disk or environment variable."""
#     if ETB_AZURE_ACCOUNT_NAME_PATH.exists():
#         return ETB_AZURE_ACCOUNT_NAME_PATH.read_text()
#     elif (token := os.environ.get("ETB_AZURE_ACCOUNT_NAME")) is not None:
#         return token
#     raise ValueError(
#         "No Azure account name found, either re-run `etb cloud init` "
#         "or set ETB_AZURE_ACCOUNT_NAME environment variable."
#     )


def storage_options():
    """Simplify reading from Azure using :mod:`polars`.

    When using :mod:`pandas` or writing to Azure, see :func:`.rmi_cloud_fs`.

    Examples
    --------
    >>> import polars as pl
    >>> from etoolbox.utils.cloud import storage_options

    >>> df = pl.read_parquet("az://patio-data/test_data.parquet", **storage_options())
    >>> df.head()  # doctest: +NORMALIZE_WHITESPACE
    shape: (5, 2)
    ┌────────────────────┬──────────────────┐
    │ energy_source_code ┆ co2_mt_per_mmbtu │
    │ ---                ┆ ---              │
    │ str                ┆ f64              │
    ╞════════════════════╪══════════════════╡
    │ AB                 ┆ 1.1817e-7        │
    │ ANT                ┆ 1.0369e-7        │
    │ BFG                ┆ 2.7432e-7        │
    │ BIT                ┆ 9.3280e-8        │
    │ BLQ                ┆ 9.4480e-8        │
    └────────────────────┴──────────────────┘

    """
    config = read_config()
    return {
        "storage_options": {"account_name": config["account_name"]} | config["auth_pl"]
    }


def rmi_cloud_fs(account_name=None, token=None) -> WholeFileCacheFileSystem:
    """Work with files on Azure.

    This can be used to read or write arbitrary files to or from Azure. And for files
    read from Azure, it will create and manage a local cache.

    Examples
    --------
    >>> import pandas as pd
    >>> from etoolbox.utils.cloud import rmi_cloud_fs

    >>> fs = rmi_cloud_fs()
    >>> df = pd.read_parquet("az://patio-data/test_data.parquet", filesystem=fs)
    >>> df.head()  # doctest: +NORMALIZE_WHITESPACE
      energy_source_code  co2_mt_per_mmbtu
    0                 AB      1.181700e-07
    1                ANT      1.036900e-07
    2                BFG      2.743200e-07
    3                BIT      9.328000e-08
    4                BLQ      9.448000e-08

    Read with :mod:`polars` using the same filecache as with :mod:`pandas`.

    >>> import polars as pl

    >>> with fs.open("az://patio-data/test_data.parquet") as f:
    ...     df = pl.read_parquet(f)
    >>> df.head()  # doctest: +NORMALIZE_WHITESPACE
    shape: (5, 2)
    ┌────────────────────┬──────────────────┐
    │ energy_source_code ┆ co2_mt_per_mmbtu │
    │ ---                ┆ ---              │
    │ str                ┆ f64              │
    ╞════════════════════╪══════════════════╡
    │ AB                 ┆ 1.1817e-7        │
    │ ANT                ┆ 1.0369e-7        │
    │ BFG                ┆ 2.7432e-7        │
    │ BIT                ┆ 9.3280e-8        │
    │ BLQ                ┆ 9.4480e-8        │
    └────────────────────┴──────────────────┘

    Write a parquet file, or really anything to Azure...

    >>> with fs.open("az://patio-data/file.parquet", mode="wb") as f:  # doctest: +SKIP
    ...     df.write_parquet(f)

    """
    config = read_config()
    auth = config["auth"] if token is None else {"sas_token": token}
    return filesystem(
        "filecache",
        target_protocol="az",
        target_options={
            "account_name": config["account_name"]
            if account_name is None
            else account_name,
        }
        | auth,
        cache_storage=config["cache_path"],
        check_files=True,
        cache_timeout=None,
    )


def cache_info():
    """Return info about cloud cache contents."""
    config = read_config()
    cache_path = Path(config["cache_path"])
    with open(cache_path / "cache", "rb") as f:
        cache_data = json.loads(f.read())
    cdl = [
        v
        | {
            "size": (cache_path / v["fn"]).stat().st_size,
            "time": datetime.fromtimestamp(v["time"]),
        }
        for v in cache_data.values()
        if (cache_path / v["fn"]).exists()
    ]
    return pd.DataFrame.from_records(cdl).set_index("original")[
        ["time", "size", "fn", "uid"]
    ]


def cached_path(cloud_path: str, *, download=False) -> str | None:
    """Get the local cache path of a cloud file.

    Args:
        cloud_path: path on azure, e.g. ``az://raw-data/test_data.parquet``
        download: download the file from Azure to create a local cache if it
            does not exist.

    Examples
    --------
    >>> import polars as pl
    >>> from etoolbox.utils.cloud import rmi_cloud_fs, cached_path

    >>> fs = rmi_cloud_fs()
    >>> cloud_path = "az://patio-data/test_data.parquet"
    >>> with fs.open(cloud_path) as f:
    ...     df = pl.read_parquet(f)
    >>> cached_path(cloud_path)
    '656706c40cb490423b652aa6d3b4903c56ab6c798ac4eb2fa3ccbab39ceebc4a'

    """
    cloud_path = cloud_path.removeprefix("az://").removeprefix("abfs://")
    if download:
        f = rmi_cloud_fs().open(cloud_path)
        f.close()
    try:
        return cache_info().loc[cloud_path, "fn"]
    except KeyError:
        return None


def cloud_list(path: str, *, detail=False) -> list[str] | dict:
    """List cloud files in a folder.

    Args:
        path: remote folder to list contents of e.g. '<container>/...'
        detail: include detail information

    """
    fs = rmi_cloud_fs()
    return fs.ls(path, detail=detail)


AZ_MSG = (
    "azcopy not installed at ``/opt/homebrew/bin/azcopy``\n"
    "for better performance``brew install azcopy``, or set ``azcopy_path`` argument "
    "if installed elsewhere"
    "more info\nhttps://github.com/Azure/azure-storage-azcopy"
)


def get(
    to_get_path: str,
    destination: Path | str,
    fs=None,
    *,
    quiet=True,
    clobber=False,
    azcopy_path="/opt/homebrew/bin/azcopy",
) -> None:
    """Download a remote file from the cloud.

    Uses ``azcopy`` CLI if available.

    Args:
        to_get_path: remote file or folder to download of the form ``<container>/...``
        destination: local destination for the downloaded files
        fs: filesystem
        quiet: disable logging of adlfs output
        clobber: overwrite existing files and directories if True
        azcopy_path: path to azcopy executable
    """
    config = read_config()
    account_name = config["account_name"]
    to_get_path = (
        to_get_path.removeprefix("az://")
        .removeprefix("abfs://")
        .removeprefix(f"https://{account_name}.blob.core.windows.net/")
    )
    token = (
        "" if (tok := config["auth_pl"].get("sas_token", None)) is None else f"?{tok}"
    )
    try:
        subprocess.run([azcopy_path], capture_output=True)  # noqa: S603
    except Exception:
        print(AZ_MSG)
        fs = rmi_cloud_fs() if fs is None else fs
        context = all_logging_disabled if quiet else nullcontext
        with context():
            ls = fs.ls(to_get_path)
            if ls[0]["name"] != to_get_path:
                raise TypeError(
                    "`to_get_path` must be a file when not using azcopy."
                ) from None
            fs.get(
                rpath="az://" + to_get_path,
                lpath=str(destination),
                recursive=False,
                callback=ProgressCallback(),
            )
    else:
        subprocess.run(  # noqa: S603
            [
                azcopy_path,
                "cp",
                f"https://{account_name}.blob.core.windows.net/{to_get_path}{token}",
                f"{destination}",
                f"--overwrite={str(clobber).casefold()}",
                "--recursive=True",
            ],
        )


def put(
    to_put_path: Path,
    destination: str,
    fs=None,
    *,
    quiet=True,
    clobber=False,
    azcopy_path="/opt/homebrew/bin/azcopy",
) -> None:
    """Upload local files or directories to the cloud.

    Copies a specific file or tree of files. If destination
    ends with a "/", it will be assumed to be a directory, and target files
    will go within.

    Uses ``azcopy`` CLI if available.

    Args:
        to_put_path: local file or folder to copy
        destination: copy destination of the form ``<container>/...``
        fs: filesystem
        quiet: disable logging of adlfs output
        clobber: force overwriting of existing files (only works when azcopy is used)
        azcopy_path: path to azcopy executable
    """
    if not to_put_path.exists():
        raise FileNotFoundError(to_put_path)
    lpath = str(to_put_path)
    config = read_config()
    account_name = config["account_name"]
    destination = (
        destination.removeprefix("az://")
        .removeprefix("abfs://")
        .removeprefix(f"https://{account_name}.blob.core.windows.net/")
    )
    token = (
        "" if (tok := config["auth_pl"].get("sas_token", None)) is None else f"?{tok}"
    )
    recursive = to_put_path.is_dir()
    try:
        subprocess.run([azcopy_path], capture_output=True)  # noqa: S603
    except Exception:
        print(AZ_MSG)
        context = all_logging_disabled if quiet else nullcontext
        fs = rmi_cloud_fs() if fs is None else fs
        with context():
            fs.put(
                lpath=lpath,
                rpath="az://" + destination,
                recursive=recursive,
                callback=ProgressCallback(),
            )
    else:
        subprocess.run(  # noqa: S603
            [
                azcopy_path,
                "cp",
                lpath,
                f"https://{account_name}.blob.core.windows.net/{destination}{token}",
                f"--overwrite={str(clobber).casefold()}",
                f"--recursive={str(recursive).casefold()}",
            ],
        )


def read_patio_resource_results(datestr: str) -> dict[str, pd.DataFrame]:
    """Reads patio resource results from Azure.

    Reads patio resource results from Azure and returns the extracted data as a
    dictionary (named list). The method handles the specific format of patio resource
    files and manages file system interactions as well as cache mechanisms.

    Args:
        datestr: Date string that identifies the model run.

    """
    out = read_cloud_file(f"patio-results/{datestr}/BAs_{datestr}_results.zip")
    for k, v in out.items():
        if isinstance(v, pd.DataFrame):
            out[k] = v.convert_dtypes(
                convert_boolean=True,
                convert_string=False,
                convert_floating=False,
                convert_integer=False,
            )
        elif isinstance(v, pl.DataFrame):
            out[k] = v.to_pandas()
    return out


def read_cloud_file(filename: str) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """Read parquet, csv, or DataZip files from Azure.

    The method handles the specific format of patio resource
    files and manages file system interactions as well as cache mechanisms.

    Args:
        filename: the full path to the file including container and file extension.

    Examples
    --------
    >>> from etoolbox.utils.cloud import read_cloud_file

    >>> df = read_cloud_file("patio-data/20241031/utility_ids.parquet")
    >>> df.head()  # doctest: +NORMALIZE_WHITESPACE
       utility_id_ferc1  ...  public_private_unmapped
    0               1.0  ...                 unmapped
    1             342.0  ...                   public
    2             294.0  ...                   public
    3             394.0  ...                   public
    4             349.0  ...                   public
    <BLANKLINE>
    [5 rows x 37 columns]

    """
    fs = rmi_cloud_fs()
    filename = "az://" + filename.removeprefix("az://").removeprefix("abfs://")
    config = read_config()
    if ".parquet" in filename:
        try:
            return pd.read_parquet(filename, filesystem=fs)
        except Exception:
            with fs.open(filename) as fp:
                return pl.read_parquet(fp).to_pandas()
    if ".csv" in filename:
        with fs.open(filename, "rb") as fp:
            return pd.read_csv(fp)
    if ".json" in filename:
        with fs.open(filename, "rb") as fp:
            return json.loads(fp.read())
    if ".toml" in filename:
        with fs.open(filename, "rb") as fp:
            return tomllib.load(fp)
    if ".txt" in filename:
        with fs.open(filename, "r") as fp:
            return fp.read()
    if ".yaml" in filename or ".yml" in filename:
        with fs.open(filename, "rb") as fp:
            return yaml.safe_load(fp)
    if ".zip" in filename:
        f = fs.open(filename)
        f.close()
        with DataZip(str(Path(config["cache_path"]) / cached_path(filename)), "r") as z:
            return dict(z.items())
    raise ValueError(
        f"{filename} is not a parquet, csv, json, toml, txt, yaml/yml, or zip."
    )


def write_cloud_file(data: pd.DataFrame | str | bytes, filename: str) -> None:
    """Writes economic results for patio data to a specified filename in Azure storage.

    Args:
        data: DataFrame, or str or bytes representing
        filename: Target filename for storing the results, it must include the
            container, full path, and appropriate file extension, i.e., parquet for
            a DataFrame; csv json yaml yml toml or txt for str/bytes.

    """
    name, _, suffix = (
        filename.removeprefix("az://").removeprefix("abfs://").partition(".")
    )
    fs = rmi_cloud_fs()
    if isinstance(data, pd.DataFrame):
        if suffix != "parquet":
            raise TypeError("to write a DataFrame as csv, pass it as a str or bytes")
        with fs.open(f"az://{name}.parquet", mode="wb") as f:
            data.to_parquet(f)
    elif isinstance(data, str | bytes):
        allowed_file_types = ("csv", "json", "yaml", "yml", "toml", "txt")
        if suffix.lower() not in allowed_file_types:
            raise AssertionError(
                f"Unsupported file format {suffix}, must be one of {allowed_file_types}"
            )
        with fs.open(f"az://{name}.{suffix}", mode="wb") as f:
            f.write(data.encode("utf-8") if isinstance(data, str) else data)
    else:
        raise RuntimeError(f"Unsupported type {type(data)}")


def etb_cloud_init(account_name, token, clobber):
    """Write SAS token file to disk.

    ACCOUNT_NAME: Azure account name.
    TOKEN: SAS token for the account, enter ``use_azure_cli`` to use that method
        rather than a token.
    """
    if ETB_AZURE_CONFIG_PATH.exists():
        if clobber:
            ETB_AZURE_CONFIG_PATH.unlink()
        else:
            raise FileExistsError(
                f"configuration already exists at {ETB_AZURE_CONFIG_PATH}, either use "
                f"`--clobber` flag to start from scratch or `etb cloud add` to add a "
                f"new account alongside the existing one."
            )
    configs = build_config_interactive(account_name, token)
    account_name, *_ = tuple(configs)
    with open(ETB_AZURE_CONFIG_PATH, "wb") as f:
        f.write(
            json.dumps(configs | {"active": account_name}, option=json.OPT_INDENT_2)
        )


def build_config_interactive(account_name, token):
    """Create config from args or user input."""
    if not account_name:
        account_name = click.prompt("Enter Azure Account Name: ")
    cache_path = AZURE_BASE_CACHE_PATH / account_name
    if not cache_path.exists():
        cache_path.mkdir()
    new_config = {
        account_name: {"account_name": account_name, "cache_path": str(cache_path)}
    }
    if not token:
        use_cli = click.prompt("Use Azure CLI authentication? [y/N]", default="n")
        if use_cli.casefold() == "n":
            token = click.prompt("Enter Azure Token: ", type=str)
        else:
            token = "use_azure_cli"  # noqa: S105
    if all(part in token.casefold() for part in ("azure", "cli")):
        new_config[account_name]["auth"] = {"anon": False}
    else:
        token = token.strip("'").strip('"').encode("utf-8")
        if token.startswith(b"sv="):
            token = base64.b64encode(token)
        if isinstance(token, bytes):
            token = token.decode("utf8")
        new_config[account_name]["auth"] = {"sas_token": token}
    return new_config


def etb_cloud_add(account_name, token, no_activate):
    """Add a new Azure account.

    account_name: Azure account name to add.
    token: SAS token for the account, enter ``use_azure_cli`` to use that method
        rather than a token.
    """
    if not ETB_AZURE_CONFIG_PATH.exists() and ETB_AZURE_ACCOUNT_NAME_PATH.exists():
        convert_cloud_config(keep_old=True)
    new_config = build_config_interactive(account_name, token)
    account_name, *_ = tuple(new_config)
    if ETB_AZURE_CONFIG_PATH.exists():
        activate = {} if no_activate else {"active": account_name}
        with open(ETB_AZURE_CONFIG_PATH) as f:
            configs = json.loads(f.read()) | new_config | activate
    else:
        configs = new_config | {"active": account_name}
    with open(ETB_AZURE_CONFIG_PATH, "wb") as f:
        f.write(json.dumps(configs, option=json.OPT_INDENT_2))
