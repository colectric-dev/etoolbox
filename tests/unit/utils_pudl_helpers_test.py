import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from etoolbox.utils.pudl_helpers import (
    fix_eia_na,
    month_year_to_date,
    remove_leading_zeros_from_numeric_strings,
    simplify_columns,
    simplify_strings,
    sum_and_weighted_average_agg,
    zero_pad_numeric_string,
)
from etoolbox.utils.testing import assert_equal, idfn


@pytest.mark.parametrize(
    "in_df, comp_func, expected",
    [
        (
            pd.DataFrame(
                {
                    "vals": [
                        0,  # Don't touch integers, even if they're null-ish
                        0.0,  # Don't touch floats, even if they're null-ish
                        "0.",  # Should only replace naked decimals
                        ".0",  # Should only replace naked decimals
                        "..",  # Only replace single naked decimals
                        "",
                        " ",
                        "\t",
                        ".",
                        "  ",  # Multiple whitespace characters
                        "\t\t",  # 2-tabs: another Multi-whitespace
                    ]
                }
            ),
            pd.testing.assert_frame_equal,
            pd.DataFrame(
                {
                    "vals": [
                        0,
                        0.0,
                        "0.",
                        ".0",
                        "..",
                        pd.NA,
                        pd.NA,
                        pd.NA,
                        pd.NA,
                        pd.NA,
                        pd.NA,
                    ]
                }
            ),
        ),
        (
            pl.DataFrame(
                {
                    "vals": [
                        "0.",  # Should only replace naked decimals
                        ".0",  # Should only replace naked decimals
                        "..",  # Only replace single naked decimals
                        "",
                        " ",
                        "\t",
                        ".",
                        "  ",  # Multiple whitespace characters
                        "\t\t",  # 2-tabs: another Multi-whitespace
                    ]
                }
            ),
            pl.testing.assert_frame_equal,
            pl.DataFrame(
                {
                    "vals": [
                        "0.",
                        ".0",
                        "..",
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ]
                }
            ),
        ),
        (
            pl.LazyFrame(
                {
                    "vals": [
                        "0.",  # Should only replace naked decimals
                        ".0",  # Should only replace naked decimals
                        "..",  # Only replace single naked decimals
                        "",
                        " ",
                        "\t",
                        ".",
                        "  ",  # Multiple whitespace characters
                        "\t\t",  # 2-tabs: another Multi-whitespace
                    ]
                }
            ),
            pl.testing.assert_frame_equal,
            pl.LazyFrame(
                {
                    "vals": [
                        "0.",
                        ".0",
                        "..",
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ]
                }
            ),
        ),
    ],
    ids=idfn,
)
def test_fix_eia_na(in_df, comp_func, expected):
    """Test cleanup of bad EIA spreadsheet NA values."""
    out_df = fix_eia_na(in_df)
    comp_func(out_df, expected)


@pytest.mark.parametrize(
    "obj, comp_func",
    [
        (pd.DataFrame, pd.testing.assert_frame_equal),
        (pl.DataFrame, pl.testing.assert_frame_equal),
        (pl.LazyFrame, pl.testing.assert_frame_equal),
    ],
    ids=idfn,
)
def test_remove_leading_zeros_from_numeric_strings(obj, comp_func):
    """Test removal of leading zeroes from EIA generator IDs."""
    in_df = obj(
        {
            "generator_id": [
                "0001",  # Leading zeroes, all numeric string.
                "26",  # An appropriate numeric string w/o leading zeroes.
                "100",  # Integer, should get stringified.
                "100.0",  # What happens if it's a float?
                "01-A",  # Leading zeroes, alphanumeric. Should not change.
                "HRSG-01",  # Alphanumeric, should be no change.
            ]
        }
    )
    expected_df = obj(
        {
            "generator_id": [
                "1",
                "26",
                "100",
                "100.0",
                "01-A",
                "HRSG-01",
            ]
        }
    )
    out_df = remove_leading_zeros_from_numeric_strings(in_df, "generator_id")
    comp_func(out_df, expected_df)


@pytest.mark.parametrize(
    "df, comp_func, expected",
    [
        (
            pd.DataFrame(np.eye(3), columns=["WHat?", "FOO", "A  LONG column name."]),
            lambda x: list(x.columns),
            ["what", "foo", "a_long_column_name"],
        ),
        (
            pl.DataFrame(np.eye(3), schema=["WHat?", "FOO", "A  LONG column name."]),
            lambda x: x.columns,
            ["what", "foo", "a_long_column_name"],
        ),
        (
            pl.LazyFrame(np.eye(3), schema=["WHat?", "FOO", "A  LONG column name."]),
            lambda x: x.collect_schema().names(),
            ["what", "foo", "a_long_column_name"],
        ),
    ],
    ids=idfn,
)
def test_simplify_columns(df, comp_func, expected):
    """Test helper that simplifies column names."""
    assert_equal(
        comp_func(simplify_columns(df)),
        expected,
    )


@pytest.mark.parametrize(
    "obj, comp_func",
    [
        (pd.DataFrame, pd.testing.assert_frame_equal),
        (pl.DataFrame, pl.testing.assert_frame_equal),
        (pl.LazyFrame, pl.testing.assert_frame_equal),
    ],
    ids=idfn,
)
def test_month_year_to_date(obj, comp_func):
    """Test helper that replaces year/month columns with date."""
    data = {
        "report_year": [2020, 2019, 2018, 2020],
        "report_month": [1, 1, 1, 6],
        "build_year": [2020, 2019, 2018, 2020],
        "data": range(4),
    }
    df = obj(data)
    expected = obj(
        {
            "build_year": data["build_year"],
            "data": data["data"],
            "report_date": [
                datetime.datetime(y, m, 1)
                for y, m in zip(data["report_year"], data["report_month"], strict=True)
            ],
        }
    )
    df = month_year_to_date(df)
    comp_func(df, expected)


@pytest.mark.parametrize(
    "df,n_digits",
    [
        (
            pd.DataFrame(
                [
                    (512, "512"),
                    (5, "005"),
                    (5.0, "005"),
                    (5.00, "005"),
                    ("5.0", "005"),
                    ("5.", "005"),
                    ("005", "005"),
                    (0, pd.NA),
                    (-5, pd.NA),
                    ("000", pd.NA),
                    ("5000", pd.NA),
                    ("IMP", pd.NA),
                    ("I9P", pd.NA),
                    ("", pd.NA),
                    ("nan", pd.NA),
                    (np.nan, pd.NA),
                ],
                columns=["input", "expected"],
            ).convert_dtypes(),
            3,
        ),
        (
            pd.DataFrame(
                [
                    (93657, "93657"),
                    (93657.0, "93657"),
                    ("93657.0", "93657"),
                    (9365.7, "09365"),
                    (9365, "09365"),
                    ("936S7", pd.NA),
                    ("80302-7509", pd.NA),
                    ("B2A X19", pd.NA),
                    ("", pd.NA),
                    ("nan", pd.NA),
                    (np.nan, pd.NA),
                ],
                columns=["input", "expected"],
            ).convert_dtypes(),
            5,
        ),
    ],
    ids=idfn,
)
def test_zero_pad_numeric_string(df, n_digits):
    """Test zero-padding of numeric codes like ZIP and FIPS."""
    output = zero_pad_numeric_string(df["input"], n_digits)
    pd.testing.assert_series_equal(
        output,
        df["expected"],
        check_names=False,
    )
    # Make sure all outputs are the right length
    assert (output.str.len() == n_digits).all()
    # Make sure all outputs are entirely numeric
    assert output.str.match(f"^[\\d]{{{n_digits}}}$").all()


@pytest.mark.parametrize(
    "df,n_digits",
    [
        (
            pl.DataFrame(
                [
                    ("512", "512"),
                    ("5", "005"),
                    ("5.00", "005"),
                    ("5.0", "005"),
                    ("5.", "005"),
                    ("005", "005"),
                    ("0", None),
                    ("-5", None),
                    ("000", None),
                    ("5000", None),
                    ("IMP", None),
                    ("I9P", None),
                    ("", None),
                    ("nan", None),
                ],
                schema=["input", "expected"],
            ),
            3,
        ),
        (
            pl.DataFrame(
                [
                    ("93657", "93657"),
                    ("93657.0", "93657"),
                    ("93657.0", "93657"),
                    ("9365.7", "09365"),
                    ("9365", "09365"),
                    ("936S7", None),
                    ("80302-7509", None),
                    ("B2A X19", None),
                    ("", None),
                    ("nan", None),
                ],
                schema=["input", "expected"],
            ),
            5,
        ),
    ],
    ids=idfn,
)
def test_zero_pad_numeric_string_pl(df, n_digits):
    """Test zero-padding of numeric codes like ZIP and FIPS."""
    df = df.with_columns(
        transformed=pl.col("input").pipe(zero_pad_numeric_string, n_digits)
    )
    pl.testing.assert_series_equal(
        df["transformed"],
        df["expected"],
        check_names=False,
    )
    # Make sure all outputs are the right length
    assert df.select(
        (pl.col("transformed").drop_nulls().str.len_chars() == n_digits).all()
    ).item()
    # Make sure all outputs are entirely numeric
    assert df.select(
        (pl.col("transformed").drop_nulls().str.replace(r"\d+", "") == "").all()
    ).item()


@pytest.mark.parametrize(
    "df, comp_func, expected",
    [
        (
            pd.DataFrame({"a": ["Hello  World"]}),
            pd.testing.assert_frame_equal,
            pd.DataFrame({"a": ["hello world"]}),
        ),
        (
            pl.DataFrame({"a": ["Hello  World"]}),
            pl.testing.assert_frame_equal,
            pl.DataFrame({"a": ["hello world"]}),
        ),
        (
            pl.LazyFrame({"a": ["Hello  World"]}),
            pl.testing.assert_frame_equal,
            pl.LazyFrame({"a": ["hello world"]}),
        ),
    ],
    ids=idfn,
)
def test_simplify_strings(df, comp_func, expected):
    """Test helper that simplify strings."""
    comp_func(
        df.pipe(simplify_strings, columns=["a"]),
        expected,
    )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"sum_cols": ["a"], "wtavg_dict": {"b": "a", "c": "a"}},
            pd.DataFrame(
                {
                    "ix": [1, 2],
                    "a": [100, 1000],
                    "b": [36.0, 430.2],
                    "c": [40.0, 71.3],
                }
            ),
        ),
        (
            {
                "agg_dict": {"a": "sum", "d": "first"},
                "wtavg_dict": {"b": "a", "c": "a"},
            },
            pd.DataFrame(
                {
                    "ix": [1, 2],
                    "a": [100, 1000],
                    "d": ["a", "c"],
                    "b": [36.0, 430.2],
                    "c": [40.0, 71.3],
                }
            ),
        ),
        (
            {
                "agg_dict": {"a": "sum", "d": "first"},
            },
            pd.DataFrame(
                {
                    "ix": [1, 2],
                    "a": [100, 1000],
                    "d": ["a", "c"],
                }
            ),
        ),
        ({"wtavg_dict": {"b": "a", "c": "a"}}, ValueError),
        ({"agg_dict": {"a": "sum", "d": "first"}, "sum_cols": ["b"]}, ValueError),
    ],
    ids=idfn,
)
def test_sum_and_weighted_average_agg(kwargs, expected):
    """Test weighted averages."""
    data = pd.DataFrame(
        {
            "ix": [1, 1, 2, 2, 2],
            "a": [80, 20, 150, 250, 600],
            "b": [40, 20, 18, 30, 700],
            "c": [25, 100, 32, 50, 90],
            "d": ["a", "b", "c", "d", "e"],
        }
    )
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(
            sum_and_weighted_average_agg(data, by=["ix"], **kwargs),
            expected,
        )
    else:
        with pytest.raises(expected):
            sum_and_weighted_average_agg(data, by=["ix"], **kwargs)
