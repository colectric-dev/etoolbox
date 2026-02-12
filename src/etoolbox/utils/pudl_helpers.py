"""Helpers from PUDL for working with EIA and similar data.

These helpers are a selection of those included in the :mod:`pudl.helpers` module of the
`catalystcoop.pudl <https://github.com/catalyst-cooperative/pudl>`_ package created by
`Catalyst Cooperative <https://catalyst.coop>`_. From the original:

Copyright 2017-2022 Catalyst Cooperative

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import re
from collections.abc import Sequence
from functools import singledispatch

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

logger = logging.getLogger("etoolbox")


def fix_int_na(
    df: pd.DataFrame, columns, float_na=np.nan, int_na=-1, str_na=""
) -> pd.DataFrame:
    """Convert NA containing integer columns from float to string.

    Numpy doesn't have a real NA value for integers. When pandas stores integer
    data which has NA values, it thus upcasts integers to floating point
    values, using np.nan values for NA. However, in order to dump some of our
    dataframes to CSV files for use in data packages, we need to write out
    integer formatted numbers, with empty strings as the NA value. This
    function replaces np.nan values with a sentinel value, converts the column
    to integers, and then to strings, finally replacing the sentinel value with
    the desired NA string.

    This is an interim solution -- now that pandas extension arrays have been
    implemented, we need to go back through and convert all of these integer
    columns that contain NA values to Nullable Integer types like Int64.

    Args:
        df (pandas.DataFrame): The dataframe to be fixed. This argument allows
            method chaining with the pipe() method.
        columns (iterable of strings): A list of DataFrame column labels
            indicating which columns need to be reformatted for output.
        float_na (float): The floating point value to be interpreted as NA and
            replaced in col.
        int_na (int): Sentinel value to substitute for float_na prior to
            conversion of the column to integers.
        str_na (str): sa.String value to substitute for int_na after the column
            has been converted to strings.

    Returns:
        df (pandas.DataFrame): a new DataFrame, with the selected columns
        converted to strings that look like integers, compatible with
        the postgresql COPY FROM command.
    """
    return (
        df.replace(dict.fromkeys(columns, float_na), int_na)
        .astype(dict.fromkeys(columns, int))
        .astype(dict.fromkeys(columns, str))
        .replace({c: str(int_na) for c in columns}, str_na)
    )


@singledispatch
def month_year_to_date(df):
    """Convert all pairs of year/month fields in a dataframe into Date fields.

    This function finds all column names within a dataframe that match the
    regular expression '_month$' and '_year$', and looks for pairs that have
    identical prefixes before the underscore. These fields are assumed to
    describe a date, accurate to the month.  The two fields are used to
    construct a new _date column (having the same prefix) and the month/year
    columns are then dropped.

    Todo:
        This function needs to be combined with convert_to_date, and improved:
        * find and use a _day$ column as well
        * allow specification of default month & day values, if none are found.
        * allow specification of lists of year, month, and day columns to be
        combined, rather than automatically finding all the matching ones.
        * Do the Right Thing when invalid or NA values are encountered.

    Args:
        df (pandas.DataFrame): The DataFrame in which to convert year/months
            fields to Date fields.

    Returns:
        pandas.DataFrame: A DataFrame in which the year/month fields have been
        converted into Date fields.
    """
    raise NotImplementedError


@month_year_to_date.register(pd.DataFrame)
def _(df):
    df = df.copy()
    mo_re, yr_re = "_month$", "_year$"
    # Columns that match our month or year patterns.
    # Base column names that don't include the month or year pattern
    yr_base = [re.sub(yr_re, "", y) for y in df.filter(regex=yr_re)]

    # For each base column that DOES have both a month and year,
    # We need to grab the real column names corresponding to each,
    # so we can access the values in the data frame, and use them
    # to create a corresponding Date column named [BASE]_date
    month_year_date = []
    for m in df.filter(regex=mo_re):
        # We only want to retain columns that have BOTH month and year
        # matches -- otherwise there's no point in creating a Date.
        if (base := re.sub(mo_re, "", m)) in yr_base:
            (month_col,) = df.filter(regex=f"^{base}{mo_re}")
            (year_col,) = df.filter(regex=f"^{base}{yr_re}")
            month_year_date.append((month_col, year_col, f"{base}_date"))

    for month_col, year_col, date_col in month_year_date:
        df = fix_int_na(df, columns=[year_col, month_col])

        date_mask = (df[year_col] != "") & (df[month_col] != "")
        years = df.loc[date_mask, year_col]
        months = df.loc[date_mask, month_col]

        df.loc[date_mask, date_col] = pd.to_datetime(
            {"year": years, "month": months, "day": 1}, errors="coerce"
        )

        # Now that we've replaced these fields with a date, we drop them.
        df = df.drop([month_col, year_col], axis=1)

    return df


@month_year_to_date.register(pl.DataFrame | pl.LazyFrame)
def _[T: pl.DataFrame | pl.LazyFrame](df: T) -> T:
    mo_re, yr_re = "_month$", "_year$"
    # Base column names that don't include the month or year pattern
    yr_base = [
        re.sub(yr_re, "", y)
        for y in df.lazy().select(cs.matches(yr_re)).collect_schema().names()
    ]
    # For each base column that DOES have both a month and year,
    # We need to grab the real column names corresponding to each,
    # so we can access the values in the data frame, and use them
    # to create a corresponding Date column named [BASE]_date
    to_exclude, new_cols = [], {}
    for m in df.lazy().select(cs.matches(mo_re)).collect_schema().names():
        # We only want to retain columns that have BOTH month and year
        # matches -- otherwise there's no point in creating a Date.
        if (base := re.sub(mo_re, "", m)) in yr_base:
            to_exclude.extend((m_ := f"^{base}{mo_re}", y_ := f"^{base}{yr_re}"))
            new_cols[f"{base}_date"] = pl.datetime(pl.col(y_), pl.col(m_), 1)

    return df.select(pl.exclude(*to_exclude), **new_cols)


@singledispatch
def remove_leading_zeros_from_numeric_strings(
    df: pd.DataFrame, col_name: str
) -> pd.DataFrame:
    """Remove leading zeros frame column values that are numeric strings.

    Sometimes an ID column (like generator_id or unit_id) will be reported with leading
    zeros and sometimes it won't. For example, in the Excel spreadsheets published by
    EIA, the same generator may show up with the ID "0001" and "1" in different years
    This function strips the leading zeros from those numeric strings so the data can
    be mapped accross years and datasets more reliably.

    Alphanumeric generator IDs with leadings zeroes are not affected, as we
    found no instances in which an alphanumeric ID appeared both with
    and without leading zeroes. The ID "0A1" will stay "0A1".

    Args:
        df: A DataFrame containing the column you'd like to remove numeric leading zeros
            from.
        col_name: The name of the column you'd like to remove numeric leading zeros
            from.

    Returns:
        A DataFrame without leading zeros for numeric string values in the desired
        column.
    """
    raise NotImplementedError


@remove_leading_zeros_from_numeric_strings.register(pd.DataFrame)
def _(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    leading_zeros = df[col_name].str.contains(r"^0+\d+$").fillna(value=False)
    if leading_zeros.any():
        logger.debug("Fixing leading zeros in %s column", col_name)
        df.loc[leading_zeros, col_name] = df[col_name].str.replace(
            r"^0+", "", regex=True
        )
    else:
        logger.debug("Found no numeric leading zeros in %s", col_name)
    return df


@remove_leading_zeros_from_numeric_strings.register(pl.DataFrame | pl.LazyFrame)
def _[T: pl.DataFrame | pl.LazyFrame](df: T, col_name: str) -> T:
    return df.with_columns(
        pl.when(pl.col(col_name).str.contains(r"^0+\d+$"))
        .then(pl.col(col_name).str.replace(r"^0+", ""))
        .otherwise(pl.col(col_name))
        .alias(col_name)
    )


@singledispatch
def fix_eia_na(df):
    """Replace common ill-posed EIA NA spreadsheet values with np.nan.

    Currently, replaces empty string, single decimal points with no numbers,
    and any single whitespace character with np.nan.

    Args:
        df (pandas.DataFrame): The DataFrame to clean.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    raise NotImplementedError


@fix_eia_na.register(pd.DataFrame)
def _(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(
        to_replace=[
            r"^\.$",  # Nothing but a decimal point
            r"^\s*$",  # The empty string and entirely whitespace strings
        ],
        value=pd.NA,
        regex=True,
    )


@fix_eia_na.register(pl.DataFrame | pl.LazyFrame)
def _[T: pl.DataFrame | pl.LazyFrame](df: T) -> T:
    return df.with_columns(
        cs.by_dtype(pl.Utf8)
        .str.replace(
            r"^\.$",  # Nothing but a decimal point
            "__to_null__",
        )
        .str.replace(
            r"^\s*$",  # The empty string and entirely whitespace strings
            "__to_null__",
        )
        .replace({"__to_null__": None})
    )


@singledispatch
def simplify_columns(df):
    """Simplify column labels for use as snake_case database fields.

    All columns will be re-labeled by:
    * Replacing all non-alphanumeric characters with spaces.
    * Forcing all letters to be lower case.
    * Compacting internal whitespace to a single " ".
    * Stripping leading and trailing whitespace.
    * Replacing all remaining whitespace with underscores.

    Args:
        df : The DataFrame to clean.
    """
    raise NotImplementedError


@simplify_columns.register(pd.DataFrame)
def _(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.replace(r"[^0-9a-zA-Z]+", " ", regex=True)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(" ", "_")
    )
    return df


@simplify_columns.register(pl.DataFrame | pl.LazyFrame)
def _[T: pl.DataFrame | pl.LazyFrame](df: T) -> T:
    def renamer(string: str) -> str:
        return re.sub(
            r"\s+", " ", re.sub(r"[^0-9a-zA-Z]+", " ", string).strip().casefold()
        ).replace(" ", "_")

    return df.rename(renamer)


@singledispatch
def simplify_strings(df, columns: Sequence[str]):
    """Simplify the strings contained in a set of dataframe columns.

    Performs several operations to simplify strings for comparison and parsing purposes.
    These include removing Unicode control characters, stripping leading and trailing
    whitespace, using lowercase characters, and compacting all internal whitespace to a
    single space.

    Leaves null values unaltered. Casts other values with astype(str).

    Args:
        df: DataFrame whose columns are being cleaned up.
        columns (iterable): The labels of the string columns to be simplified.

    Returns:
        pandas.DataFrame: The whole DataFrame that was passed in, with
        the string columns cleaned up.
    """
    raise NotImplementedError


@simplify_strings.register(pd.DataFrame)
def _(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out_df = df.copy()
    for col in columns:
        if col in out_df.columns:
            out_df.loc[out_df[col].notna(), col] = (
                out_df.loc[out_df[col].notna(), col]
                .astype(str)
                .str.replace(r"[\x00-\x1f\x7f-\x9f]", "", regex=True)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
            )
    return out_df


@simplify_strings.register(pl.DataFrame | pl.LazyFrame)
def _[T: pl.DataFrame | pl.LazyFrame](df: T, columns: Sequence[str]) -> T:
    return df.with_columns(
        pl.col(*columns)
        .str.replace(r"[\x00-\x1f\x7f-\x9f]", "")
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace(r"\s+", " ")
    )


@singledispatch
def zero_pad_numeric_string(
    col: pd.Series,
    n_digits: int,
) -> pd.Series:
    """Clean up fixed-width leading zero padded numeric (e.g. ZIP, FIPS) codes.

    Often values like ZIP and FIPS codes are stored as integers, or get
    converted to floating point numbers because there are NA values in the
    column. Sometimes other non-digit strings are included like Canadian
    postal codes mixed in with ZIP codes, or IMP (imported) instead of a
    FIPS county code. This function attempts to manage these irregularities
    and produce either fixed-width leading zero padded strings of digits
    having a specified length (n_digits) or NA.

    * Convert the Series to a nullable string.
    * Remove any decimal point and all digits following it.
    * Remove any non-digit characters.
    * Replace any empty strings with NA.
    * Replace any strings longer than n_digits with NA.
    * Pad remaining digit-only strings to n_digits length.
    * Replace (invalid) all-zero codes with NA.

    Args:
        col: The Series to clean. May be numeric, string, object, etc.
        n_digits: the desired length of the output strings.

    Returns:
        A Series of nullable strings, containing only all-numeric strings
        having length n_digits, padded with leading zeroes if necessary.
    """
    raise NotImplementedError


@zero_pad_numeric_string.register(pd.Series)
def _(col: pd.Series, n_digits: int) -> pd.Series:
    out_col = (
        col.astype("string")
        # Remove decimal points and any digits following them.
        # This turns floating point strings into integer strings
        .replace(r"[\.]+\d*", "", regex=True)
        # Remove any whitespace
        .replace(r"\s+", "", regex=True)
        # Replace anything that's not entirely digits with NA
        .replace(r"[^\d]+", pd.NA, regex=True)
        # Set any string longer than n_digits to NA
        .replace(f"[\\d]{{{n_digits + 1},}}", pd.NA, regex=True)
        # Pad the numeric string with leading zeroes to n_digits length
        .str.zfill(n_digits)
        # All-zero ZIP & FIPS codes are invalid.
        # Also catches empty strings that were zero padded.
        .replace({n_digits * "0": pd.NA})
    )
    if not out_col.str.match(f"^[\\d]{{{n_digits}}}$").all():
        raise ValueError(
            f"Failed to generate zero-padded numeric strings of length {n_digits}."
        )
    return out_col


@zero_pad_numeric_string.register(pl.Expr)
def _(col: pl.Expr, n_digits: int) -> pl.Expr:
    transformed = (
        col
        # Remove decimal points and any digits following them.
        # This turns floating point strings into integer strings
        .str.replace(r"[\.]+\d*", "")
        # Remove any whitespace
        .str.replace(r"\s+", "")
        # Replace anything that's not entirely digits with NA
        .str.replace(r"[^\d]+", "__to_null__")
        # Set any string longer than n_digits to NA
        .str.replace(f"[\\d]{{{n_digits + 1},}}", "__to_null__")
    )
    return (
        pl.when(transformed.str.contains("__to_null__"))
        .then(pl.lit(None))
        .otherwise(
            transformed
            # Pad the numeric string with leading zeroes to n_digits length
            .str.zfill(n_digits)
            # All-zero ZIP & FIPS codes are invalid.
            # Also catches empty strings that were zero padded.
            .replace({n_digits * "0": None})
        )
    )


def weighted_average(df, data_col, weight_col, by):
    """Generate a weighted average.

    Args:
        df (pandas.DataFrame): A DataFrame containing, at minimum, the columns
            specified in the other parameters data_col and weight_col.
        data_col (string): column name of data column to average
        weight_col (string): column name to weight on
        by (list): A list of the columns to group by when calcuating
            the weighted average value.

    Returns:
        pandas.DataFrame: a table with ``by`` columns as the index and the
        weighted ``data_col``.
    """
    df["_data_times_weight"] = df[data_col] * df[weight_col]
    df["_weight_where_notnull"] = df.loc[df[data_col].notna(), weight_col]
    g = df.groupby(by, observed=True)
    result = g["_data_times_weight"].sum(min_count=1) / g["_weight_where_notnull"].sum(
        min_count=1
    )
    del df["_data_times_weight"], df["_weight_where_notnull"]
    return result.to_frame(name=data_col)  # .reset_index()


def sum_and_weighted_average_agg(
    df_in: pd.DataFrame,
    by: list,
    sum_cols: list | None = None,
    agg_dict: dict | None = None,
    wtavg_dict: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate dataframe by summing and using weighted averages.

    Many times we want to aggreate a data table using the same groupby columns
    but with different aggregation methods. This function combines two of our
    most common aggregation methods (summing and applying a weighted average)
    into one function. Because pandas does not have a built-in weighted average
    method for groupby we use :func:`weighted_average`.

    Args:
        df_in: input table to aggregate. Must have columns
            in ``id_cols``, ``sum_cols`` and keys from ``wtavg_dict``.
        by: columns to group/aggregate based on. These columns
            will be passed as an argument into grouby as ``by`` arg.
        sum_cols: columns to sum.
        agg_dict: dictionary of columns (keys) and function (values) passed to
            :meth:`pandas.DataFrame.agg`.
        wtavg_dict: dictionary of columns to average (keys) and
            columns to weight by (values).

    Returns:
        table with join of columns from ``by``, ``sum_cols`` and keys of
        ``wtavg_dict``. Primary key of table will be ``by``.
    """
    logger.debug("grouping by %s", by)
    # we are keeping the index here for easy merging of the weighted cols below
    if sum(x is None for x in (sum_cols, agg_dict)) != 1:
        raise ValueError("specify one and only one of sum_cols or agg_dict")
    elif sum_cols is not None:
        df_out = df_in.groupby(by=by, as_index=True, observed=True)[sum_cols].sum(
            min_count=1
        )
    else:
        df_out = df_in.groupby(by=by, as_index=True, observed=True).agg(agg_dict)

    if wtavg_dict is not None:
        for data_col, weight_col in wtavg_dict.items():
            df_out.loc[:, data_col] = weighted_average(
                df_in, data_col=data_col, weight_col=weight_col, by=by
            )[data_col]
    return df_out.reset_index()
