import pandas as pd
import streamlit as st
from streamlit_prophet.lib.utils.mapping import dayname_to_daynumber


def input_cleaning(resampling: dict, readme: dict) -> dict:
    """Lets the user enter cleaning specifications.

    Parameters
    ----------
    resampling : dict
        Dictionary containing dataset frequency information.
    readme : dict
        Dictionary containing tooltips to guide user's choices.

    Returns
    -------
    dict
        Cleaning specifications (remove_days, del_days, del_negative, del_zeros, log_transform).
    """
    # TODO : Ajouter une option "Remove holidays"
    cleaning = dict()
    if resampling["freq"][-1] in ["s", "H", "D"]:
        del_days = st.multiselect(
            "Remove days",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            default=[],
            help=readme["tooltips"]["remove_days"],
        )
        cleaning["del_days"] = dayname_to_daynumber(del_days)
    else:
        cleaning["del_days"] = []
    cleaning["del_zeros"] = st.checkbox(
        "Delete rows where target = 0", True, help=readme["tooltips"]["del_zeros"]
    )
    cleaning["del_negative"] = st.checkbox(
        "Delete rows where target < 0", True, help=readme["tooltips"]["del_negative"]
    )
    cleaning["log_transform"] = st.checkbox(
        "Target log transform", False, help=readme["tooltips"]["log_transform"]
    )
    return cleaning


def input_dimensions(df: pd.DataFrame, readme: dict) -> dict:
    """Lets the user enter filtering and aggregation specifications.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that will be used to detect dimension columns.
    readme : dict
        Dictionary containing tooltips to guide user's choices.

    Returns
    -------
    dict
        Filtering and aggregation specifications (dimensions, values to keep, aggregation function).
    """
    dimensions = dict()
    eligible_cols = set(df.columns) - {"ds", "y"}
    if len(eligible_cols) > 0:
        dimensions_cols = st.multiselect(
            "Select dataset dimensions if any",
            list(eligible_cols),
            default=_autodetect_dimensions(df),
            help=readme["tooltips"]["dimensions"],
        )
        for col in dimensions_cols:
            values = list(df[col].unique())
            if st.checkbox(
                f"Keep all values for {col}",
                True,
                help=readme["tooltips"]["dimensions_keep"] + col + ".",
            ):
                dimensions[col] = values.copy()
            else:
                dimensions[col] = st.multiselect(
                    f"Values to keep for {col}",
                    values,
                    default=[values[0]],
                    help=readme["tooltips"]["dimensions_filter"],
                )
        dimensions["agg"] = st.selectbox(
            "Target aggregation function over dimensions",
            ["Mean", "Sum", "Max", "Min"],
            help=readme["tooltips"]["dimensions_agg"],
        )
    else:
        st.write("Date and target are the only columns in your dataset, there are no dimensions.")
        dimensions["agg"] = "Mean"
    return dimensions


def _autodetect_dimensions(df: pd.DataFrame) -> list:
    """Detects dimension columns in input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that will be used to detect dimension columns.

    Returns
    -------
    list
        List of dimension columns detected. The user will be able to change that list later if it is incorrect.
    """
    eligible_cols = set(df.columns) - {"ds", "y"}
    detected_cols = []
    for col in eligible_cols:
        values = df[col].value_counts()
        values = values.loc[values > 0].to_list()
        if (len(values) > 1) & (len(values) < 0.05 * len(df)):
            if max(values) / min(values) <= 20:
                detected_cols.append(col)
    return detected_cols


def input_resampling(df: pd.DataFrame, readme: dict) -> dict:
    """Lets the user enter resampling specifications.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that will be used to detect current frequency in dataset.
    readme : dict
        Dictionary containing tooltips to guide user's choices.

    Returns
    -------
    dict
        Resampling specifications (resample or not, frequency, aggregation function).
    """
    resampling = dict()
    resampling["freq"] = _autodetect_freq(df)
    st.write(f"Frequency detected in dataset: {resampling['freq']}")
    resampling["resample"] = st.checkbox(
        "Resample my dataset", False, help=readme["tooltips"]["resample_choice"]
    )
    if resampling["resample"]:
        current_freq = resampling["freq"][-1]
        possible_freq_names = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
        possible_freq = [freq[0] for freq in possible_freq_names]
        current_freq_index = possible_freq.index(current_freq)
        if current_freq != "Y":
            new_freq = st.selectbox(
                "Select new frequency",
                possible_freq_names[current_freq_index + 1 :],
                help=readme["tooltips"]["resample_new_freq"],
            )
            resampling["freq"] = new_freq[0]
            resampling["agg"] = st.selectbox(
                "Target aggregation function when resampling",
                ["Mean", "Sum", "Max", "Min"],
                help=readme["tooltips"]["resample_agg"],
            )
        else:
            st.write("Frequency is already yearly, resampling is not possible.")
            resampling["resample"] = False
    return resampling


def _autodetect_freq(df: pd.DataFrame) -> str:
    """Detects date frequency of input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that will be used to detect dataset frequency.

    Returns
    -------
    str
        Frequency detected. The user will be able to resample later if it is not the value expected.
    """
    min_delta = pd.Series(df["ds"]).diff().min()
    days = min_delta.days
    seconds = min_delta.seconds
    if days == 1:
        return "D"
    elif days < 1:
        if seconds >= 3600:
            return f"{round(seconds/3600)}H"
        else:
            return f"{seconds}s"
    elif days > 1:
        if days < 7:
            return f"{days}D"
        elif days < 28:
            return f"{round(days/7)}W"
        elif days < 90:
            return f"{round(days/30)}M"
        elif days < 365:
            return f"{round(days/90)}Q"
        else:
            return f"{round(days/365)}Y"