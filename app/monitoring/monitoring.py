"""Monitoring file.

This module provides streamlit real-time monitoring dashboard.
"""

import streamlit as st
import pandas as pd
import json
import os

from streamlit_autorefresh import st_autorefresh

from pos_classifier.config.config import MONITORING_PATH

CATEGORIES = [
    "Beverages",
    "Dry Goods & Pantry Staples",
    "Fresh & Perishable Items",
    "Household & Personal Care",
    "Specialty & Miscellaneous",
]


def load_monitoring_data():
    """Load monitoring data from the JSON file.

    Returns
    -------
    dict
        Dictionary containing monitoring metrics, or empty if not available.

    """
    try:
        with open(MONITORING_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def display_category_counters(data):
    """Display the current count for each product category.

    Parameters
    ----------
    data : dict
        Monitoring data loaded from the JSON file.

    """
    st.subheader("Category Counters")
    cols = st.columns(5)
    for i, category in enumerate(CATEGORIES):
        with cols[i]:
            st.metric(label=category, value=data.get(category, 0))


def display_prediction_metrics(data):
    """Display total and correct prediction metrics along with a bar chart.

    Parameters
    ----------
    data : dict
        Monitoring data loaded from the JSON file.

    """
    st.subheader("Prediction Metrics")
    correct = data.get("correct_predictions", 0)
    total = data.get("total_predictions", 0)
    accuracy = round((correct / total) * 100, 2) if total > 0 else 0.0

    bar_data = pd.DataFrame(
        {
            "Predictions": ["Total Predictions", "Correct Predictions"],
            "Count": [total, correct],
        }
    )

    cols = st.columns(2)
    with cols[0]:
        st.bar_chart(bar_data.set_index("Predictions"))

    with cols[1]:
        st.metric("Accuracy (%)", accuracy)


def display_request_time(data):
    """Display request timing metrics (average, max, and total).

    Parameters
    ----------
    data : dict
        Monitoring data loaded from the JSON file.

    """
    st.markdown("Request Times")
    cols_time = st.columns(3)
    cols_time[0].metric("Avg. Request Time (s)", data.get("avg_time", 0.0))
    cols_time[1].metric("Max Request Time (s)", data.get("max_time", 0.0))
    cols_time[2].metric("Total Request Time (s)", data.get("total_time", 0.0))


def reset_monitoring_data():
    """Delete the monitoring JSON file to reset all metrics."""
    if os.path.exists(MONITORING_PATH):
        os.remove(MONITORING_PATH)
    st.success("Metrics file removed.")


st.set_page_config(page_title="Real-Time Monitoring", layout="wide")
st.title("Real-Time Product Category Monitoring")

st_autorefresh(interval=5000, limit=None, key="data_refresh")

if st.button("Reset Metrics"):
    reset_monitoring_data()

monitoring_data = load_monitoring_data()
display_category_counters(monitoring_data)
st.markdown("---")
display_prediction_metrics(monitoring_data)
st.markdown("---")
display_request_time(monitoring_data)
