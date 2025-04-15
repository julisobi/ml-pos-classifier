"""Json monitor file.

This module provides methods for updating json file for real-time monitoring.
"""

import json

from pos_classifier.config.config import MONITORING_PATH


def update_monitoring_json(key: str):
    """Update monitoring counter for the given key in the JSON file.

    Parameters
    ----------
    key : str
       The key in the monitoring JSON to increment

    """
    try:
        with open(MONITORING_PATH) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    current_value = data.get(key, 0)
    data[key] = current_value + 1

    with open(MONITORING_PATH, "w") as f:
        json.dump(data, f, indent=4)


def update_prediction_time(time):
    """Update request timing statistics in the monitoring JSON file.

    Parameters
    ----------
    time : float
        Duration of the current request in seconds.

    """
    try:
        with open(MONITORING_PATH) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    total_requests = data.get("total_requests", 0)
    total_time = data.get("total_time", 0.0)
    max_time = data.get("max_time", 0.0)

    total_requests += 1
    total_time += time
    max_time = max(max_time, time)

    avg_time = total_time / total_requests if total_requests > 0 else 0.0

    data["total_requests"] = total_requests
    data["total_time"] = total_time
    data["max_time"] = max_time
    data["avg_time"] = avg_time

    with open(MONITORING_PATH, "w") as f:
        json.dump(data, f)
