from datetime import datetime


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse a timestamp string into a datetime object.

    Args:
    timestamp_str: A string representing a timestamp. This should be in the format "%Y-%m-%d %H:%M:%S.%f" or "%Y-%m-%d %H:%M:%S".

    Returns:
    A datetime object representing the given timestamp.
    """
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    return timestamp
