"""
Viridis feature leakage firewall.

This module defines forbidden target-derived fields.

Any model-training script should import and use these rules.
"""

FORBIDDEN_FEATURE_NAMES = {
    "future_acres",
    "future_acres_change_pct",
    "high_decline",
    "future_year",
    "target",
    "label",
    "outcome",
}

FORBIDDEN_NAME_FRAGMENTS = (
    "future",
    "target",
    "label",
    "outcome",
)


def assert_no_leakage(feature_names):
    """
    Fail loudly if a feature name appears to contain target/future data.
    """

    for feature in feature_names:

        name = str(feature).lower()

        if name in FORBIDDEN_FEATURE_NAMES:
            raise ValueError(
                f"LEAKAGE DETECTED: {feature}"
            )

        if any(fragment in name for fragment in FORBIDDEN_NAME_FRAGMENTS):
            raise ValueError(
                f"POSSIBLE LEAKAGE DETECTED: {feature}"
            )

    return True
