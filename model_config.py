"""
Central configuration for the Viridis predictive model.

This file intentionally contains NO target-derived variables.
"""

ORIGIN_TRAIN = 2012
ORIGIN_TEST = 2017

TARGET = "high_decline"

# These are the ONLY types of variables we intend to use initially.
#
# The exact raw NASS mappings will be finalized after inspecting
# the cleanly copied data.
#
# Do NOT add future-period measurements here.
CANDIDATE_FEATURES = [
    "acres",
    "farm_count",
    "avg_farm_size",
    "acres_chg_prior",
    "farms_chg_prior",
    "size_chg_prior",
    "producer_age",
    "net_income_per_operation",
    "sales_per_operation",
    "government_payments",
    "operations_with_loss",
    "cropland_acres",
    "rented_acres",
]
