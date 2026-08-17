"""Shared feature-extraction contract for training and inference."""

# Increment whenever feature order, dimension, or semantics changes. A trained
# bundle must declare the same version even when its feature count is unchanged.
FEATURE_SCHEMA_VERSION = "2"

__all__ = ["FEATURE_SCHEMA_VERSION"]
