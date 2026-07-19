"""Public API, estimator entry point, and single-source version tests."""

from importlib.metadata import version

import mpsboost as mb


def test_version_is_injected_by_native_build():
    """The native package version must match installed package metadata."""

    assert mb.__version__ == version("mpsboost")


def test_only_completed_regressor_is_public():
    """Only completed estimators may be exported from the public package namespace."""

    assert mb.GradientBoostingRegressor is mb.MPSBoostRegressor
    assert mb.GradientBoostingRegressor.__name__ == "MPSBoostRegressor"
    assert mb.MPSBoostRegressor.__name__ == "MPSBoostRegressor"
    assert not hasattr(mb, "MPSBoostClassifier")
    assert set(mb.__all__) == {
        "GradientBoostingRegressor",
        "MPSBoostRegressor",
        "__version__",
        "cache_info",
        "clear_cache",
        "create_cache",
        "is_available",
        "system_info",
    }
