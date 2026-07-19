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
        "EstimatorCapability",
        "GradientBoostingRegressor",
        "MPSBoostRegressor",
        "MetricHistory",
        "MetricObservation",
        "EarlyStoppingDecision",
        "EarlyStoppingMonitor",
        "DeviceDecision",
        "TreeFamilySpec",
        "__version__",
        "available_estimators",
        "bootstrap_sample_indices",
        "cache_info",
        "clear_cache",
        "choose_device",
        "create_cache",
        "estimator_capability",
        "estimator_capabilities",
        "estimator_status",
        "is_available",
        "mps_training_families",
        "ordered_boosting_permutations",
        "planned_estimators",
        "random_threshold_candidates",
        "require_estimator_supported",
        "sample_without_replacement_indices",
        "system_info",
        "subsample_feature_indices",
        "tree_family_spec",
        "tree_family_specs",
        "validate_indices_cover_range",
    }


def test_estimator_capability_registry_fails_early_for_planned_models():
    """Planned tree names must be discoverable without exporting fake estimator classes."""

    assert mb.estimator_status("GradientBoostingRegressor") == "available"
    assert mb.estimator_capability("GradientBoostingRegressor").family.task == (
        "regression"
    )
    assert mb.available_estimators() == (
        "GradientBoostingRegressor",
        "MPSBoostRegressor",
    )
    assert "RandomForestRegressor" in mb.planned_estimators()
    assert "ExtraTreesClassifier" in mb.planned_estimators()
    assert "CatBoostRegressor" in mb.planned_estimators()
    assert "CatBoostClassifier" in mb.planned_estimators()
    assert not hasattr(mb, "RandomForestRegressor")
    assert not hasattr(mb, "CatBoostRegressor")

    try:
        mb.require_estimator_supported("RandomForestRegressor")
    except NotImplementedError as exc:
        assert "planned for MPSBoost v2" in str(exc)
    else:
        raise AssertionError("planned estimator did not fail early")

    try:
        mb.estimator_status("DefinitelyNotAnEstimator")
    except ValueError as exc:
        assert "Unknown estimator" in str(exc)
    else:
        raise AssertionError("unknown estimator did not fail early")
