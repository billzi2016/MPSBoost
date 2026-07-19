"""Public API, estimator entry point, and single-source version tests."""

from importlib.metadata import version

import mpsboost as mb


def test_version_is_injected_by_native_build():
    """The native package version must match installed package metadata."""

    assert mb.__version__ == version("mpsboost")


def test_completed_estimators_are_public():
    """Only completed estimators may be exported from the public package namespace."""

    assert mb.GradientBoostingRegressor is mb.MPSBoostRegressor
    assert mb.GradientBoostingClassifier is mb.MPSBoostClassifier
    assert mb.GradientBoostingRegressor.__name__ == "MPSBoostRegressor"
    assert mb.MPSBoostRegressor.__name__ == "MPSBoostRegressor"
    assert mb.GradientBoostingClassifier.__name__ == "MPSBoostClassifier"
    assert set(mb.__all__) == {
        "CatBoostClassifier",
        "CatBoostRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "EstimatorCapability",
        "ExtraTreeClassifier",
        "ExtraTreeRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "MPSBoostClassifier",
        "MPSBoostRegressor",
        "MetricHistory",
        "MetricObservation",
        "RandomForestClassifier",
        "RandomForestRegressor",
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


def test_estimator_capability_registry_reports_available_and_planned_models():
    """The registry should expose completed estimators and still fail early for planned names."""

    assert mb.estimator_status("GradientBoostingRegressor") == "available"
    assert mb.estimator_capability("GradientBoostingRegressor").family.task == (
        "regression"
    )
    assert mb.available_estimators() == (
        "GradientBoostingRegressor",
        "MPSBoostRegressor",
        "GradientBoostingClassifier",
        "MPSBoostClassifier",
        "RandomForestRegressor",
        "RandomForestClassifier",
        "ExtraTreesRegressor",
        "ExtraTreesClassifier",
        "ExtraTreeRegressor",
        "ExtraTreeClassifier",
        "DecisionTreeRegressor",
        "DecisionTreeClassifier",
        "CatBoostRegressor",
        "CatBoostClassifier",
    )
    assert mb.estimator_status("GradientBoostingClassifier") == "available"
    assert mb.estimator_status("DecisionTreeRegressor") == "available"
    assert mb.estimator_status("RandomForestRegressor") == "available"
    assert mb.estimator_status("ExtraTreesClassifier") == "available"
    assert mb.estimator_status("CatBoostRegressor") == "available"
    assert mb.estimator_status("CatBoostClassifier") == "available"
    assert hasattr(mb, "CatBoostRegressor")
    assert hasattr(mb, "CatBoostClassifier")
    mb.require_estimator_supported("CatBoostRegressor")

    try:
        mb.estimator_status("DefinitelyNotAnEstimator")
    except ValueError as exc:
        assert "Unknown estimator" in str(exc)
    else:
        raise AssertionError("unknown estimator did not fail early")
