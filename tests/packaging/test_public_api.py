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
        "IsolationForest",
        "LearningToRankRegressor",
        "MPSBoostClassifier",
        "MPSIsolationForest",
        "MPSBoostRegressor",
        "MetricHistory",
        "MetricObservation",
        "PortableBackendDecision",
        "PortableEstimatorAdapter",
        "mps_setup_instructions",
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
        "choose_portable_backend",
        "create_cache",
        "estimator_capability",
        "estimator_capabilities",
        "estimator_status",
        "export_native_trees_for_shap",
        "is_available",
        "is_shap_available",
        "mps_training_families",
        "official_shap_tree_explainer",
        "optional_dependency_status",
        "ordered_boosting_permutations",
        "planned_estimators",
        "portable_setup_instructions",
        "random_threshold_candidates",
        "require_estimator_supported",
        "sample_without_replacement_indices",
        "system_info",
        "subsample_feature_indices",
        "tree_family_spec",
        "tree_family_specs",
        "validate_indices_cover_range",
        "warn_if_mps_unavailable",
        "shap_setup_instructions",
    }


def test_estimator_capability_registry_reports_available_and_planned_models():
    """The registry should expose completed estimators and still fail early for unknown names."""

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
        "IsolationForest",
        "MPSIsolationForest",
        "LearningToRankRegressor",
    )
    assert mb.estimator_status("GradientBoostingClassifier") == "available"
    assert mb.estimator_status("DecisionTreeRegressor") == "available"
    assert mb.estimator_status("RandomForestRegressor") == "available"
    assert mb.estimator_status("ExtraTreesClassifier") == "available"
    assert mb.estimator_status("CatBoostRegressor") == "available"
    assert mb.estimator_status("CatBoostClassifier") == "available"
    assert mb.estimator_status("IsolationForest") == "available"
    assert mb.estimator_status("MPSIsolationForest") == "available"
    assert mb.estimator_status("LearningToRankRegressor") == "available"
    assert hasattr(mb, "CatBoostRegressor")
    assert hasattr(mb, "CatBoostClassifier")
    assert mb.IsolationForest is mb.MPSIsolationForest
    mb.require_estimator_supported("CatBoostRegressor")

    try:
        mb.estimator_status("DefinitelyNotAnEstimator")
    except ValueError as exc:
        assert "Unknown estimator" in str(exc)
    else:
        raise AssertionError("unknown estimator did not fail early")
