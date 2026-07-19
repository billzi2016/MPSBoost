"""Tree-family semantic registry tests.

These tests protect the v2 foundation from drifting into fake estimator classes or duplicated
per-model math. The registry may describe planned models, but only implemented estimators may be
exported as classes.
"""

import mpsboost as mb


def test_available_regressor_and_alias_share_one_family_contract():
    """The short name and project-branded alias must share one implementation contract."""

    primary = mb.estimator_capability("GradientBoostingRegressor")
    branded = mb.estimator_capability("MPSBoostRegressor")

    assert primary.family_key == "histogram_gbdt_regression"
    assert branded.family_key == primary.family_key
    assert branded.alias_for == "GradientBoostingRegressor"
    assert primary.family is branded.family
    assert primary.family.objective == "squared_error"
    assert primary.family.growth == "level_wise"
    assert primary.family.aggregation == "sum"
    assert primary.family.supports_mps_training is True
    assert "histogram_gbdt_regression" in mb.mps_training_families()


def test_available_classifier_and_alias_share_one_family_contract():
    """The binary classifier should be represented as a real available estimator family."""

    primary = mb.estimator_capability("GradientBoostingClassifier")
    branded = mb.estimator_capability("MPSBoostClassifier")

    assert primary.family_key == "histogram_gbdt_classification"
    assert branded.family_key == primary.family_key
    assert branded.alias_for == "GradientBoostingClassifier"
    assert primary.family is branded.family
    assert primary.family.task == "classification"
    assert primary.family.objective == "logistic"
    assert primary.family.supports_mps_training is True
    assert "histogram_gbdt_classification" in mb.mps_training_families()


def test_available_decision_trees_have_single_tree_family_contracts():
    """Decision tree estimators should be available without placeholder classes."""

    regressor = mb.estimator_capability("DecisionTreeRegressor")
    classifier = mb.estimator_capability("DecisionTreeClassifier")

    assert regressor.family_key == "decision_tree_regression"
    assert regressor.family.task == "regression"
    assert regressor.family.supports_mps_training is True
    assert classifier.family_key == "decision_tree_classification"
    assert classifier.family.task == "classification"
    assert classifier.family.supports_mps_training is True


def test_available_random_forests_have_independent_tree_family_contracts():
    """Random forest estimators should expose implemented bagging semantics."""

    regressor = mb.estimator_capability("RandomForestRegressor")
    classifier = mb.estimator_capability("RandomForestClassifier")

    assert regressor.family_key == "random_forest_regression"
    assert regressor.family.aggregation == "mean"
    assert regressor.family.supports_mps_training is True
    assert classifier.family_key == "random_forest_classification"
    assert classifier.family.aggregation == "vote"
    assert classifier.family.supports_mps_training is True


def test_available_extra_trees_have_random_split_family_contracts():
    """ExtraTrees estimators should expose native random-threshold split semantics."""

    regressor = mb.estimator_capability("ExtraTreesRegressor")
    classifier = mb.estimator_capability("ExtraTreesClassifier")
    single = mb.estimator_capability("ExtraTreeRegressor")

    assert regressor.family_key == "extra_trees_regression"
    assert regressor.family.objective == "random_split"
    assert regressor.family.supports_mps_training is True
    assert classifier.family_key == "extra_trees_classification"
    assert classifier.family.supports_mps_training is True
    assert single.alias_for == "ExtraTreesRegressor"


def test_planned_tree_families_are_specs_not_fake_classes():
    """Planned models must be queryable without appearing as importable estimators."""

    planned = set(mb.planned_estimators())
    assert {
        "IsolationForest",
        "LearningToRankRegressor",
    } <= planned

    for name in planned:
        assert not hasattr(mb, name)
        capability = mb.estimator_capability(name)
        assert capability.family.supports_mps_training is False


def test_available_catboost_like_estimators_have_ordered_boosting_contracts():
    """CatBoost-like estimators should be public and wired to ordered-boosting specs."""

    regressor = mb.estimator_capability("CatBoostRegressor")
    classifier = mb.estimator_capability("CatBoostClassifier")

    assert mb.CatBoostRegressor.__name__ == "CatBoostRegressor"
    assert mb.CatBoostClassifier.__name__ == "CatBoostClassifier"
    assert regressor.family_key == "catboost_regression"
    assert regressor.family.growth == "ordered_boosting"
    assert regressor.family.supports_mps_training is True
    assert classifier.family_key == "catboost_classification"
    assert classifier.family.task == "classification"
    assert classifier.family.supports_mps_training is True


def test_family_specs_cover_every_estimator_capability_once():
    """Every estimator capability must point at a valid shared family spec."""

    family_keys = {spec.key for spec in mb.tree_family_specs()}
    capability_keys = {
        capability.family_key for capability in mb.estimator_capabilities()
    }

    assert capability_keys <= family_keys
    assert mb.tree_family_spec("random_forest_regression").aggregation == "mean"
    assert mb.tree_family_spec("random_forest_classification").aggregation == "vote"
    assert mb.tree_family_spec("extra_trees_regression").objective == "random_split"
    assert mb.tree_family_spec("catboost_regression").growth == "ordered_boosting"
    assert mb.tree_family_spec("catboost_classification").task == "classification"


def test_unknown_family_and_estimator_names_fail_early():
    """Typos in internal family keys or public estimator names must never silently pass."""

    for callable_, name in (
        (mb.tree_family_spec, "not_a_family"),
        (mb.estimator_capability, "not_an_estimator"),
    ):
        try:
            callable_(name)
        except ValueError as exc:
            assert "Unknown" in str(exc)
        else:
            raise AssertionError(f"{name} did not fail early")
