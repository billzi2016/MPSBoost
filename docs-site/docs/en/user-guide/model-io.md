# Model Saving and Loading

MPSBoost uses a versioned native model format.

## Supported scope

Currently supported:

- numeric regression models
- binary classifier models
- native CPU multiclass softmax models
- advanced regression objective metadata

## Design constraints

Saved files must contain:

- model structure
- version metadata
- objective metadata
- feature/bin schema
- class mapping, for classifiers

Saved files must not contain:

- training data
- credentials
- telemetry
- device identifiers

Loading performs compatibility validation to prevent reading a model with the wrong estimator or
wrong objective.
