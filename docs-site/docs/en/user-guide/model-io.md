# Model Saving and Loading

MPSBoost uses a versioned native model format for numeric regression, binary
classification, native CPU multiclass softmax, and advanced-objective metadata.
Files contain model structure, version and objective metadata, feature/bin schema,
and classifier class mapping. They never contain training data, credentials,
telemetry, or device identifiers.
