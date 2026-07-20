# Forests and Single Trees

Decision trees, Random Forests, and ExtraTrees share native tree format, sampling
semantics, and prediction aggregation. Forests provide deterministic `random_state`
and `n_jobs` scheduling. ExtraTrees use random threshold candidates while reusing
native split, tree, prediction, and forest-container logic.
