# goal: build a model that can infer dual threshold parameters for a signal that give best burst detection parameters.
# hints from ryan: use pytorch clamps to constrain the variables into reasonable ranges.
# the idea is to train a model to beat the specparam get peaks function + dual_thresh=(1,2) param set.

