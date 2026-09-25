import numpy as np
from .evaluate import evaluate

def predict(candidates, threshold):
    selected = candidates[candidates.score >= threshold]
    return selected.groupby("source1_entity_id").target_entity_id.agg(set).to_dict()

def select_threshold(candidates, truth, ids):
    # Search only the calibration S1 groups. Include predict-none as a valid policy.
    trials = {}
    def score(t):
        t = float(t)
        if t not in trials:
            trials[t] = evaluate(truth, predict(candidates, t), ids)["macro_F0.5"]
    for t in np.r_[np.arange(0.30, 0.951, 0.05), 0.0, 1.000001]:
        score(t)
    best = max(trials, key=lambda t: (trials[t], t))
    for t in np.arange(max(0, best-0.05), min(1, best+0.05)+0.00001, 0.005):
        score(t)
    best = max(trials, key=lambda t: (trials[t], t))
    return best, [{"threshold": t, "macro_F0.5": v} for t, v in sorted(trials.items())]
