"""Set-based macro F0.5; every reference entity, including singletons, counts."""
import numpy as np

def entity_f05(truth, predicted):
    truth, predicted = set(truth), set(predicted)
    if not truth:
        return float(not predicted)
    tp = len(truth & predicted)
    return 1.25 * tp / (0.25 * len(truth) + len(predicted))

def evaluate(truth, predicted, ids=None):
    ids = list(truth if ids is None else ids)
    if not ids:
        return {k: None for k in ("macro_F0.5", "precision", "recall", "singleton_F0.5", "matched_F0.5", "multi_match_F0.5")}
    if set(predicted) - set(truth):
        raise ValueError("Predictions contain unknown S1 IDs")
    scores, singleton, matched, multi = [], [], [], []
    tp = actual = proposed = 0
    for entity in ids:
        t, p = set(truth[entity]), set(predicted.get(entity, []))
        score = entity_f05(t, p)
        scores.append(score)
        (matched if t else singleton).append(score)
        if len(t) > 1:
            multi.append(score)
        tp += len(t & p)
        actual += len(t)
        proposed += len(p)
    mean = lambda xs: float(np.mean(xs)) if xs else None
    return {"macro_F0.5": mean(scores), "precision": tp / proposed if proposed else 0.0,
            "recall": tp / actual if actual else 0.0, "singleton_F0.5": mean(singleton),
            "matched_F0.5": mean(matched), "multi_match_F0.5": mean(multi)}

def blocking_metrics(truth, candidates, ids=None):
    ids = list(truth if ids is None else ids)
    total = sum(len(truth[i]) for i in ids)
    found = sum(len(set(truth[i]) & set(candidates.get(i, []))) for i in ids)
    matched = [i for i in ids if truth[i]]
    complete = lambda subset: sum(set(truth[i]) <= set(candidates.get(i, [])) for i in subset) / len(subset) if subset else None
    return {"candidate_pair_recall": found / total if total else None,
            "entity_complete_coverage": complete(ids), "matched_entity_complete_coverage": complete(matched)}
