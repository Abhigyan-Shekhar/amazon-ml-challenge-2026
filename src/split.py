from sklearn.model_selection import GroupShuffleSplit

def grouped_split(ids, seed=2026, validation_fraction=0.2):
    ids = list(ids)
    if len(ids) < 5 or len(ids) != len(set(ids)):
        raise ValueError("Need at least five unique S1 entities for grouped validation")
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed).split(ids, groups=ids))
    return [ids[i] for i in train], [ids[i] for i in validation]
