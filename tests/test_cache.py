"""Cache identity tests. These tests do not fit a generator."""

from experiments.cache import classifier_cache_hit, generator_cache_hit


def _request(**updates):
    payload = {
        "generator": "ctgan",
        "seed": 42,
        "n_rows": 100,
        "n_training_rows": 100,
        "hyperparameters": {"epochs": 80},
        "library_versions": {"sdv": "2.2.0"},
        "training_fingerprint": "train-hash",
        "generators_yaml_sha256": "yaml-hash",
        "synthetic_sha256": "csv-hash",
        "generation_json_sha256": "json-hash",
        "upstream_synthetic_sha256": None,
    }
    payload.update(updates)
    return payload


def _legacy(request):
    return {
        "status": "completed",
        "generator": request["generator"],
        "seed": request["seed"],
        "n_rows": request["n_rows"],
        "n_training_rows": request["n_training_rows"],
        "hyperparameters": request["hyperparameters"],
        "library_versions": request["library_versions"],
    }


def _ledger(request):
    return {
        "note": "recomputed_at_audit_not_stored_in_the_run",
        "recorded_at_training_time": False,
        "synthetic_sha256": request["synthetic_sha256"],
        "generation_json_sha256": request["generation_json_sha256"],
        "recomputed_training_fingerprint": request["training_fingerprint"],
    }


def test_legacy_cache_rejects_a_changed_hyperparameter():
    request = _request()
    changed = _request(hyperparameters={"epochs": 81})
    assert generator_cache_hit(_legacy(request), changed, _ledger(request)) is False


def test_legacy_cache_rejects_a_changed_training_fingerprint():
    request = _request()
    changed = _request(training_fingerprint="other-train")
    assert generator_cache_hit(_legacy(request), changed, _ledger(request)) is False


def test_legacy_cache_rejects_a_missing_ledger():
    request = _request()
    assert generator_cache_hit(_legacy(request), request, None) is False


def test_legacy_cache_accepts_matching_audit_evidence():
    request = _request()
    assert generator_cache_hit(_legacy(request), request, _ledger(request)) is True


def test_contract_cache_rejects_a_changed_generator_config_hash():
    request = _request()
    cached = _legacy(request)
    cached.update(
        {
            "cache_contract": 2,
            "training_fingerprint": request["training_fingerprint"],
            "generators_yaml_sha256": request["generators_yaml_sha256"],
            "upstream_synthetic_sha256": None,
        }
    )
    changed = _request(generators_yaml_sha256="edited")
    assert generator_cache_hit(cached, changed, None) is False


def test_classifier_cache_rejects_a_changed_dataset_fingerprint():
    context = {
        "experiment_id": "main__real__share-0.00__logistic_regression__seed-42",
        "dataset_fingerprint": "data",
        "split_fingerprint": "split",
        "config_fingerprint": "config",
        "seed": 42,
        "classifier": "logistic_regression",
        "n_real": 10,
        "n_synthetic": 0,
    }
    params = {"C": 0.1, "random_state": 42}
    cached = dict(context)
    cached.update(
        {
            "status": "completed",
            "hyperparameters": params,
            "n_train": 10,
            "test_metrics_computed": False,
            "scored_on": "validation",
            "metrics": {
                "roc_auc": 0.5,
                "f1": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "balanced_accuracy": 0.5,
            },
        }
    )
    assert classifier_cache_hit(cached, context, params) is True
    edited = dict(context)
    edited["dataset_fingerprint"] = "changed"
    assert classifier_cache_hit(cached, edited, params) is False
    assert classifier_cache_hit(cached, context, {"C": 1.0, "random_state": 42}) is False
