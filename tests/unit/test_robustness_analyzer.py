from chrysalis.src.analysis.robustness_analyzer import evaluate_success


def test_evaluate_success_list():
    run_data = {"main_results": {"magnetization": [0.1, 0.5, 0.9]}}
    # Final value 0.9 >= 0.8 -> True
    assert evaluate_success(run_data, threshold=0.8)
    # Final value 0.9 < 0.95 -> False
    assert not evaluate_success(run_data, threshold=0.95)


def test_evaluate_success_scalar():
    run_data = {"main_results": {"order_param": 0.85}}
    assert evaluate_success(run_data, threshold=0.8)
    assert not evaluate_success(run_data, threshold=0.9)


def test_evaluate_success_missing():
    run_data = {"main_results": {}}
    assert not evaluate_success(run_data)


def test_evaluate_success_negative():
    run_data = {"main_results": {"magnetization": -0.9}}
    # Should use abs()
    assert evaluate_success(run_data, threshold=0.8)
