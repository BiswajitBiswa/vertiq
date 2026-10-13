"""Experiment sizes match the textbook two-proportion formula, and the recommended next step follows the
evidence."""
import pytest

from vertiq.design import detectable_effect, next_step, sample_size


def test_equal_groups_match_the_standard_formula():
    # 10% to 15%, alpha 0.05, power 0.8, equal allocation: about 686 per group
    n = sample_size(0.10, 0.05, control_share=0.5)
    assert 670 <= n["control"] <= 700 and n["treated"] == n["control"]


def test_a_smaller_control_needs_more_entities_overall():
    eq = sample_size(0.2, -0.05, control_share=0.5)["total"]
    small = sample_size(0.2, -0.05, control_share=0.2)["total"]
    assert small > eq


def test_detectable_effect_round_trips_with_sample_size():
    n = sample_size(0.3, -0.06, control_share=0.25)
    mde = detectable_effect(n["treated"], n["control"], 0.3)
    assert 0.05 <= mde <= 0.07


def test_next_step_follows_the_evidence():
    clear = next_step({"effect": -0.1, "p_value": 0.001, "control_rate": 0.3}, 800, 200)
    assert clear["decision"] == "effect detected"
    unclear = next_step({"effect": -0.03, "p_value": 0.4, "control_rate": 0.3}, 300, 75)
    assert unclear["decision"] == "run a larger experiment" and unclear["needed"]["total"] > 375


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        sample_size(0.9, 0.2)
