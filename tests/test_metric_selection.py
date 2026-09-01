import numpy as np
import pytest

from ifcred.core.fairness import FairnessEvaluation, assess_ifcred, evaluate_metric
from ifcred.metrics import WeightedPredictionConsistency, WeightedPredictionConsistencyInputs


class PresetFairnessMetric:
    """A minimal example proving that IF-CRED does not fix the F calculation."""

    name = "preset_test_metric"

    def evaluate(self, context: dict[str, float]) -> FairnessEvaluation:
        return FairnessEvaluation(metric_name=self.name, model_scores=context)


def test_ifcred_accepts_a_custom_selected_fairness_metric():
    evaluation = evaluate_metric(PresetFairnessMetric(), {"model_a": 0.4, "model_b": 0.6})
    result = assess_ifcred(C=0.8, D=0.9, fairness=evaluation)

    assert result.metric_name == "preset_test_metric"
    assert result.F == pytest.approx(0.5)
    assert result.M == pytest.approx(0.8)
    assert result.V == pytest.approx(0.8 * 0.9 * 0.5 * 0.8)
    assert result.F_min == pytest.approx(0.4)


def test_reference_metric_is_one_plugin_and_preserves_local_scores():
    context = WeightedPredictionConsistencyInputs(
        probabilities_by_model={"model": [0.2, 0.8]},
        neighbour_indices=np.array([[1], [0]]),
        weights=np.ones((2, 1)),
    )
    evaluation = evaluate_metric(WeightedPredictionConsistency(), context)

    assert evaluation.metric_name == "weighted_prediction_consistency"
    assert evaluation.model_scores["model"] == pytest.approx(0.4)
    np.testing.assert_allclose(evaluation.local_scores["model"], 0.4)


def test_metric_outputs_must_have_a_common_orientation_and_scale():
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        FairnessEvaluation(metric_name="unscaled", model_scores={"model": 12.0})


def test_metric_adapter_must_return_the_standardized_contract():
    class InvalidMetric:
        name = "invalid"

        def evaluate(self, context):
            return 0.5

    with pytest.raises(TypeError, match="FairnessEvaluation"):
        evaluate_metric(InvalidMetric(), None)
