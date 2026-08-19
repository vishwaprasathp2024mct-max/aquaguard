"""
AquaGuard-X
AquaIntel - Hybrid Water Intelligence Agent

Pipeline:

Sensor Data
    ↓
Rule Engine
    ↓
SVM
    ↓
Trend Analysis
    ↓
Decision Fusion
    ↓
Output Validation
    ↓
Final AquaIntel Result
"""

from typing import Dict, Any, List

from ai.rule_engine import evaluate_rules
from ai.svm_model import AquaSVM
from ai.decison_fusion import fuse_decision
from ai.output_validator import validate_output


# =========================================================
# AquaIntel Agent
# =========================================================

class AquaIntel:

    def __init__(self):

        self.svm = AquaSVM()

        self.svm_ready = False

    # =====================================================
    # Train SVM
    # =====================================================

    def train_svm(
        self,
        readings: List[Dict[str, Any]],
        labels: List[int]
    ):

        self.svm.train(
            readings,
            labels
        )

        self.svm_ready = True

    # =====================================================
    # Rule analysis
    # =====================================================

    def analyze_rules(
        self,
        reading: Dict[str, Any]
    ):

        return evaluate_rules(reading)

    # =====================================================
    # SVM analysis
    # =====================================================

    def analyze_svm(
        self,
        reading: Dict[str, Any]
    ):

        if not self.svm_ready:

            return {
                "prediction": "UNKNOWN",
                "label": -1,
                "confidence": 0.0,
                "available": False
            }

        result = self.svm.predict(
            reading
        )

        result["available"] = True

        return result

    # =====================================================
    # Trend analysis
    # =====================================================

    @staticmethod
    def analyze_trends(
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        if not history or len(history) < 2:

            return {
                "status": "INSUFFICIENT_DATA",
                "parameters": {}
            }

        parameters = [
            "ph",
            "turbidity",
            "temperature",
            "flow"
        ]

        trends = {}

        for parameter in parameters:

            values = []

            for reading in history:

                value = reading.get(parameter)

                if value is not None:

                    values.append(
                        float(value)
                    )

            if len(values) < 2:

                trends[parameter] = (
                    "INSUFFICIENT_DATA"
                )

                continue

            first = values[0]
            last = values[-1]

            difference = last - first

            # Small changes are treated as stable
            tolerance = max(
                abs(first) * 0.02,
                0.01
            )

            if difference > tolerance:

                trends[parameter] = "INCREASING"

            elif difference < -tolerance:

                trends[parameter] = "DECREASING"

            else:

                trends[parameter] = "STABLE"

        return {
            "status": "AVAILABLE",
            "parameters": trends
        }

    # =====================================================
    # Risk score
    # =====================================================

    @staticmethod
    def calculate_risk_score(
        rule_result,
        svm_result,
        final_status
    ):

        score = 0.0

        # Rule contribution
        violations = rule_result.get(
            "violation_count",
            0
        )

        if violations == 1:

            score += 0.35

        elif violations >= 2:

            score += 0.70

        # SVM contribution
        if svm_result.get(
            "prediction"
        ) == "ANOMALY":

            confidence = float(
                svm_result.get(
                    "confidence",
                    0.0
                )
            )

            score += 0.30 * confidence

        # Final status adjustment
        if final_status == "CRITICAL":

            score = max(
                score,
                0.90
            )

        elif final_status == "WARNING":

            score = max(
                score,
                0.60
            )

        elif final_status == "CAUTION":

            score = max(
                score,
                0.40
            )

        return round(
            min(score, 1.0),
            4
        )

    # =====================================================
    # Recommendations
    # =====================================================

    @staticmethod
    def generate_recommendations(
        rule_result,
        svm_result,
        trends,
        final_status
    ):

        recommendations = []

        # Rule-based recommendations

        for violation in rule_result.get(
            "violations",
            []
        ):

            parameter = violation.get(
                "parameter"
            )

            condition = violation.get(
                "condition"
            )

            if parameter == "ph":

                if condition == "LOW":

                    recommendations.append(
                        "Investigate acidic water conditions "
                        "and verify pH sensor calibration."
                    )

                else:

                    recommendations.append(
                        "Investigate alkaline water conditions "
                        "and verify pH sensor calibration."
                    )

            elif parameter == "turbidity":

                recommendations.append(
                    "Inspect the water source and filtration "
                    "system for increased suspended particles."
                )

            elif parameter == "temperature":

                recommendations.append(
                    "Check the water source and environmental "
                    "conditions affecting temperature."
                )

            elif parameter == "flow":

                recommendations.append(
                    "Inspect the flow path, pump and "
                    "possible blockage."
                )

        # SVM anomaly

        if svm_result.get(
            "prediction"
        ) == "ANOMALY":

            recommendations.append(
                "Review recent sensor patterns because "
                "the SVM detected an unusual combination "
                "of measurements."
            )

        # Trend warnings

        trend_parameters = trends.get(
            "parameters",
            {}
        )

        for parameter, direction in trend_parameters.items():

            if direction == "INCREASING":

                if parameter in [
                    "turbidity",
                    "temperature"
                ]:

                    recommendations.append(
                        f"Monitor increasing {parameter} trend."
                    )

            elif direction == "DECREASING":

                if parameter in [
                    "flow"
                ]:

                    recommendations.append(
                        f"Monitor decreasing {parameter} trend."
                    )

        if not recommendations:

            recommendations.append(
                "Continue routine water-quality monitoring."
            )

        # Remove duplicate recommendations

        return list(
            dict.fromkeys(
                recommendations
            )
        )

    # =====================================================
    # Explanation
    # =====================================================

    @staticmethod
    def generate_explanation(
        rule_result,
        svm_result,
        validation_result,
        final_status
    ):

        rule_status = rule_result.get(
            "status",
            "NORMAL"
        )

        svm_prediction = svm_result.get(
            "prediction",
            "UNKNOWN"
        )

        if (
            rule_status == "NORMAL"
            and svm_prediction == "NORMAL"
        ):

            return (
                "Known safety rules are within limits and "
                "the SVM detected no unusual sensor pattern."
            )

        if (
            rule_status == "WARNING"
            and svm_prediction == "ANOMALY"
        ):

            return (
                "A known water-quality rule was violated and "
                "the SVM independently detected an unusual "
                "sensor pattern. Both intelligence layers "
                "support the warning decision."
            )

        if (
            rule_status == "NORMAL"
            and svm_prediction == "ANOMALY"
        ):

            return (
                "No known rule violation was detected, but "
                "the SVM identified an unusual combination "
                "of sensor values. The result is therefore "
                "classified as caution for further review."
            )

        if final_status == "CRITICAL":

            return (
                "Multiple safety-rule violations were detected. "
                "The deterministic safety layer therefore "
                "forces a critical decision regardless of the "
                "SVM prediction."
            )

        if validation_result.get(
            "decision"
        ) == "CORRECTED":

            return (
                "The initial fused decision was inconsistent "
                "with deterministic safety constraints and "
                "was corrected by the output validator."
            )

        return (
            "AquaIntel combined deterministic rules, "
            "machine-learning analysis and validation "
            "to produce the final decision."
        )

    # =====================================================
    # Complete analysis
    # =====================================================

    def analyze(
        self,
        reading: Dict[str, Any],
        history: List[Dict[str, Any]] = None
    ):

        if history is None:

            history = []

        # ---------------------------------------------
        # Rule engine
        # ---------------------------------------------

        rule_result = self.analyze_rules(
            reading
        )

        # ---------------------------------------------
        # SVM
        # ---------------------------------------------

        svm_result = self.analyze_svm(
            reading
        )

        # ---------------------------------------------
        # Decision fusion
        # ---------------------------------------------

        if svm_result.get(
            "available",
            False
        ):

            fusion_result = fuse_decision(
                rule_result,
                svm_result
            )

        else:

            # Safe fallback while SVM is unavailable

            fusion_result = {
                "status": rule_result.get(
                    "status",
                    "NORMAL"
                ),
                "svm_confidence": 0.0,
                "reason": (
                    "SVM is not currently trained. "
                    "Deterministic rule result used."
                )
            }

        # ---------------------------------------------
        # Output validation
        # ---------------------------------------------

        validation_result = validate_output(
            rule_result,
            svm_result,
            fusion_result
        )

        final_status = validation_result.get(
            "status",
            fusion_result.get(
                "status",
                "NORMAL"
            )
        )

        # ---------------------------------------------
        # Trends
        # ---------------------------------------------

        trends = self.analyze_trends(
            history
        )

        # ---------------------------------------------
        # Risk
        # ---------------------------------------------

        risk_score = self.calculate_risk_score(
            rule_result,
            svm_result,
            final_status
        )

        # ---------------------------------------------
        # Recommendations
        # ---------------------------------------------

        recommendations = (
            self.generate_recommendations(
                rule_result,
                svm_result,
                trends,
                final_status
            )
        )

        # ---------------------------------------------
        # Explanation
        # ---------------------------------------------

        explanation = (
            self.generate_explanation(
                rule_result,
                svm_result,
                validation_result,
                final_status
            )
        )

        # ---------------------------------------------
        # Confidence
        # ---------------------------------------------

        svm_confidence = float(
            svm_result.get(
                "confidence",
                0.0
            )
        )

        if validation_result.get(
            "decision"
        ) == "ACCEPT":

            confidence = svm_confidence

        else:

            # Validation correction reduces confidence
            confidence = max(
                0.50,
                svm_confidence * 0.75
            )

        # ---------------------------------------------
        # Final result
        # ---------------------------------------------

        return {

            "agent": "AquaIntel",

            "status": final_status,

            "risk_score": risk_score,

            "confidence": round(
                confidence,
                4
            ),

            "rule_result": rule_result,

            "svm_result": svm_result,

            "fusion": fusion_result,

            "validation": validation_result,

            "trends": trends,

            "recommendations": recommendations,

            "explanation": explanation
        }


# =========================================================
# Global AquaIntel instance
# =========================================================

aqua_intel = AquaIntel()


# =========================================================
# Compatibility function
# =========================================================

def analyze_water(
    reading: Dict[str, Any],
    history: List[Dict[str, Any]] = None
):

    return aqua_intel.analyze(
        reading,
        history
    )