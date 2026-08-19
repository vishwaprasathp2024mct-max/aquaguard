"""
AquaGuard-X
AquaIntel Decision Fusion

Combines deterministic rule results and
SVM anomaly detection.
"""


def fuse_decision(rule_result, svm_result):
    """
    Combine rule-based safety validation and SVM output.

    Deterministic rules always have priority over SVM.
    """

    rule_status = rule_result.get("status", "NORMAL")

    svm_prediction = svm_result.get(
        "prediction",
        "NORMAL"
    )

    svm_confidence = svm_result.get(
        "confidence",
        0.0
    )

    # --------------------------------------------------
    # CRITICAL RULE VIOLATION
    # --------------------------------------------------

    if rule_status == "CRITICAL":

        final_status = "CRITICAL"

        reason = (
            "Critical rule violations detected. "
            "Deterministic safety rules override "
            "the machine-learning prediction."
        )

    # --------------------------------------------------
    # WARNING + SVM ANOMALY
    # --------------------------------------------------

    elif (
        rule_status == "WARNING"
        and svm_prediction == "ANOMALY"
    ):

        final_status = "WARNING"

        reason = (
            "Known rule violation and SVM anomaly "
            "detection agree."
        )

    # --------------------------------------------------
    # WARNING + SVM NORMAL
    # --------------------------------------------------

    elif (
        rule_status == "WARNING"
        and svm_prediction == "NORMAL"
    ):

        final_status = "WARNING"

        reason = (
            "A known rule violation exists. "
            "SVM did not detect an additional anomaly."
        )

    # --------------------------------------------------
    # NORMAL + SVM ANOMALY
    # --------------------------------------------------

    elif (
        rule_status == "NORMAL"
        and svm_prediction == "ANOMALY"
    ):

        final_status = "CAUTION"

        reason = (
            "No known rule violation was detected, "
            "but SVM identified an unusual sensor pattern."
        )

    # --------------------------------------------------
    # NORMAL + SVM NORMAL
    # --------------------------------------------------

    else:

        final_status = "NORMAL"

        reason = (
            "Rules and SVM both indicate normal "
            "water-quality conditions."
        )

    return {
        "status": final_status,
        "svm_confidence": svm_confidence,
        "reason": reason
    }