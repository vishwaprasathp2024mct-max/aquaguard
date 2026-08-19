"""
AquaGuard-X
AquaIntel Rule Engine

Deterministic water-quality validation.
"""

from typing import Dict, Any


# ---------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------

PH_LOW = 6.5
PH_HIGH = 8.5

TURBIDITY_HIGH = 5.0

TEMPERATURE_LOW = 10.0
TEMPERATURE_HIGH = 35.0

FLOW_LOW = 5.0


# ---------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------

def evaluate_rules(reading: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate known water-quality safety rules.

    Returns deterministic rule results.
    """

    violations = []

    ph = reading.get("ph")
    turbidity = reading.get("turbidity")
    temperature = reading.get("temperature")
    flow = reading.get("flow")

    # -------------------------
    # pH
    # -------------------------

    if ph is not None:

        if ph < PH_LOW:
            violations.append({
                "parameter": "ph",
                "condition": "LOW",
                "value": ph,
                "threshold": PH_LOW,
                "message": f"pH below safe threshold ({PH_LOW})"
            })

        elif ph > PH_HIGH:
            violations.append({
                "parameter": "ph",
                "condition": "HIGH",
                "value": ph,
                "threshold": PH_HIGH,
                "message": f"pH above safe threshold ({PH_HIGH})"
            })

    # -------------------------
    # Turbidity
    # -------------------------

    if turbidity is not None and turbidity > TURBIDITY_HIGH:

        violations.append({
            "parameter": "turbidity",
            "condition": "HIGH",
            "value": turbidity,
            "threshold": TURBIDITY_HIGH,
            "message": f"Turbidity above threshold ({TURBIDITY_HIGH})"
        })

    # -------------------------
    # Temperature
    # -------------------------

    if temperature is not None:

        if temperature < TEMPERATURE_LOW:

            violations.append({
                "parameter": "temperature",
                "condition": "LOW",
                "value": temperature,
                "threshold": TEMPERATURE_LOW,
                "message": f"Temperature below threshold ({TEMPERATURE_LOW})"
            })

        elif temperature > TEMPERATURE_HIGH:

            violations.append({
                "parameter": "temperature",
                "condition": "HIGH",
                "value": temperature,
                "threshold": TEMPERATURE_HIGH,
                "message": f"Temperature above threshold ({TEMPERATURE_HIGH})"
            })

    # -------------------------
    # Flow
    # -------------------------

    if flow is not None and flow < FLOW_LOW:

        violations.append({
            "parameter": "flow",
            "condition": "LOW",
            "value": flow,
            "threshold": FLOW_LOW,
            "message": f"Flow below threshold ({FLOW_LOW})"
        })

    # -----------------------------------------------------
    # Determine rule status
    # -----------------------------------------------------

    violation_count = len(violations)

    if violation_count == 0:
        status = "NORMAL"

    elif violation_count == 1:
        status = "WARNING"

    else:
        status = "CRITICAL"

    return {
        "status": status,
        "violation_count": violation_count,
        "violations": violations
    }