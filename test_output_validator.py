import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from ai.output_validator import validate_output


# ==================================================
# CASE 1 - NORMAL
# ==================================================

result = validate_output(
    {
        "status": "NORMAL"
    },
    {
        "prediction": "NORMAL",
        "confidence": 0.95
    },
    {
        "status": "NORMAL"
    }
)

print("\nCASE 1 - NORMAL")
print(result)


# ==================================================
# CASE 2 - WARNING
# ==================================================

result = validate_output(
    {
        "status": "WARNING"
    },
    {
        "prediction": "ANOMALY",
        "confidence": 0.91
    },
    {
        "status": "WARNING"
    }
)

print("\nCASE 2 - WARNING")
print(result)


# ==================================================
# CASE 3 - CAUTION
# ==================================================

result = validate_output(
    {
        "status": "NORMAL"
    },
    {
        "prediction": "ANOMALY",
        "confidence": 0.88
    },
    {
        "status": "CAUTION"
    }
)

print("\nCASE 3 - CAUTION")
print(result)


# ==================================================
# CASE 4 - CRITICAL OVERRIDE
# ==================================================

result = validate_output(
    {
        "status": "CRITICAL"
    },
    {
        "prediction": "NORMAL",
        "confidence": 0.96
    },
    {
        "status": "NORMAL"
    }
)

print("\nCASE 4 - CRITICAL OVERRIDE")
print(result)