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

from ai.decison_fusion import fuse_decision


# ==================================================
# CASE 1
# ==================================================

result = fuse_decision(
    {
        "status": "NORMAL"
    },
    {
        "prediction": "NORMAL",
        "confidence": 0.95
    }
)

print("\nCASE 1")
print(result)


# ==================================================
# CASE 2
# ==================================================

result = fuse_decision(
    {
        "status": "WARNING"
    },
    {
        "prediction": "ANOMALY",
        "confidence": 0.91
    }
)

print("\nCASE 2")
print(result)


# ==================================================
# CASE 3
# ==================================================

result = fuse_decision(
    {
        "status": "NORMAL"
    },
    {
        "prediction": "ANOMALY",
        "confidence": 0.88
    }
)

print("\nCASE 3")
print(result)


# ==================================================
# CASE 4
# ==================================================

result = fuse_decision(
    {
        "status": "CRITICAL"
    },
    {
        "prediction": "NORMAL",
        "confidence": 0.96
    }
)

print("\nCASE 4")
print(result)