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

from ai.aquaintel import AquaIntel


# =========================================================
# Training data
# =========================================================

training_data = []

labels = []


# Normal samples

for i in range(30):

    training_data.append({
        "ph": 7.0 + (i % 5) * 0.05,
        "turbidity": 1.5 + (i % 4) * 0.2,
        "temperature": 25.0 + (i % 5),
        "flow": 10.0 + (i % 4)
    })

    labels.append(0)


# Abnormal samples

for i in range(30):

    training_data.append({
        "ph": 5.0 + (i % 5) * 0.1,
        "turbidity": 8.0 + (i % 4) * 0.5,
        "temperature": 38.0 + (i % 5),
        "flow": 2.0 + (i % 3)
    })

    labels.append(1)


# =========================================================
# Create AquaIntel
# =========================================================

agent = AquaIntel()

agent.train_svm(
    training_data,
    labels
)


# =========================================================
# Current reading
# =========================================================

current_reading = {

    "ph": 7.2,

    "turbidity": 8.4,

    "temperature": 28.0,

    "flow": 11.0
}


# =========================================================
# Historical readings
# =========================================================

history = [

    {
        "ph": 7.1,
        "turbidity": 2.0,
        "temperature": 27.0,
        "flow": 12.0
    },

    {
        "ph": 7.2,
        "turbidity": 3.0,
        "temperature": 28.0,
        "flow": 11.0
    },

    {
        "ph": 7.2,
        "turbidity": 8.4,
        "temperature": 28.0,
        "flow": 11.0
    }
]


# =========================================================
# Run AquaIntel
# =========================================================

result = agent.analyze(
    current_reading,
    history
)


# =========================================================
# Display result
# =========================================================

import json

print(
    json.dumps(
        result,
        indent=4
    )
)