import sys
import os

# Allow Python to find the AQX project root
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from ai.svm_model import AquaSVM


# ==================================================
# TRAINING DATA
# ==================================================

training_data = []
labels = []


# --------------------------------------------------
# NORMAL samples
# --------------------------------------------------

for i in range(20):

    training_data.append({
        "ph": 7.0 + (i % 5) * 0.05,
        "turbidity": 1.5 + (i % 4) * 0.2,
        "temperature": 25.0 + (i % 5),
        "flow": 10.0 + (i % 4)
    })

    labels.append(0)


# --------------------------------------------------
# ABNORMAL samples
# --------------------------------------------------

for i in range(20):

    training_data.append({
        "ph": 5.0 + (i % 5) * 0.1,
        "turbidity": 8.0 + (i % 4) * 0.5,
        "temperature": 38.0 + (i % 5),
        "flow": 2.0 + (i % 3)
    })

    labels.append(1)


# ==================================================
# CREATE AND TRAIN SVM
# ==================================================

svm = AquaSVM()

svm.train(
    training_data,
    labels
)

print("\nSVM TRAINING SUCCESSFUL")


# ==================================================
# NORMAL TEST
# ==================================================

normal_reading = {
    "ph": 7.2,
    "turbidity": 2.0,
    "temperature": 28.0,
    "flow": 11.0
}

normal_result = svm.predict(normal_reading)

print("\nNORMAL READING")
print(normal_result)


# ==================================================
# ABNORMAL TEST
# ==================================================

abnormal_reading = {
    "ph": 5.5,
    "turbidity": 9.0,
    "temperature": 40.0,
    "flow": 2.0
}

abnormal_result = svm.predict(abnormal_reading)

print("\nABNORMAL READING")
print(abnormal_result)