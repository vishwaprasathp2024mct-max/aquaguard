"""
AquaGuard-X
AquaIntel SVM Model

Lightweight SVM anomaly detector.

Architecture:

Sensor Features
      ↓
StandardScaler
      ↓
SVC
      ↓
CalibratedClassifierCV
      ↓
Prediction + Confidence
"""

from typing import Dict, Any, List

import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "ph",
    "turbidity",
    "temperature",
    "flow"
]


class AquaSVM:

    """
    Lightweight SVM-based anomaly detector.

    Labels:

        0 → NORMAL
        1 → ABNORMAL / ANOMALY
    """

    def __init__(self):

        # -------------------------------------------------
        # Base SVM
        # -------------------------------------------------

        base_svm = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=42
        )

        # -------------------------------------------------
        # Probability calibration
        #
        # This replaces:
        #
        # SVC(probability=True)
        #
        # which causes a FutureWarning in newer
        # scikit-learn versions.
        # -------------------------------------------------

        calibrated_svm = CalibratedClassifierCV(
            base_svm,
            ensemble=False
        )

        # -------------------------------------------------
        # Complete ML pipeline
        # -------------------------------------------------

        self.model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "svm",
                calibrated_svm
            )
        ])

        self.trained = False

        self.training_samples = 0

        self.training_classes = []

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    @staticmethod
    def extract_features(
        reading: Dict[str, Any]
    ) -> List[float]:

        features = []

        for feature in FEATURES:

            value = reading.get(feature)

            if value is None:

                raise ValueError(
                    f"Missing required feature: {feature}"
                )

            try:

                value = float(value)

            except (TypeError, ValueError):

                raise ValueError(
                    f"Invalid value for feature "
                    f"'{feature}': {value}"
                )

            if not np.isfinite(value):

                raise ValueError(
                    f"Non-finite value for feature "
                    f"'{feature}': {value}"
                )

            features.append(value)

        return features

    # =====================================================
    # TRAINING
    # =====================================================

    def train(
        self,
        readings: List[Dict[str, Any]],
        labels: List[int]
    ):
        """
        Train the SVM.

        readings:
            List of sensor dictionaries.

        labels:
            0 = NORMAL
            1 = ABNORMAL
        """

        # -------------------------------------------------
        # Basic validation
        # -------------------------------------------------

        if not readings:

            raise ValueError(
                "Training readings cannot be empty"
            )

        if not labels:

            raise ValueError(
                "Training labels cannot be empty"
            )

        if len(readings) != len(labels):

            raise ValueError(
                "Number of readings must match "
                "number of labels"
            )

        if len(readings) < 10:

            raise ValueError(
                "At least 10 training samples "
                "are recommended"
            )

        # -------------------------------------------------
        # Validate labels
        # -------------------------------------------------

        labels = [
            int(label)
            for label in labels
        ]

        unique_classes = sorted(
            set(labels)
        )

        if len(unique_classes) < 2:

            raise ValueError(
                "SVM requires at least two classes: "
                "NORMAL (0) and ABNORMAL (1)"
            )

        # -------------------------------------------------
        # Validate class labels
        # -------------------------------------------------

        invalid_labels = [
            label
            for label in unique_classes
            if label not in (0, 1)
        ]

        if invalid_labels:

            raise ValueError(
                "Invalid labels found: "
                f"{invalid_labels}. "
                "Only 0 and 1 are supported."
            )

        # -------------------------------------------------
        # Feature matrix
        # -------------------------------------------------

        X = np.array(
            [
                self.extract_features(reading)
                for reading in readings
            ],
            dtype=float
        )

        y = np.array(
            labels,
            dtype=int
        )

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        self.model.fit(
            X,
            y
        )

        # -------------------------------------------------
        # Store training information
        # -------------------------------------------------

        self.trained = True

        self.training_samples = len(
            readings
        )

        self.training_classes = unique_classes

        return {
            "status": "TRAINED",
            "samples": self.training_samples,
            "classes": self.training_classes
        }

    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(
        self,
        reading: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict whether a sensor pattern is normal
        or anomalous.
        """

        if not self.trained:

            raise RuntimeError(
                "SVM model has not been trained"
            )

        # -------------------------------------------------
        # Extract features
        # -------------------------------------------------

        X = np.array(
            [
                self.extract_features(reading)
            ],
            dtype=float
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = int(
            self.model.predict(X)[0]
        )

        # -------------------------------------------------
        # Probability / confidence
        # -------------------------------------------------

        probabilities = self.model.predict_proba(X)[0]

        confidence = float(
            np.max(probabilities)
        )

        # -------------------------------------------------
        # Classification
        # -------------------------------------------------

        if prediction == 0:

            classification = "NORMAL"

        else:

            classification = "ANOMALY"

        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

        return {
            "prediction": classification,
            "label": prediction,
            "confidence": round(
                confidence,
                4
            ),
            "probabilities": {
                "normal": round(
                    float(probabilities[0]),
                    4
                ),
                "anomaly": round(
                    float(probabilities[1]),
                    4
                )
            }
        }

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    def get_model_info(self):

        return {
            "model": "SVM",
            "kernel": "RBF",
            "features": FEATURES,
            "trained": self.trained,
            "training_samples": self.training_samples,
            "training_classes": self.training_classes
        }