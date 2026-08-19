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

from ai.rule_engine import evaluate_rules


normal_reading = {
    "ph": 7.2,
    "turbidity": 2.0,
    "temperature": 28.0,
    "flow": 12.0
}


abnormal_reading = {
    "ph": 7.2,
    "turbidity": 8.4,
    "temperature": 28.0,
    "flow": 11.0
}


critical_reading = {
    "ph": 5.5,
    "turbidity": 9.0,
    "temperature": 40.0,
    "flow": 2.0
}


print("\nNORMAL TEST")
print(evaluate_rules(normal_reading))


print("\nWARNING TEST")
print(evaluate_rules(abnormal_reading))


print("\nCRITICAL TEST")
print(evaluate_rules(critical_reading))