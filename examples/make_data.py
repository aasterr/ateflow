"""Generates a synthetic dataset with a Simpson's paradox on the corridor scenario.

Ground truth: the causal effect of the LED on speed is +0.15.
The naive estimate comes out negative because the LED turns on more often
when the corridor is crowded, and crowding lowers speed.
"""

import numpy as np
import pandas as pd

TRUE_ATE = 0.15


def make(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    crowding = rng.integers(0, 3, n)                     # 0 empty, 1 medium, 2 full
    p_led = np.array([0.15, 0.5, 0.85])[crowding]        # confounding
    led = rng.binomial(1, p_led)
    speed = (
        1.10
        - 0.30 * crowding                                 # crowding slows people down
        + TRUE_ATE * led                                  # causal effect of the LED
        + rng.normal(0, 0.10, n)
    )
    waiting_time = 2.0 - 0.8 * led + rng.normal(0, 0.3, n)  # descendant of the treatment
    return pd.DataFrame(
        {"crowding": crowding, "led": led, "speed": speed, "waiting_time": waiting_time}
    )


if __name__ == "__main__":
    make().to_csv("examples/corridor.csv", index=False)
    print("wrote examples/corridor.csv  (true ATE =", TRUE_ATE, ")")
