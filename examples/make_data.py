"""Genera un dataset sintetico con paradosso di Simpson sullo scenario corridoio.

Verità a terra: l'effetto causale del LED sulla velocità è +0.15.
La stima naive risulta negativa perché il LED si accende più spesso quando
il corridoio è affollato, e l'affollamento abbassa la velocità.
"""

import numpy as np
import pandas as pd

TRUE_ATE = 0.15


def make(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    crowding = rng.integers(0, 3, n)                     # 0 vuoto, 1 medio, 2 pieno
    p_led = np.array([0.15, 0.5, 0.85])[crowding]        # confondimento
    led = rng.binomial(1, p_led)
    speed = (
        1.10
        - 0.30 * crowding                                 # l'affollamento rallenta
        + TRUE_ATE * led                                  # effetto causale del LED
        + rng.normal(0, 0.10, n)
    )
    waiting_time = 2.0 - 0.8 * led + rng.normal(0, 0.3, n)  # discendente del trattamento
    return pd.DataFrame(
        {"crowding": crowding, "led": led, "speed": speed, "waiting_time": waiting_time}
    )


if __name__ == "__main__":
    make().to_csv("examples/corridor.csv", index=False)
    print("scritto examples/corridor.csv  (ATE vero =", TRUE_ATE, ")")
