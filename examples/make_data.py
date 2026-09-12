"""Generates the synthetic example datasets, each with a known ground truth.

corridor: the causal effect of the LED on speed is +0.15. The naive estimate
comes out negative because the LED turns on more often when the corridor is
crowded, and crowding lowers speed.

onboarding: the causal effect of an onboarding email on 30-day retention is
+0.09. The growth team sent the email mostly to the users it feared would
churn (free plan, paid-ads signups), so the naive comparison hides the effect.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TRUE_ATE = 0.15

# Onboarding ground truth. The email acts directly (+0.04) and through
# first-week activity (+0.25 activation * +0.20 retention = +0.05).
ONBOARDING_DIRECT = 0.04
ONBOARDING_ACTIVATION_LIFT = 0.25
ONBOARDING_ACTIVITY_EFFECT = 0.20
ONBOARDING_TRUE_ATE = ONBOARDING_DIRECT + ONBOARDING_ACTIVATION_LIFT * ONBOARDING_ACTIVITY_EFFECT


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


def make_onboarding(n: int = 8000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plan = rng.choice(["free", "pro"], n, p=[0.7, 0.3])
    channel = rng.choice(["organic", "paid", "referral"], n, p=[0.4, 0.4, 0.2])
    pro = (plan == "pro").astype(float)

    # Targeting: free users from paid ads were the churn worry, so they got the email.
    p_email = np.where(
        pro == 1,
        pd.Series(channel).map({"organic": 0.25, "paid": 0.40, "referral": 0.15}),
        pd.Series(channel).map({"organic": 0.60, "paid": 0.85, "referral": 0.50}),
    )
    email = rng.binomial(1, p_email)

    # Mediator: the email pushes people to come back in the first week.
    active = rng.binomial(1, 0.35 + ONBOARDING_ACTIVATION_LIFT * email)

    p_retained = (
        0.30
        + 0.30 * pro
        + pd.Series(channel).map({"organic": 0.05, "paid": -0.15, "referral": 0.15}).to_numpy()
        + ONBOARDING_DIRECT * email
        + ONBOARDING_ACTIVITY_EFFECT * active
    )
    retained = rng.binomial(1, p_retained)
    return pd.DataFrame({
        "plan": plan,
        "channel": channel,
        "onboarding_email": email,
        "active_week1": active,
        "retained_30d": retained,
    })


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    make().to_csv(here / "corridor.csv", index=False)
    print("wrote examples/corridor.csv    (true ATE =", TRUE_ATE, ")")
    make_onboarding().to_csv(here / "onboarding.csv", index=False)
    print("wrote examples/onboarding.csv  (true ATE =", round(ONBOARDING_TRUE_ATE, 4), ")")
