"""Environment-specific knowledge for LunarLander-v3.

Everything here encodes facts about *this* environment: how wide its
observation dimensions are, how many actions it has, what those actions
mean, and how to read an episode's outcome. Agents and the CLI import
from here rather than hard-coding LunarLander details of their own.
"""
import numpy as np

ENV_ID = "LunarLander-v3"

# LunarLander's observation is 8-dimensional:
#   0-5 : continuous (x, y, vx, vy, angle, angular velocity)
#   6-7 : binary left/right leg-contact flags
#
# Gymnasium reports an unbounded observation space, so these are
# approximate practical ranges used for discretisation.
OBS_BOUNDS = np.array(
    [
        [-1.5, 1.5],
        [-0.5, 1.5],
        [-5.0, 5.0],
        [-5.0, 5.0],
        [-3.14, 3.14],
        [-5.0, 5.0],
    ]
)

# Number of leading dimensions covered by OBS_BOUNDS; the remainder are binary.
N_CONTINUOUS_DIMS = len(OBS_BOUNDS)
N_BINARY_DIMS = 2

N_ACTIONS = 4

ACTION_NAMES = {
    0: "noop",
    1: "left engine",
    2: "main engine",
    3: "right engine",
}


def format_action(action: int) -> str:
    """Render an action as `index (name)` for human-readable output."""
    return f"{action} ({ACTION_NAMES.get(action, '?')})"


def decode_outcome(terminated: bool, truncated: bool, total_reward: float) -> str:
    """Summarise how an episode ended.

    LunarLander does not report landed-vs-crashed directly, so the sign of
    the episode return is used as the discriminator on termination.
    """
    if truncated:
        return "TRUNCATED (time limit)"
    if terminated and total_reward < 0:
        return "CRASHED"
    return "LANDED"
