"""Environment construction and per-environment metadata.

`make` is a thin wrapper over `gym.make` so callers have one place to go,
and `spec_for` resolves an env ID to the module describing it. Agents that
need environment-specific constants (observation bounds, action count) go
through `spec_for` instead of hard-coding them, which means an unsupported
env ID fails loudly here rather than silently producing a useless agent.
"""
from types import ModuleType

import gymnasium as gym

from rl_games.envs import lunar_lander

DEFAULT_ENV_ID = lunar_lander.ENV_ID

# env ID -> module holding that environment's constants and helpers
_REGISTRY: dict[str, ModuleType] = {
    lunar_lander.ENV_ID: lunar_lander,
}

__all__ = ["DEFAULT_ENV_ID", "make", "spec_for", "supported_env_ids"]


def supported_env_ids() -> tuple[str, ...]:
    """Env IDs this package has agent-facing metadata for."""
    return tuple(_REGISTRY)


def spec_for(env_id: str) -> ModuleType:
    """Return the metadata module for `env_id`.

    Raises:
        ValueError: if the environment has no registered metadata. Agents
            rely on these constants, so guessing would mean training a
            silently mis-specified agent.
    """
    try:
        return _REGISTRY[env_id]
    except KeyError:
        supported = ", ".join(supported_env_ids())
        raise ValueError(
            f"No environment metadata registered for {env_id!r}. "
            f"Supported: {supported}."
        ) from None


def make(env_id: str | None = None, *, render_mode: str | None = None) -> gym.Env:
    """Create an environment, defaulting to this project's primary env."""
    return gym.make(env_id or DEFAULT_ENV_ID, render_mode=render_mode)
