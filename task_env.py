import jax.numpy as jnp
from craftax.craftax.constants import Achievement
from craftax.craftax.envs.craftax_symbolic_env import (
    CraftaxSymbolicEnv,
    CraftaxSymbolicEnvNoAutoReset,
)


def _make(BaseEnv):
    class _Env(BaseEnv):
        def __init__(self, target_achievement: str,
                     terminate_on_complete: bool = False):
            super().__init__()
            try:
                self.target_idx = Achievement[target_achievement.upper()].value
            except KeyError:
                valid = sorted(a.name.lower() for a in Achievement)
                raise ValueError(
                    f"Unknown achievement {target_achievement!r}. Valid: {valid}"
                )
            self.target_name = target_achievement.lower()
            self.terminate_on_complete = bool(terminate_on_complete)

        def step_env(self, key, state, action, params):
            prev = state.achievements[self.target_idx].astype(jnp.bool_)
            obs, next_state, _orig_reward, done, info = super().step_env(
                key, state, action, params,
            )
            now = next_state.achievements[self.target_idx].astype(jnp.bool_)
            hit = now & ~prev
            reward = hit.astype(jnp.float32)
            done = done | (jnp.bool_(self.terminate_on_complete) & now)

            info = dict(info)
            info["task/reward"] = reward
            info["task/completed"] = now.astype(jnp.float32)

            return obs, next_state, reward, done, info

    _Env.__name__ = f"{BaseEnv.__name__}TaskEnv"
    _Env.__qualname__ = _Env.__name__
    return _Env


CraftaxSymbolicTaskEnv = _make(CraftaxSymbolicEnv)
CraftaxSymbolicTaskEnvNoAutoReset = _make(CraftaxSymbolicEnvNoAutoReset)
