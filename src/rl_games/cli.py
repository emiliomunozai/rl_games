import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np

from rl_games import envs

ENV_ID = envs.DEFAULT_ENV_ID
SAVE_DIR = Path("saves")
AGENT_CHOICES = ("qlearning", "dqn")
VERSION = version("rl_games")


def _save_path(agent_type: str) -> Path:
    if agent_type == "qlearning":
        return SAVE_DIR / "qlearning_lunar.pkl"
    return SAVE_DIR / "dqn_lunar.pt"


def _load_agent(agent_type: str):
    path = _save_path(agent_type)
    if agent_type == "qlearning":
        from rl_games.agents.qlearning import QLearningAgent
        return QLearningAgent.load(path)
    from rl_games.agents.dqn import DQNAgent
    return DQNAgent.load(path)


def _run_episodes(agent, env, n_episodes: int) -> list[float]:
    """Run n_episodes greedily and return each episode's total reward."""
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        returns.append(total)
    return returns


# ── commands ─────────────────────────────────────────────────────────


def cmd_inspect(args: argparse.Namespace) -> None:
    env_id = args.env or ENV_ID
    env = envs.make(env_id)

    print(f"Environment: {env_id}\n")
    print(f"Observation space : {env.observation_space}")
    print(f"  shape           : {env.observation_space.shape}")
    if hasattr(env.observation_space, "low"):
        print(f"  low             : {env.observation_space.low}")
        print(f"  high            : {env.observation_space.high}")
    print(f"\nAction space      : {env.action_space}")
    if hasattr(env.action_space, "n"):
        print(f"  n actions       : {env.action_space.n}")
    print(f"Max episode steps : {env.spec.max_episode_steps if env.spec else 'N/A'}")

    n = args.steps
    print(f"\n-- Sample transitions ({n} steps, random policy) --\n")
    obs, info = env.reset()
    print(f"  Initial state: {np.array2string(obs, precision=3)}")
    print()

    for step in range(1, n + 1):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(
            f"  step {step:>3} | action={action} | "
            f"reward={reward:+.3f} | done={done}"
        )
        print(f"           state -> {np.array2string(next_obs, precision=3)}")
        obs = next_obs
        if done:
            print("           [episode ended, resetting]")
            obs, info = env.reset()
            print(f"           state -> {np.array2string(obs, precision=3)}")
        print()

    env.close()


def cmd_init(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)

    if path.exists():
        print(f"Save already exists at {path}. Run 'rlgames delete {args.agent}' first.")
        return

    if args.agent == "qlearning":
        from rl_games.agents.qlearning import QLearningAgent
        agent = QLearningAgent(ENV_ID)
    else:
        from rl_games.agents.dqn import DQNAgent
        agent = DQNAgent(ENV_ID)

    agent.save(path)
    print(f"Initialized {args.agent} agent.")


def cmd_train(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)

    if args.agent == "qlearning":
        from rl_games.agents.qlearning import QLearningAgent
        agent = QLearningAgent.load(path) if path.exists() else QLearningAgent(ENV_ID)
        agent.train(total_episodes=args.episodes)
        agent.save(path)

    else:
        from rl_games.agents.dqn import DQNAgent
        agent = DQNAgent.load(path) if path.exists() else DQNAgent(ENV_ID)
        agent.train(total_episodes=args.episodes)
        agent.save(path)

    print("Training complete.")


def cmd_delete(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)
    if path.exists():
        path.unlink()
        print(f"Deleted {path}")
    else:
        print(f"No save found at {path}")


def cmd_load(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)
    if not path.exists():
        print(f"No save found at {path}")
        return

    agent = _load_agent(args.agent)
    print(agent.info())

    if args.eval:
        print("\nEvaluating (10 episodes) ...")
        env = envs.make(ENV_ID)
        rewards = _run_episodes(agent, env, 10)
        env.close()
        print(f"  Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


def cmd_sim(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)
    if not path.exists():
        print(f"No save found at {path}")
        return

    agent = _load_agent(args.agent)
    env = envs.make(ENV_ID)

    all_rewards: list[float] = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"== Episode {ep}/{args.episodes} ==\n")
        print(f"  initial state: {np.array2string(obs, precision=3)}\n")

        limit = args.steps  # None means show all

        while not done:
            step += 1
            action, _ = agent.predict(obs, deterministic=True)
            action = int(action)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if limit is None or step <= limit:
                print(
                    f"  step {step:>4} | action={envs.spec_for(ENV_ID).format_action(action):>18} | "
                    f"reward={reward:+8.3f} | total={total_reward:+9.2f}"
                )
                if args.verbose:
                    print(f"           state -> {np.array2string(next_obs, precision=3)}")

            obs = next_obs

        if limit is not None and step > limit:
            print(f"  ... ({step - limit} more steps) ...")

        outcome = envs.spec_for(ENV_ID).decode_outcome(
            terminated, truncated, total_reward
        )

        print(f"\n  Result: {outcome} | Steps: {step} | Total reward: {total_reward:+.2f}\n")
        all_rewards.append(total_reward)

    env.close()

    if len(all_rewards) > 1:
        print(
            f"Summary over {len(all_rewards)} episodes: "
            f"mean={np.mean(all_rewards):+.2f} +/- {np.std(all_rewards):.2f}"
        )


def cmd_render(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)
    if not path.exists():
        print(f"No save found at {path}")
        return

    agent = _load_agent(args.agent)
    env = envs.make(ENV_ID, render_mode="human")

    for ep, total_reward in enumerate(_run_episodes(agent, env, args.episodes), 1):
        print(f"Episode {ep}/{args.episodes} | Reward: {total_reward:.2f}")

    env.close()


def cmd_version(_args: argparse.Namespace) -> None:
    print(f"rl_games {VERSION}")


def cmd_list(_args: argparse.Namespace) -> None:
    print("Available agents:\n")
    for agent in AGENT_CHOICES:
        path = _save_path(agent)
        status = "saved" if path.exists() else "no save"
        print(f"  {agent:<14} [{status}]  {path}")


# ── argument parser ──────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlgames",
        description="Train and evaluate RL agents on LunarLander-v3",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # version
    p = sub.add_parser("version", help="Show the package version")
    p.set_defaults(func=cmd_version)

    # list
    p = sub.add_parser("list", help="List available agents and their save status")
    p.set_defaults(func=cmd_list)

    # inspect
    p = sub.add_parser(
        "inspect",
        help="Inspect an environment: show state/action spaces and sample transitions",
    )
    p.add_argument("--env", type=str, default=None, help=f"Gymnasium env ID (default: {ENV_ID})")
    p.add_argument("--steps", type=int, default=5, help="Random steps to sample (default: 5)")
    p.set_defaults(func=cmd_inspect)

    # init
    p = sub.add_parser("init", help="Initialize a new (untrained) agent and save it")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.set_defaults(func=cmd_init)

    # train
    p = sub.add_parser("train", help="Train an agent and save the result")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.add_argument("--episodes", type=int, default=10_000, help="Training episodes (default: 10k)")
    p.set_defaults(func=cmd_train)

    # delete
    p = sub.add_parser("delete", help="Delete a saved agent")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.set_defaults(func=cmd_delete)

    # load
    p = sub.add_parser("load", help="Load a saved agent and display info")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.add_argument("--eval", action="store_true", help="Run a quick 10-episode evaluation")
    p.set_defaults(func=cmd_load)

    # sim
    p = sub.add_parser("sim", help="Simulate episodes with a trained agent (text output)")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to simulate (default: 1)")
    p.add_argument("--steps", type=int, default=None, help="Limit output to the first N steps per episode (default: show all)")
    p.add_argument("--verbose", action="store_true", help="Print every step with full state vectors")
    p.set_defaults(func=cmd_sim)

    # render
    p = sub.add_parser("render", help="Render episodes using a saved agent (graphical window)")
    p.add_argument("agent", choices=AGENT_CHOICES)
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to render (default: 1)")
    p.set_defaults(func=cmd_render)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)
