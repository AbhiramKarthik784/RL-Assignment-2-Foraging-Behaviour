import numpy as np
import matplotlib.pyplot as plt
import os

SEED = 42
EPISODES = 1200
STEPS = 60
ALPHA = 0.15
GAMMA = 0.95
EPSILON = 0.20
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.995

class ForagingEnv:
    """Two-patch foraging environment with food depletion and regeneration."""
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.max_food = np.array([10, 8], dtype=int)
        self.regen = np.array([1, 1], dtype=int)
        self.move_cost = 0.5
        self.reset()

    def reset(self):
        self.food = self.max_food.copy()
        self.location = 0
        return self._state()

    def _state(self):
        # Discretized state: current patch (0/1), food in patch A (0..10), food in patch B (0..8)
        return (int(self.location), int(self.food[0]), int(self.food[1]))

    def step(self, action):
        # 0 = forage at current patch; 1 = move to the other patch
        reward = 0.0
        if action == 0:
            available = self.food[self.location]
            collected = min(2, int(available))
            self.food[self.location] -= collected
            reward = float(collected)
        else:
            self.location = 1 - self.location
            reward = -self.move_cost

        # Food regenerates gradually after the action
        self.food = np.minimum(self.max_food, self.food + self.regen)
        return self._state(), reward

def q_learning_train(seed=SEED):
    env = ForagingEnv(seed)
    q = np.zeros((2, 11, 9, 2), dtype=float)
    epsilon = EPSILON
    episode_rewards = []

    for ep in range(EPISODES):
        state = env.reset()
        total = 0.0
        for _ in range(STEPS):
            s = state
            if np.random.random() < epsilon:
                action = np.random.randint(2)
            else:
                action = int(np.argmax(q[s]))
            next_state, reward = env.step(action)
            target = reward + GAMMA * np.max(q[next_state])
            q[s + (action,)] += ALPHA * (target - q[s + (action,)])
            state = next_state
            total += reward
        episode_rewards.append(total)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    return q, np.array(episode_rewards)

def run_policy(policy, episodes=200, seed=1000):
    env = ForagingEnv(seed)
    scores = []
    patch_visits = np.zeros(2, dtype=int)
    for ep in range(episodes):
        state = env.reset()
        total = 0.0
        for t in range(STEPS):
            action = policy(state, env, t)
            state, reward = env.step(action)
            total += reward
            patch_visits[env.location] += 1
        scores.append(total)
    return np.array(scores), patch_visits

def main():
    np.random.seed(SEED)
    os.makedirs("results", exist_ok=True)
    q, training_rewards = q_learning_train()

    def learned_policy(state, env, t):
        return int(np.argmax(q[state]))

    def random_policy(state, env, t):
        return int(np.random.randint(2))

    def heuristic_policy(state, env, t):
        # Stay when the current patch has food; move if it is nearly depleted
        current_food = state[1 + state[0]]
        other_food = state[1 + (1 - state[0])]
        if current_food <= 2 and other_food >= current_food + 2:
            return 1
        return 0

    q_scores, q_visits = run_policy(learned_policy, seed=2000)
    random_scores, random_visits = run_policy(random_policy, seed=3000)
    heuristic_scores, heuristic_visits = run_policy(heuristic_policy, seed=4000)

    # Learning curve: moving average over 40 episodes
    window = 40
    moving = np.convolve(training_rewards, np.ones(window)/window, mode="valid")
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(window, EPISODES + 1), moving, linewidth=1.8)
    plt.title("Q-learning Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total reward per episode (40-episode average)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/learning_curve.png", dpi=180)
    plt.close()

    means = [q_scores.mean(), random_scores.mean(), heuristic_scores.mean()]
    stds = [q_scores.std(), random_scores.std(), heuristic_scores.std()]
    labels = ["Q-learning", "Random", "Heuristic"]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, means, yerr=stds, capsize=5)
    plt.title("Average Evaluation Reward by Strategy")
    plt.ylabel("Mean cumulative reward over 60 steps")
    plt.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, means):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{v:.1f}", ha="center")
    plt.tight_layout()
    plt.savefig("results/strategy_comparison.png", dpi=180)
    plt.close()

    visits = [q_visits, random_visits, heuristic_visits]
    x = np.arange(3)
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x-width/2, [v[0] for v in visits], width, label="Patch A")
    plt.bar(x+width/2, [v[1] for v in visits], width, label="Patch B")
    plt.xticks(x, labels)
    plt.ylabel("Number of time-step visits (200 episodes)")
    plt.title("Food Patch Visit Distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/patch_visits.png", dpi=180)
    plt.close()

    with open("results/metrics.txt", "w") as f:
        f.write("Evaluation results (200 episodes, 60 steps each)\\n")
        for label, scores in zip(labels, [q_scores, random_scores, heuristic_scores]):
            f.write(f"{label}: mean={scores.mean():.3f}, std={scores.std():.3f}\\n")
        f.write(f"Q-learning final training moving average: {moving[-1]:.3f}\\n")

    print("Evaluation results (mean ± standard deviation):")
    for label, scores in zip(labels, [q_scores, random_scores, heuristic_scores]):
        print(f"{label:12s}: {scores.mean():.2f} ± {scores.std():.2f}")
    print(f"Final 40-episode training average: {moving[-1]:.2f}")
    print("Graphs saved in results/")

if __name__ == "__main__":
    main()
