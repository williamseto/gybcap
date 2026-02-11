import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import EvalCallback

class ExecutionEnv(gym.Env):
    """
    Trade-execution environment with discrete position actions and noisy multi-step forecasts.
    Actions map to fixed position sizes from -1 to +1.
    """
    def __init__(self, price_series=None, sigma_noise=0.02, lambda_risk=0.1, max_steps=1000):
        super().__init__()
        self.max_steps = max_steps
        self.lambda_risk = lambda_risk
        self.sigma_noise = sigma_noise

        # Price series generation
        if price_series is None:
            self.price_series = self._generate_price_series()
        else:
            self.price_series = price_series
        self.n_steps = len(self.price_series) - 1

        # Discrete action: five positions
        self.positions = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.positions))

        # Observations: [pred_horizon_ret, ma_diff, vol_est, current_pos, drawdown]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.reset()

    def _generate_price_series(self, mu=0.0, sigma=0.01, S0=100, T=1.0, dt=0.001):
        steps = int(T / dt)
        prices = [S0]
        for _ in range(steps):
            ret = np.random.normal(mu * dt, sigma * np.sqrt(dt))
            prices.append(prices[-1] * np.exp(ret))
        return np.array(prices)

    def reset(self):
        self.current_step = 0
        self.position = 0.0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        self.drawdown = 0.0
        return self._get_obs()

    def _get_obs(self):
        t = self.current_step
        # Multi-step horizon return (signal)
        H = 50
        future_idx = min(t + H, len(self.price_series) - 1)
        # trueH = np.log(self.price_series[future_idx] / self.price_series[t])
        trueH = (self.price_series[future_idx] - self.price_series[t])
        predH = trueH + np.random.normal(0, self.sigma_noise)

        # MA crossover feature
        if t >= 20:
            ma_short = np.mean(self.price_series[t-5:t])
            ma_long  = np.mean(self.price_series[t-20:t])
            ma_diff  = (ma_short - ma_long) / ma_long
            vol_est  = np.std(np.diff(np.log(self.price_series[t-20:t])))
        else:
            ma_diff = 0.0
            vol_est = 0.0

        # store for reward and diagnostics
        self._last_trueH = trueH
        self._last_predH = predH

        return np.array([predH, ma_diff, vol_est, self.position, self.drawdown], dtype=np.float32)

    def step(self, action):
        # Map discrete action to position
        self.position = self.positions[int(action)]

        t = self.current_step
        prev_price = self.price_series[t]
        new_price  = self.price_series[t + 1]
        ret = np.log(new_price / prev_price)

        # Compute PnL
        pnl = self.position * ret
        self.portfolio_value *= np.exp(pnl)

        # Update drawdown
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.drawdown   = (self.peak_value - self.portfolio_value) / self.peak_value

        # Risk penalty
        risk_penalty = self.lambda_risk * (self.position**2) * abs(ret)
        reward = pnl - risk_penalty

        self.current_step += 1
        done = self.current_step >= min(self.n_steps, self.max_steps)

        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape)
        info = {
            'true_horizon_ret': self._last_trueH,
            'pred_horizon_ret': self._last_predH,
            'pnl': pnl,
            'risk_penalty': risk_penalty,
            'portfolio_value': self.portfolio_value
        }
        return obs, reward, done, info

    def render(self, mode='human'):
        print(f"Step {self.current_step}: Pos={self.position}, Value={self.portfolio_value:.4f}, DD={self.drawdown:.4f}")

if __name__ == "__main__":
    # Instantiate environments
    env = ExecutionEnv()
    eval_env = ExecutionEnv()

    # Evaluation callback
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=True
    )

    # Create QR-DQN (distributional DQN) agent
    model = QRDQN(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        exploration_fraction=0.5,
    )

    # Train the agent
    model.learn(total_timesteps=50000)
    # model.save('qrdqn_execution_agent')

    # Post-training evaluation
    obs = env.reset()
    done = False
    positions, rewards, portfolio_vals = [], [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        positions.append(env.position)
        rewards.append(reward)
        portfolio_vals.append(info['portfolio_value'])

    # Plot evaluation results
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    axes[0].plot(positions)
    axes[0].set_title('Position Over Time')
    axes[1].plot(portfolio_vals)
    axes[1].set_title('Portfolio Value Over Time')
    axes[2].plot(rewards)
    axes[2].set_title('Reward Over Time')
    axes[2].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()
