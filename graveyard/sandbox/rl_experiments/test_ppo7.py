import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    
class ExecutionEnv(gym.Env):
    """
    A simple trade-execution environment where the agent manages position size
    using a discrete set of actions under noisy price forecasts.
    """
    def __init__(self, price_series=None, sigma_noise=0.02, lambda_risk=0.1, max_steps=1000):
        super(ExecutionEnv, self).__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.lambda_risk = lambda_risk
        self.sigma_noise = sigma_noise

        # Price series
        if price_series is None:
            self.price_series = self._generate_price_series()
        else:
            self.price_series = price_series
        self.n_steps = len(self.price_series) - 1

        # Discrete actions: map to positions
        self.positions = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.positions))

        # Observation: [predicted_return, predicted_vol, current_position, drawdown]
        lows = np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32)
        highs = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=lows, high=highs, shape=(4,), dtype=np.float32
        )

        self.reset()

    def _generate_price_series(self, mu=0.0, sigma=0.01, S0=100, T=1.0, dt=0.001):
        # np.random.seed(42)
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
        true_ret = np.log(self.price_series[self.current_step+1] / self.price_series[self.current_step])
        # pred_ret = true_ret + np.random.normal(0, self.sigma_noise)
        # pred_vol = abs(np.random.normal(self.sigma_noise, 0.005))
        # pred_ret = true_ret
        pred_vol = 0.00

        horizon = 600
        future_idx = min(self.current_step + horizon, len(self.price_series)-1)
        trueH = self.price_series[self.current_step+1] - self.price_series[self.current_step]
        pred_ret = trueH + np.random.normal(0, 0.0001)

        trueH2 = self.price_series[future_idx] - self.price_series[self.current_step]

        # store for logging later
        self._last_true_ret = true_ret
        self._last_pred_ret = pred_ret
        return np.array([pred_ret, pred_vol, self.position, self.drawdown], dtype=np.float32)

    def step(self, action):
        # map action to position
        self.position = self.positions[action]
        prev_price = self.price_series[self.current_step]
        new_price = self.price_series[self.current_step + 1]
        ret = np.log(new_price / prev_price)

        # PnL and portfolio update
        pnl = self.position * ret
        self.portfolio_value *= np.exp(pnl)

        # drawdown
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # risk penalty
        risk_penalty = self.lambda_risk * (self.position**2) * abs(ret)
        # reward = pnl - risk_penalty
        reward = pnl

        self.current_step += 1
        done = self.current_step >= min(self.n_steps, self.max_steps)

        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape)
        info = {
            'true_ret': self._last_true_ret,
            'pred_ret': self._last_pred_ret,
            'action': self.position,
            'pnl': pnl,
            'risk_penalty': risk_penalty,
            'reward': reward,
            'portfolio_value': self.portfolio_value,
        }
        return obs, reward, done, info

    def render(self, mode='human'):
        print(f"Step {self.current_step}: Pos={self.position}, Value={self.portfolio_value:.4f}, DD={self.drawdown:.4f}")

class DiagnosticCallback(BaseCallback):
    """
    Collects debug info and plots diagnostics for first epochs.
    """
    def __init__(self, n_epochs=3, verbose=0):
        super(DiagnosticCallback, self).__init__(verbose)
        self.n_epochs = n_epochs
        self.epoch = 0
        self.buffer = []

    def _on_rollout_start(self):
        if self.epoch < self.n_epochs:
            self.buffer = []

    def _on_step(self) -> bool:
        if self.epoch < self.n_epochs:
            infos = self.locals.get('infos', [])
            actions = self.locals.get('actions', [])
            for info, a in zip(infos, actions):
                self.buffer.append({
                    'true_ret': info['true_ret'],
                    'pred_ret': info['pred_ret'],
                    'action': a,
                    'pnl': info['pnl'],
                    'risk_penalty': info['risk_penalty'],
                    'reward': info['reward'],
                })
        return True

    def _on_rollout_end(self):
        if self.epoch == self.n_epochs - 1:
            data = self.buffer
            true_rets = np.array([d['true_ret'] for d in data])
            pred_rets = np.array([d['pred_ret'] for d in data])
            actions = np.array([d['action'] for d in data])
            pnls = np.array([d['pnl'] for d in data])
            penalties = np.array([d['risk_penalty'] for d in data])

            # 1. Scatter: pred_ret vs action
            plt.scatter(pred_rets, actions, alpha=0.3)
            plt.title(f'Epoch {self.epoch+1}: Action vs. Predicted Return')
            plt.xlabel('Predicted Return')
            plt.ylabel('Position')
            plt.show()

            # 2. Time series: pnl vs penalty
            plt.plot(pnls, label='PnL')
            plt.plot(penalties, label='Risk Penalty')
            plt.title(f'Epoch {self.epoch+1}: PnL and Risk Penalty')
            plt.legend(); plt.show()

            # 3. Hist: true vs pred returns
            plt.hist(true_rets, bins=50, alpha=0.5, label='True')
            plt.hist(pred_rets, bins=50, alpha=0.5, label='Pred')
            plt.title(f'Epoch {self.epoch+1}: Return Distributions')
            plt.legend(); plt.show()

            # 4. Actions by signal quintile
            quintiles = np.percentile(pred_rets, [0,20,40,60,80,100])
            for i in range(5):
                mask = (pred_rets >= quintiles[i]) & (pred_rets < quintiles[i+1])
                plt.hist(actions[mask], bins=len(self.training_env.get_attr('positions')[0]), alpha=0.5, label=f'Q{i+1}')
            plt.title(f'Epoch {self.epoch+1}: Actions by Predicted Return Quintile')
            plt.xlabel('Position'); plt.legend(); plt.show()

        self.epoch += 1

if __name__ == "__main__":
    env = ExecutionEnv()
    eval_env = ExecutionEnv()
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=500,
                                 deterministic=True, render=False)
    diag_cb = DiagnosticCallback(n_epochs=50)

    vec_env = DummyVecEnv([lambda: ExecutionEnv()])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)


    model = PPO('MlpPolicy', vec_env, verbose=1, ent_coef=0.1, n_steps=1000)
    model.learn(total_timesteps=50000, callback=[diag_cb])

    # Post-training evaluation
    obs = env.reset()
    done = False
    rewards, positions, portfolio_values = [], [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print(f"Action: {action} obs: {obs}")
        obs, reward, done, info = env.step(action)
        positions.append(info['action'])
        rewards.append(reward)
        portfolio_values.append(info['portfolio_value'])

    # Plot evaluation results
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    axes[0].plot(positions);
    axes[0].set_title('Eval: Position over Time'); axes[0].set_ylabel('Position')
    axes[1].plot(portfolio_values);
    axes[1].set_title('Eval: Portfolio Value over Time'); axes[1].set_ylabel('Value')
    axes[2].plot(rewards);
    axes[2].set_title('Eval: Reward over Time'); axes[2].set_ylabel('Reward'); axes[2].set_xlabel('Time Step')
    fig.tight_layout()
    plt.show()
