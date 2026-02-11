import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
import torch.nn as nn
from torch.distributions import Categorical
from datetime import datetime

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# ==================== Environment Definition ====================
class StockTradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, initial_balance=1e6,
                 max_daily_loss=1e4, max_daily_trades=100):
        super(StockTradingEnv, self).__init__()

        self.price_df = data
        self.trading_day_groups = self.price_df.groupby(['trading_day'])
        self.current_day = 0
        self.t_idx = 0


        self.initial_balance = initial_balance
        self.max_daily_loss = max_daily_loss
        self.max_daily_trades = max_daily_trades

        # Actions: 0 = hold, 1 = buy 1 share, 2 = sell 1 share
        self.action_space = spaces.Discrete(3)
        # Observations: [price, holdings, cash, pnl, trades_today]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # self._reset_session()

    def _update_t(self):
        reset_t = False
        self.t_idx += 1

        rth_mask = self.curr_td_df['ovn'] == 0
        if self.t_idx >= rth_mask[rth_mask].index[-1]:
            self.current_day += 1
            self.t_idx = 0
            reset_t = True

        if self.current_day >= len(self.trading_day_groups):
            self.current_day = 0
            self.t_idx = 0
            reset_t = True

        return reset_t

    def _reset_session(self):

        # print(self.current_day, self.t_idx)

        self.curr_td_df = self.trading_day_groups.get_group((self.current_day,))

        if self.t_idx == 0:
            rth_mask = self.curr_td_df['ovn'] == 0
            self.t_idx = rth_mask[rth_mask].index[0]
    
        self.balance = self.initial_balance
        self.holdings = 0

        self.daily_pnl = 0
        self.trades_today = 0
        self.done = False

    def reset(self):
        self._reset_session()
        return self._get_obs()

    def _get_price(self):
        return self.curr_td_df.loc[self.t_idx]['Close']

    def _get_obs(self):

        prices = self.curr_td_df.loc[self.t_idx-5:self.t_idx]['Close']
        prices_diff = prices - prices.shift(1)
        prices_diff_feat = prices_diff[1:].apply(sigmoid)

        pnl_scaled = np.tanh(self.daily_pnl / self.max_daily_loss)

        # trades_scaled = self.trades_today / self.max_daily_trades
        # trades_feat = 1 / (1 + np.exp(-trades_scaled))
        trades_feat = self.trades_today > 1

        obs = np.array(prices_diff_feat.to_list() + [self.holdings, pnl_scaled, trades_feat], dtype=np.float32)
        
        return obs

    def step(self, action):
        price = self._get_price()
        reward = 0.0

        completed_trade = False

        # if self.trades_today >= self.max_daily_trades:
        #     action = 0

        if action != 0 and self.holdings == 0:
            self.entry_price = price
            reward -= 0.05
            self.balance -= 0.05

        # Execute action
        if action == 1 and self.holdings < 1:  # buy
            self.holdings += 1
            self.balance -= price
            # print(f"Buy: {price}")
        elif action == 2 and self.holdings > -1:  # sell
            self.holdings -= 1
            self.balance += price
            # print(f"Sell: {price}")

        if action != 0 and self.holdings == 0:
            self.exit_price = price
            completed_trade = True
            self.trades_today += 1

            if action == 1:
                price_diff = self.exit_price - self.entry_price
            elif action == 2:
                price_diff = self.entry_price - self.exit_price
            reward = price_diff
            
        # Compute PnL
        mark_to_market = self.holdings * price + self.balance
        self.daily_pnl = mark_to_market - self.initial_balance
        # reward = mark_to_market - (self.initial_balance + self.daily_pnl)

        # print(self.holdings, self.balance, mark_to_market, self.daily_pnl, price, reward)
        # if completed_trade:
        #     print(f"Enter: {self.entry_price} Exit: {self.exit_price} | Profit: {reward} PnL: {self.daily_pnl}")

        # Constraint violation indicators
        loss_violation = max(0, -self.daily_pnl - self.max_daily_loss)
        # trade_violation = max(0, self.trades_today - self.max_daily_trades)
        trade_violation = self.trades_today > 0


        reset_t = self._update_t()

        hit_daily_loss = self.daily_pnl < -self.max_daily_loss
        # hit_trades_limit = self.trades_today >= self.max_daily_trades
        self.done = reset_t or hit_daily_loss

        if self.done:
            print(f"Trades: {self.trades_today} | PnL: {self.daily_pnl} , reset: {reset_t} , hit_loss: {hit_daily_loss}")

        obs = self._get_obs()
        info = {
            'loss_violation': loss_violation,
            'trade_violation': trade_violation
        }

        return obs, reward, self.done, info

# ==================== PPO Agent with Lagrangian ====================
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)

        self.critic = nn.Linear(hidden_dim, 1)

        self.loss_value_head = nn.Linear(hidden_dim, 1)

        self.trade_value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        hidden = self.shared(x)
        return self.actor(hidden), self.critic(hidden), self.loss_value_head(hidden), self.trade_value_head(hidden)

class PPOLagTrader:
    def __init__(self, env, lr=3e-4, gamma=0.99, clip_epsilon=0.2,
                 lagr_lr=1e-3, target_loss=1e4, target_trades=100):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        self.policy = ActorCritic(obs_dim, 16, act_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_params = list(self.policy.actor.parameters()) + list(self.policy.shared.parameters())
        self.critic_params = list(self.policy.shared.parameters()) + list(self.policy.trade_value_head.parameters())

        self.policy_optimizer = torch.optim.Adam(self.policy_params, lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=1e-2)


        # Lagrangian multipliers
        self.lagr_loss = torch.tensor(1.0, requires_grad=True)
        self.lagr_trades = torch.tensor(1.0, requires_grad=True)
        self.lagr_optimizer = torch.optim.Adam([self.lagr_loss, self.lagr_trades], lr=lagr_lr)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.target_loss = target_loss
        self.target_trades = target_trades

    def collect_trajectory(self, timesteps=2048):
        obs = self.env.reset()
        storage = []
        for _ in range(timesteps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                logits, value, _, _ = self.policy(obs_tensor)
            distrib = Categorical(logits=logits)
            action = distrib.sample().item()

            # action_probs = distrib.probs.detach().cpu().numpy()
            # print(f"{action_probs} | {action}")

            next_obs, reward, done, info = self.env.step(action)

            storage.append((obs, action, reward, value.item(), distrib.log_prob(torch.tensor(action)),
                            info['loss_violation'], info['trade_violation']))
            obs = next_obs
            if done:
                obs = self.env.reset()
        return storage

    def update(self, storage):
        # Convert storage to tensors
        obs = torch.FloatTensor([s[0] for s in storage])
        actions = torch.LongTensor([s[1] for s in storage])
        rewards = [s[2] for s in storage]
        old_values = torch.FloatTensor([s[3] for s in storage])
        old_logprobs = torch.stack([s[4] for s in storage])
        losses_vio = torch.FloatTensor([s[5] for s in storage])
        trades_vio = torch.FloatTensor([s[6] for s in storage])

        # Compute returns and advantages
        returns = []
        discounted = 0
        for r in reversed(rewards):
            discounted = r + self.gamma * discounted
            returns.insert(0, discounted)
        returns = torch.FloatTensor(returns)
        advantages = returns - old_values


        # PPO surrogate loss
        logits, values, loss_values, trade_values = self.policy(obs)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        ratios = torch.exp(logprobs - old_logprobs)
        # surr1 = ratios * advantages
        # surr2 = torch.clamp(ratios, 1 - self.clip_epsilon,
        #                     1 + self.clip_epsilon) * advantages
        # actor_loss = -torch.min(surr1, surr2).mean()
        # critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()

        actor_loss = torch.tensor(0.0)
        critic_loss = torch.tensor(0.0)


        ratio_clipped = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        # Entropy regularization
        # entropy = dist.entropy().mean()
        # entropy_coeff = 0.01  # You can adjust this value
        # actor_loss -= entropy_coeff * entropy

        # Lagrangian constraint term
        # lagr_term = (self.lagr_loss * (ratio_clipped * losses_vio).mean() +
        #              self.lagr_trades * (ratio_clipped * trades_vio).mean())

        trade_returns = []
        discounted_t = 0.0

        trades_buf = [s[6] for s in storage]
        for t in trades_buf[::-1]:
            discounted_t = t + self.gamma * discounted_t
            trade_returns.insert(0, discounted_t)
        trade_returns = torch.tensor(trade_returns)

        trade_adv = trade_returns - trade_values.detach().squeeze()

        adv = (trade_adv - trade_adv.mean()) / (trade_adv.std() + 1e-8)

        print("  trade_adv ▶ mean:{:.3f}, std:{:.3f}, %>0:{:.1%}".format(
            trade_adv.mean(), trade_adv.std(), (trade_adv>0).float().mean()
        ))

        for i in range(5):
            print(f"step {i:2d} | return={trade_returns[i]:.2f} | "
                f"value={trade_values[i].item():.2f} | adv={trade_adv[i].item():.2f}")
            
        
        normalized_returns = (trade_returns - trade_returns.mean())/(trade_returns.std()+1e-8)
        value_loss = nn.MSELoss()(trade_values.squeeze(), normalized_returns)
        # lagr_term = self.lagr_trades * (trades_vio).mean()


        lagr_term = self.lagr_trades * (ratios * adv).mean()

        # total_loss = actor_loss + critic_loss + lagr_term
        total_loss = lagr_term

        print(f"lagr_term: {lagr_term.item()}, value_loss: {value_loss.item()}")


        self.policy_optimizer.zero_grad()

        # Update policy
        total_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.policy_params, max_norm=1.0)

        self.policy_optimizer.step()

        for _ in range(50):
            self.critic_optimizer.zero_grad()
            _, _, _, trade_values2 = self.policy(obs)
            value_loss2 = nn.MSELoss()(trade_values2.squeeze(), normalized_returns)
            value_loss2.backward()                     # no need to retain graph
            torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=1.0)
            self.critic_optimizer.step()


        # Dual ascent on multipliers
        self.lagr_optimizer.zero_grad()
        # Want to maximize over multipliers => minimize negative
        
        # dual_loss = -lagr_term

        new_lagr_term = self.lagr_trades * (trades_vio).mean()
        dual_loss = -new_lagr_term
        dual_loss.backward()
        self.lagr_optimizer.step()

        # Ensure multipliers stay >= 0
        with torch.no_grad():
            self.lagr_loss.clamp_(min=0)
            self.lagr_trades.clamp_(min=0)

        return actor_loss.item(), critic_loss.item(), self.lagr_loss.item(), self.lagr_trades.item()

    def train(self, epochs=100, timesteps_per_epoch=2048):
        for epoch in range(1, epochs+1):
            storage = self.collect_trajectory(timesteps_per_epoch)
            a_loss, c_loss, lam_loss, lam_trade = self.update(storage)
            # Calculate average reward
            avg_reward = np.mean([s[2] for s in storage if s[2] > 0])
            print(f"Epoch {epoch}: Actor Loss={a_loss:.3f}, Critic Loss={c_loss:.3f}, "
                  f"\u03BB_loss={lam_loss:.3f}, \u03BB_trade={lam_trade:.3f}, "
                  f"Average Reward={avg_reward:.3f}")
            
            # exit()

# ==================== Usage Example ====================
if __name__ == '__main__':

    data_filename = 'test_seconds_td0.csv'
    price_data_df = pd.read_csv(data_filename)

    dt_format_str = "%m/%d/%Y %H:%M:%S"
    price_data_df['dt'] = price_data_df.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", dt_format_str), axis=1)

    env = StockTradingEnv(price_data_df,
                          initial_balance=1e6,
                          max_daily_loss=10,
                          max_daily_trades=10)
    agent = PPOLagTrader(env)
    agent.train(epochs=50, timesteps_per_epoch=7200)
