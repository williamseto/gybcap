import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import pandas as pd
import os
from datetime import datetime, timedelta
import talib
import stats_util

# --- 1. Environment Definition ---
class TradingEnv:
    def __init__(self, price_df, P_max, L_max, N_max):
        self.price_df = price_df       # tick data series
        self.P_max = P_max         # max position units
        self.L_max = L_max         # daily stop-loss threshold
        self.N_max = N_max         # max trades per day

        self.trading_day_groups = price_df.groupby(['trading_day'])
        self.current_day = 0
        self.t_idx = 0


    def _init_day(self):

        # initialize or advance to start of next day
        self.curr_td_df = self.trading_day_groups.get_group((self.current_day,))
        rth_mask = self.curr_td_df['ovn'] == 0

        self.t_idx = rth_mask[rth_mask].index[0]

        ovn_df = self.curr_td_df[~rth_mask & (self.curr_td_df.index < self.t_idx)]

        self.daily_stats = {}
        self.daily_stats['ovn_lo'] = ovn_df['Close'].min()
        self.daily_stats['ovn_hi'] = ovn_df['Close'].max()

        self.daily_stats.update(stats_util.calc_ib_stats(self.curr_td_df))

        self.minute_df = self.curr_td_df.resample('1Min', on='dt').agg({'Close':'ohlc', 'Volume':'sum'}).bfill()

        self.minute_df["sma5"] = talib.SMA(self.minute_df["Close"]['close'], timeperiod=5)
        self.minute_df["sma20"] = talib.SMA(self.minute_df["Close"]['close'], timeperiod=20)

        self.curr_rth_lo = 10000.0
        self.curr_rth_hi = -10000.0

        self.curr_ib_lo = 10000.0
        self.curr_ib_hi = -10000.0
        
        self.position = 0
        self.pnl_day = 0.0
        self.trade_count = 0

    def reset(self):

        self.current_day += 1

        try:
            self._init_day()
        except:
            self.current_day = 1
            self._init_day()

        return self._get_state()
    
    def _get_price(self):
        return self.curr_td_df.loc[self.t_idx]['Close']

    def _get_state(self):
        window = max(1, self.t_idx-20)
        segment = self.curr_td_df.loc[window:self.t_idx+1]['Close']

        returns = np.diff(segment) / segment[:-1] if len(segment)>1 else [0.0]

        curr_dt = self.curr_td_df.loc[self.t_idx]['dt']
        curr_price = self._get_price()

        self.curr_rth_lo = min(self.curr_rth_lo, curr_price)
        self.curr_rth_hi = max(self.curr_rth_hi, curr_price)

        if curr_dt.hour < 7 or (curr_dt.hour == 7 and curr_dt.minute < 30):
            self.curr_ib_lo = min(self.curr_ib_lo, curr_price)
            self.curr_ib_hi = max(self.curr_ib_hi, curr_price)

        
        # Get SMAs and normalize by current price
        min_sma5 = self.minute_df['sma5'].loc[curr_dt.round('1Min')]
        min_sma20 = self.minute_df['sma20'].loc[curr_dt.round('1Min')]

        sma20 = self.curr_td_df.loc[self.t_idx-20:self.t_idx+1]['Close'].mean()
        
        # Convert to relative values (percentage deviation from current price)
        min_sma5_rel = (min_sma5 - curr_price) / curr_price
        min_sma20_rel = (min_sma20 - curr_price) / curr_price

        sma20_rel = (sma20 - curr_price) / curr_price

        # Normalize PnL by current price to make it scale invariant
        pnl_rel = self.pnl_day / curr_price

        curr_volume = self.curr_td_df.loc[self.t_idx]['Volume']
        curr_buy_volume = self.curr_td_df.loc[self.t_idx]['TickDataSaver:Buys']
        curr_sell_volume = self.curr_td_df.loc[self.t_idx]['TickDataSaver:Sells']


        dist_to_ib_hi = (self.curr_ib_hi - curr_price)
        dist_to_ib_lo = (self.curr_ib_lo - curr_price)

        dist_to_ovn_hi = (self.daily_stats['ovn_hi'] - curr_price)
        dist_to_ovn_lo = (self.daily_stats['ovn_lo'] - curr_price)

        state_vec = [returns.iloc[-1], min_sma5_rel, min_sma20_rel, sma20_rel, curr_volume,
                    curr_buy_volume, curr_sell_volume, dist_to_ib_hi, dist_to_ib_lo, dist_to_ovn_hi, dist_to_ovn_lo, self.position, pnl_rel, self.trade_count]
        
        return np.array(state_vec, dtype=np.float32)

    def step(self, action):
        # action: (a, exit_flag) tuple
        a, exit_flag = action
        price_prev = self._get_price()
        # 1) adjust position
        self.position = np.clip(self.position + a, -self.P_max, self.P_max)
        # 2) exit if signaled
        reward = 0.0
        if exit_flag and self.position != 0:
            self.t_idx += 1
            price_new = self._get_price()
            pnl = self.position * (price_new - price_prev)
            reward = pnl
            self.pnl_day += reward
            self.position = 0
            self.trade_count += 1
        else:
            # mark-to-market reward
            self.t_idx += 1
            price_new = self._get_price()
            reward = self.position * (price_new - price_prev)
            self.pnl_day += reward

        # cost signals
        cost_stop = float(self.pnl_day < -self.L_max)
        cost_trade = float(exit_flag and self.position == 0)
        done = (self.t_idx >= len(self.price_df)-1) or (self.pnl_day < -self.L_max)
        next_state = self._get_state()
        return next_state, reward, cost_stop, cost_trade, done

# --- 2. Replay Buffer ---
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'cost1', 'cost2', 'next_state', 'done'))
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self): return len(self.buffer)

# --- 3. Networks ---
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=[128,128]):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# Actor: outputs mean and log_std for continuous or logits for discrete
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.logits = MLP(state_dim, action_dim)
    def forward(self, s):
        logits = self.logits(s)
        return torch.distributions.Categorical(logits=logits)

# Critic Q-network: outputs Q(s,a)
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # now expects one-hot action vector externally
        self.net = MLP(state_dim + action_dim, 1)
    def forward(self, s, a_onehot):
        x = torch.cat([s, a_onehot], dim=1)
        return self.net(x).squeeze(-1)

# Cost critic: same structure as QNet
class CostNet(QNet): pass

# --- 4. Training Loop (simplified) ---

def train_sac(env, buffer, params, load_cpt=False):

    # 1) Build the lookup table once at init
    K = 2
    action_list = []
    for pos_adj in range(-K, K+1):      # e.g. [-2,-1,0,1,2]
        for exit_flag in [0, 1]:        # no-exit or exit
            action_list.append((pos_adj, exit_flag))
    # Now action_list has length (2K+1)*2 = 10 in this example.

    # Unpack params
    state_dim, action_dim = params['state_dim'], params['action_dim']
    batch_size = params['batch_size']
    gamma, tau = params['gamma'], params['tau']
    alpha = params['alpha']  # entropy weight
    lr = params['lr']
    eta_lambda = params['eta_lambda']
    T_max = env.N_max

    # Check if GPU is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Instantiate networks and optimizers  
    policy = PolicyNet(state_dim, action_dim).to(device)
    qr1 = QNet(state_dim, action_dim).to(device)
    qr2 = QNet(state_dim, action_dim).to(device)
    qc1_1 = QNet(state_dim, action_dim).to(device)
    qc1_2 = QNet(state_dim, action_dim).to(device)
    qc2_1 = QNet(state_dim, action_dim).to(device)
    qc2_2 = QNet(state_dim, action_dim).to(device)
    qr1_t = QNet(state_dim, action_dim).to(device)
    qr2_t = QNet(state_dim, action_dim).to(device)
    qc1_1_t = QNet(state_dim, action_dim).to(device)
    qc1_2_t = QNet(state_dim, action_dim).to(device)
    qc2_1_t = QNet(state_dim, action_dim).to(device)
    qc2_2_t = QNet(state_dim, action_dim).to(device)

    for src, tgt in [(qr1, qr1_t),(qr2, qr2_t),(qc1_1, qc1_1_t),(qc1_2, qc1_2_t),(qc2_1, qc2_1_t),(qc2_2, qc2_2_t)]:
        tgt.load_state_dict(src.state_dict())

    opt_policy = optim.Adam(policy.parameters(), lr=lr)
    opt_q = optim.Adam(list(qr1.parameters()) + list(qr2.parameters()), lr=lr)
    opt_c1 = optim.Adam(list(qc1_1.parameters()) + list(qc1_2.parameters()), lr=lr)
    opt_c2 = optim.Adam(list(qc2_1.parameters()) + list(qc2_2.parameters()), lr=lr)
    # Lagrange multipliers
    lambda1 = torch.tensor(0.0, requires_grad=False, device=device)
    lambda2 = torch.tensor(0.0, requires_grad=False, device=device)

    if load_cpt:
        checkpoint = load_checkpoint('checkpoint/sac.pt')
        policy.load_state_dict(checkpoint['policy_state'])
        qr1.load_state_dict(checkpoint['qr1_state'])
        qr2.load_state_dict(checkpoint['qr2_state'])
        qc1_1.load_state_dict(checkpoint['qc1_1_state'])
        qc1_2.load_state_dict(checkpoint['qc1_2_state'])
        qc2_1.load_state_dict(checkpoint['qc2_1_state'])
        qc2_2.load_state_dict(checkpoint['qc2_2_state'])
        qr1_t.load_state_dict(checkpoint['qr1_t_state'])
        qr2_t.load_state_dict(checkpoint['qr2_t_state'])
        qc1_1_t.load_state_dict(checkpoint['qc1_1_t_state'])
        qc1_2_t.load_state_dict(checkpoint['qc1_2_t_state'])
        qc2_1_t.load_state_dict(checkpoint['qc2_1_t_state'])
        qc2_2_t.load_state_dict(checkpoint['qc2_2_t_state'])

        opt_policy.load_state_dict(checkpoint['policy_opt_state'])
        opt_q.load_state_dict(checkpoint['q_opt_state'])
        opt_c1.load_state_dict(checkpoint['c1_opt_state'])
        opt_c2.load_state_dict(checkpoint['c2_opt_state'])
        lambda1 = torch.tensor(checkpoint['lambda1'])
        lambda2 = torch.tensor(checkpoint['lambda2'])

    for epoch in range(params['epochs']):
        print(f'Epoch {epoch + 1}/{params["epochs"]}')
        state = env.reset()
        epoch_rewards = []
        epoch_costs = []
        for t in range(params['steps_per_epoch']):
            s = torch.tensor(state, device=device).unsqueeze(0)
            dist = policy(s)
            a_idx = dist.sample().item()  # discrete action id maps to (a, exit)

            # decode to (pos_adj, exit_flag)
            pos_adj, exit_flag = action_list[a_idx]

            next_s, r, c1, c2, done = env.step((pos_adj, exit_flag))
            buffer.push(state, a_idx, r, c1, c2, next_s, done)
            state = next_s
            epoch_rewards.append(r)
            epoch_costs.append((c1, c2))
            if done:
                state = env.reset()

            # Update networks
            if len(buffer) >= batch_size:
                batch = Transition(*zip(*buffer.sample(batch_size)))
                sb = torch.tensor(np.array(batch.state), device=device)

                ab_idx = torch.tensor(batch.action, device=device)
                # one-hot encode actions for critics
                ab_onehot = nn.functional.one_hot(ab_idx, num_classes=action_dim).float().to(device)

                rb = torch.tensor(batch.reward, device=device)
                c1b = torch.tensor(batch.cost1, device=device)
                c2b = torch.tensor(batch.cost2, device=device)
                nsb = torch.tensor(np.array(batch.next_state), device=device)
                doneb = torch.tensor(batch.done, dtype=torch.float32, device=device)

                # Critic target computation with random next actions
                with torch.no_grad():
                    rand_idx = torch.randint(0, action_dim, (batch_size,), device=device)
                    rand_onehot = nn.functional.one_hot(rand_idx, action_dim).float().to(device)
                    q1n = qr1_t(nsb, rand_onehot)
                    q2n = qr2_t(nsb, rand_onehot)

                    qn = torch.min(q1n, q2n)
                    target_q = rb + (1 - doneb) * gamma * qn
                    # cost targets similarly...
                    c1n = qc1_1_t(nsb, rand_onehot)
                    c1n2 = qc1_2_t(nsb, rand_onehot)
                    target_c1 = c1b + gamma * (1 - doneb) * torch.min(c1n, c1n2)

                    c2n = qc2_1_t(nsb, rand_onehot)
                    c2n2 = qc2_2_t(nsb, rand_onehot)
                    target_c2 = c2b + gamma * (1 - doneb) * torch.min(c2n, c2n2)

                # Q-network update
                q1_pred = qr1(sb, ab_onehot)
                q2_pred = qr2(sb, ab_onehot)
                loss_q = ((q1_pred - target_q)**2 + (q2_pred - target_q)**2).mean()
                opt_q.zero_grad()
                loss_q.backward()
                opt_q.step()

                # cost critics updates
                c1_pred1 = qc1_1(sb, ab_onehot)
                c1_pred2 = qc1_2(sb, ab_onehot)
                loss_c1 = ((c1_pred1 - target_c1)**2 + (c1_pred2 - target_c1)**2).mean()
                opt_c1.zero_grad()
                loss_c1.backward()
                opt_c1.step()

                c2_pred1 = qc2_1(sb, ab_onehot)
                c2_pred2 = qc2_2(sb, ab_onehot)
                loss_c2 = ((c2_pred1 - target_c2)**2 + (c2_pred2 - target_c2)**2).mean()
                opt_c2.zero_grad()
                loss_c2.backward()
                opt_c2.step()

                # Actor update
                dist_s = policy(sb)
                a_samp = dist_s.sample()
                a_samp_onehot = nn.functional.one_hot(a_samp, action_dim).float().to(device)
                logp = dist_s.log_prob(a_samp)
                q1_val = qr1(sb, a_samp_onehot)
                q2_val = qr2(sb, a_samp_onehot)
                qc1_val = torch.min(qc1_1(sb, a_samp_onehot), qc1_2(sb, a_samp_onehot))
                qc2_val = torch.min(qc2_1(sb, a_samp_onehot), qc2_2(sb, a_samp_onehot))

                actor_loss = (alpha * logp - torch.min(q1_val, q2_val) + lambda1*qc1_val + lambda2*qc2_val).mean()
                opt_policy.zero_grad()
                actor_loss.backward()
                opt_policy.step()

                # Dual updates
                Jc1 = c1b.mean(); Jc2 = c2b.mean()
                lambda1 = torch.clamp(lambda1 + eta_lambda*(Jc1 - 0.0), min=0.0)
                lambda2 = torch.clamp(lambda2 + eta_lambda*(Jc2 - T_max), min=0.0)
                # Print losses
                # print(f"Step {t + 1}/{params['steps_per_epoch']}, Loss Q: {loss_q.item():.4f}, Loss C1: {loss_c1.item():.4f}, Loss C2: {loss_c2.item():.4f}, Actor Loss: {actor_loss.item():.4f}")

                # Soft target updates
                for (src, tgt) in [(qr1, qr1_t), (qr2, qr2_t), (qc1_1, qc1_1_t), (qc1_2, qc1_2_t), (qc2_1, qc2_1_t), (qc2_2, qc2_2_t)]:
                    for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
                        p_tgt.data.copy_(p_tgt.data * (1-tau) + p_src.data * tau)

        # Print epoch statistics
        avg_reward = np.mean(epoch_rewards)
        avg_cost1 = np.mean([c[0] for c in epoch_costs])
        avg_cost2 = np.mean([c[1] for c in epoch_costs])
        print(f"Epoch {epoch + 1} completed. Avg Reward: {avg_reward:.4f}, Avg Cost1: {avg_cost1:.4f}, Avg Cost2: {avg_cost2:.4f}")

        if epoch % 10 == 0:
            save_checkpoint('checkpoint/sac.pt', epoch, policy, qr1, qr2, qc1_1, qc1_2, qc2_1, qc2_2, qr1_t, qr2_t, qc1_1_t, qc1_2_t, qc2_1_t, qc2_2_t,
                    opt_policy, opt_q, opt_c1, opt_c2, lambda1.item(), lambda2.item())

    return policy, qr1, qr2, qc1_1, qc1_2, qc2_1, qc2_2


def save_checkpoint(path, epoch, policy, qr1, qr2, qc1_1, qc1_2, qc2_1, qc2_2, qr1_t, qr2_t, qc1_1_t, qc1_2_t, qc2_1_t, qc2_2_t,
                    policy_opt, q_opt, c1_opt, c2_opt, lambda1, lambda2):
    """
    Save model and optimizer states to a checkpoint file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'policy_state': policy.state_dict(),
        'qr1_state': qr1.state_dict(),
        'qr2_state': qr2.state_dict(),
        'qc1_1_state': qc1_1.state_dict(),
        'qc1_2_state': qc1_2.state_dict(),
        'qc2_1_state': qc2_1.state_dict(),
        'qc2_2_state': qc2_2.state_dict(),
        'qr1_t_state': qr1_t.state_dict(),
        'qr2_t_state': qr2_t.state_dict(),
        'qc1_1_t_state': qc1_1_t.state_dict(),
        'qc1_2_t_state': qc1_2_t.state_dict(),
        'qc2_1_t_state': qc2_1_t.state_dict(),
        'qc2_2_t_state': qc2_2_t.state_dict(),
        'policy_opt_state': policy_opt.state_dict(),
        'q_opt_state': q_opt.state_dict(),
        'c1_opt_state': c1_opt.state_dict(),
        'c2_opt_state': c2_opt.state_dict(),
        'lambda1': lambda1,
        'lambda2': lambda2
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path, device='cpu'):
    """
    Load checkpoint and return state dicts.
    """
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


# Example usage:
# prices = load_tick_data(...)
# env = TradingEnv(prices, P_max=5, L_max=10, N_max=20)\#
# buffer = ReplayBuffer(100000)
# params = {'state_dim':6,'action_dim':10,'batch_size':256,
#           'gamma':0.99,'tau':0.005,'alpha':0.2,'lr':3e-4,
#           'eta_lambda':1e-3,'epochs':1000,'steps_per_epoch':1000}
# train_sac(env, buffer, params)


data_filename = 'test_seconds_td0.csv'
price_data_df = pd.read_csv(data_filename)

dt_format_str = "%m/%d/%Y %H:%M:%S"
price_data_df['dt'] = price_data_df.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", dt_format_str), axis=1)


env = TradingEnv(price_data_df, P_max=3, L_max=10, N_max=5)
buffer = ReplayBuffer(100000)
params = {'state_dim':14,'action_dim':10,'batch_size':256,
          'gamma':0.99,'tau':0.005,'alpha':0.2,'lr':3e-4,
          'eta_lambda':1e-3,'epochs':1000,'steps_per_epoch':500}
train_sac(env, buffer, params)
