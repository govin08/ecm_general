"""
Multiple Offline RL Algorithms Implementation
오프라인 강화학습 알고리즘들 구현 (CQL, IQL, BC, TD3+BC, DDPG)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import copy
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================================================
# Neural Network Components
# ==============================================================================

class Actor(nn.Module):
    """Actor Network for continuous control"""
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action
        
    def forward(self, state):
        return self.max_action * self.network(state)

class Critic(nn.Module):
    """Critic Network (Q-function)"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)

class TwinCritic(nn.Module):
    """Twin Critic for TD3-style algorithms"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(TwinCritic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def Q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)

class ValueNetwork(nn.Module):
    """Value Network for IQL"""
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.network(state)

# ==============================================================================
# Replay Buffer
# ==============================================================================

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

# ==============================================================================
# 1. Behavior Cloning (BC)
# ==============================================================================

class BehaviorCloning:
    """간단한 모방학습 - 베이스라인"""
    def __init__(self, state_dim: int, action_dim: int, max_action: float, lr: float = 3e-4):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.max_action = max_action
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size: int = 256, iterations: int = 1000):
        """BC 학습"""
        print("Training Behavior Cloning...")
        
        losses = []
        for i in range(iterations):
            state, action, _, _, _ = replay_buffer.sample(batch_size)
            
            # Actor loss: MSE between predicted and actual actions
            predicted_action = self.actor(state)
            actor_loss = F.mse_loss(predicted_action, action)
            
            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()
            
            losses.append(actor_loss.item())
            
            if (i + 1) % 200 == 0:
                print(f"BC Iteration {i+1}/{iterations}, Loss: {actor_loss.item():.6f}")
        
        return {'losses': losses}

# ==============================================================================
# 2. Conservative Q-Learning (CQL)
# ==============================================================================

class ConservativeQL:
    """Conservative Q-Learning - 오프라인 RL SOTA"""
    def __init__(self, state_dim: int, action_dim: int, max_action: float, 
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 cql_alpha: float = 1.0, cql_temp: float = 1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha  # CQL regularization weight
        self.cql_temp = cql_temp    # Temperature for CQL
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = TwinCritic(state_dim, action_dim).to(device)
        self.critic_target = TwinCritic(state_dim, action_dim).to(device)
        
        # Copy parameters to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size: int = 256, iterations: int = 10000):
        """CQL 학습"""
        print("Training Conservative Q-Learning...")
        
        actor_losses = []
        critic_losses = []
        cql_losses = []
        
        for i in range(iterations):
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            
            # Critic update
            with torch.no_grad():
                next_action = self.actor(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + (1 - done) * self.gamma * target_q
            
            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # CQL regularization
            # Sample random actions
            random_actions = torch.FloatTensor(batch_size, self.action_dim).uniform_(-self.max_action, self.max_action).to(device)
            # Sample actions from current policy
            policy_actions = self.actor(state)
            
            # Q-values for random and policy actions
            q1_random, q2_random = self.critic(state, random_actions)
            q1_policy, q2_policy = self.critic(state, policy_actions)
            q1_data, q2_data = self.critic(state, action)
            
            # CQL loss: encourage low Q-values for out-of-distribution actions
            cql_loss1 = torch.logsumexp(torch.cat([q1_random, q1_policy], dim=1) / self.cql_temp, dim=1).mean()
            cql_loss2 = torch.logsumexp(torch.cat([q2_random, q2_policy], dim=1) / self.cql_temp, dim=1).mean()
            cql_loss = cql_loss1 + cql_loss2 - q1_data.mean() - q2_data.mean()
            
            total_critic_loss = critic_loss + self.cql_alpha * cql_loss
            
            self.critic_optimizer.zero_grad()
            total_critic_loss.backward()
            self.critic_optimizer.step()
            
            # Actor update (less frequent)
            if i % 2 == 0:
                actor_action = self.actor(state)
                actor_loss = -self.critic.Q1(state, actor_action).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                actor_losses.append(actor_loss.item())
            
            # Target network update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            critic_losses.append(critic_loss.item())
            cql_losses.append(cql_loss.item())
            
            if (i + 1) % 1000 == 0:
                avg_actor_loss = np.mean(actor_losses[-500:]) if actor_losses else 0
                print(f"CQL Iteration {i+1}/{iterations}")
                print(f"  Critic Loss: {critic_loss.item():.6f}")
                print(f"  CQL Loss: {cql_loss.item():.6f}")
                print(f"  Actor Loss: {avg_actor_loss:.6f}")
        
        return {
            'actor_losses': actor_losses,
            'critic_losses': critic_losses,
            'cql_losses': cql_losses
        }

# ==============================================================================
# 3. Implicit Q-Learning (IQL)
# ==============================================================================

class ImplicitQL:
    """Implicit Q-Learning - 안정적인 오프라인 RL"""
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.7, beta: float = 3.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau  # Expectile parameter
        self.beta = beta  # Temperature for policy extraction
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = TwinCritic(state_dim, action_dim).to(device)
        self.value = ValueNetwork(state_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
    def expectile_loss(self, diff, expectile=0.7):
        """Expectile regression loss"""
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size: int = 256, iterations: int = 10000):
        """IQL 학습"""
        print("Training Implicit Q-Learning...")
        
        actor_losses = []
        critic_losses = []
        value_losses = []
        
        for i in range(iterations):
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            
            # Value update
            with torch.no_grad():
                q1, q2 = self.critic(state, action)
                q = torch.min(q1, q2)
            
            v = self.value(state)
            value_loss = self.expectile_loss(q - v, self.tau).mean()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            # Critic update
            with torch.no_grad():
                target_v = self.value(next_state)
                target_q = reward + (1 - done) * self.gamma * target_v
            
            q1, q2 = self.critic(state, action)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Actor update
            if i % 2 == 0:
                with torch.no_grad():
                    v = self.value(state)
                    q1, q2 = self.critic(state, action)
                    q = torch.min(q1, q2)
                    advantage = q - v
                    weights = torch.clamp(torch.exp(advantage / self.beta), max=100.0)
                
                policy_action = self.actor(state)
                actor_loss = -torch.mean(weights * torch.sum((policy_action - action) ** 2, dim=1, keepdim=True))
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                actor_losses.append(actor_loss.item())
            
            critic_losses.append(critic_loss.item())
            value_losses.append(value_loss.item())
            
            if (i + 1) % 1000 == 0:
                avg_actor_loss = np.mean(actor_losses[-500:]) if actor_losses else 0
                print(f"IQL Iteration {i+1}/{iterations}")
                print(f"  Critic Loss: {critic_loss.item():.6f}")
                print(f"  Value Loss: {value_loss.item():.6f}")
                print(f"  Actor Loss: {avg_actor_loss:.6f}")
        
        return {
            'actor_losses': actor_losses,
            'critic_losses': critic_losses,
            'value_losses': value_losses
        }

# ==============================================================================
# 4. TD3 + BC
# ==============================================================================

class TD3BC:
    """TD3 + Behavior Cloning for offline RL"""
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 alpha: float = 2.5, policy_noise: float = 0.2, noise_clip: float = 0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # BC regularization weight
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = TwinCritic(state_dim, action_dim).to(device)
        self.critic_target = TwinCritic(state_dim, action_dim).to(device)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size: int = 256, iterations: int = 10000):
        """TD3+BC 학습"""
        print("Training TD3 + BC...")
        
        actor_losses = []
        critic_losses = []
        
        for i in range(iterations):
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            
            # Critic update
            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
                
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + (1 - done) * self.gamma * target_q
            
            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Actor update (delayed)
            if i % 2 == 0:
                policy_action = self.actor(state)
                
                # TD3 actor loss
                actor_loss = -self.critic.Q1(state, policy_action).mean()
                
                # BC regularization
                bc_loss = F.mse_loss(policy_action, action)
                
                total_actor_loss = actor_loss + self.alpha * bc_loss
                
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                self.actor_optimizer.step()
                
                # Target network update
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                actor_losses.append(total_actor_loss.item())
            
            critic_losses.append(critic_loss.item())
            
            if (i + 1) % 1000 == 0:
                avg_actor_loss = np.mean(actor_losses[-500:]) if actor_losses else 0
                print(f"TD3+BC Iteration {i+1}/{iterations}")
                print(f"  Critic Loss: {critic_loss.item():.6f}")
                print(f"  Actor Loss: {avg_actor_loss:.6f}")
        
        return {
            'actor_losses': actor_losses,
            'critic_losses': critic_losses
        }

# ==============================================================================
# 5. Offline DDPG (기존 구현 개선)
# ==============================================================================

class OfflineDDPG:
    """Offline DDPG with conservative regularization"""
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005, bc_weight: float = 0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.bc_weight = bc_weight
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size: int = 256, iterations: int = 10000):
        """Offline DDPG 학습"""
        print("Training Offline DDPG...")
        
        actor_losses = []
        critic_losses = []
        
        for i in range(iterations):
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            
            # Critic update
            with torch.no_grad():
                target_action = self.actor_target(next_state)
                target_q = self.critic_target(next_state, target_action)
                target_q = reward + (1 - done) * self.gamma * target_q
            
            current_q = self.critic(state, action)
            critic_loss = F.mse_loss(current_q, target_q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Actor update
            if i % 2 == 0:
                policy_action = self.actor(state)
                
                # Standard DDPG actor loss
                actor_loss = -self.critic(state, policy_action).mean()
                
                # BC regularization for offline setting
                bc_loss = F.mse_loss(policy_action, action)
                total_actor_loss = actor_loss + self.bc_weight * bc_loss
                
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                self.actor_optimizer.step()
                
                # Target network update
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                actor_losses.append(total_actor_loss.item())
            
            critic_losses.append(critic_loss.item())
            
            if (i + 1) % 1000 == 0:
                avg_actor_loss = np.mean(actor_losses[-500:]) if actor_losses else 0
                print(f"Offline DDPG Iteration {i+1}/{iterations}")
                print(f"  Critic Loss: {critic_loss.item():.6f}")
                print(f"  Actor Loss: {avg_actor_loss:.6f}")
        
        return {
            'actor_losses': actor_losses,
            'critic_losses': critic_losses
        }

# ==============================================================================
# Utility Functions
# ==============================================================================

def create_replay_buffer_from_data(rl_data: Dict, capacity: int = 1000000) -> ReplayBuffer:
    """RL 데이터에서 replay buffer 생성"""
    replay_buffer = ReplayBuffer(capacity)
    
    states = rl_data['states']
    actions = rl_data['actions']
    rewards = rl_data['rewards']
    next_states = rl_data['next_states']
    done = rl_data['done']
    
    for i in range(len(states)):
        replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], done[i])
    
    print(f"Replay buffer created with {len(replay_buffer)} samples")
    return replay_buffer

def normalize_data(rl_data: Dict) -> Tuple[Dict, StandardScaler, StandardScaler]:
    """데이터 정규화"""
    state_scaler = StandardScaler()
    action_scaler = StandardScaler()
    
    # 정규화
    states_norm = state_scaler.fit_transform(rl_data['states'])
    actions_norm = action_scaler.fit_transform(rl_data['actions'])
    next_states_norm = state_scaler.transform(rl_data['next_states'])
    
    # 정규화된 데이터로 새로운 딕셔너리 생성
    normalized_rl_data = {
        'states': states_norm,
        'actions': actions_norm,
        'rewards': rl_data['rewards'],
        'next_states': next_states_norm,
        'done': rl_data['done'],
        'state_columns': rl_data['state_columns'],
        'target_idx': rl_data['target_idx'],
        'action_bounds': (np.min(actions_norm), np.max(actions_norm))  # 정규화된 범위
    }
    
    return normalized_rl_data, state_scaler, action_scaler

def compare_algorithms(algorithms: Dict, rl_data: Dict, iterations: int = 5000) -> Dict:
    """여러 알고리즘 성능 비교"""
    print("Comparing multiple offline RL algorithms...")
    print("=" * 60)
    
    # 데이터 정규화
    normalized_data, state_scaler, action_scaler = normalize_data(rl_data)
    
    # 공통 파라미터
    state_dim = normalized_data['states'].shape[1]
    action_dim = normalized_data['actions'].shape[1]
    max_action = 1.0  # 정규화 후 대략적인 최대값
    
    # Replay buffer 생성
    replay_buffer = create_replay_buffer_from_data(normalized_data)
    
    results = {}
    
    for name, algorithm_class in algorithms.items():
        print(f"\n🚀 Training {name}...")
        print("-" * 40)
        
        # 알고리즘 인스턴스 생성
        if name == 'BC':
            algorithm = algorithm_class(state_dim, action_dim, max_action)
            training_results = algorithm.train(replay_buffer, iterations=iterations//5)  # BC는 더 적게
        else:
            algorithm = algorithm_class(state_dim, action_dim, max_action)
            training_results = algorithm.train(replay_buffer, iterations=iterations)
        
        results[name] = {
            'algorithm': algorithm,
            'training_results': training_results,
            'state_scaler': state_scaler,
            'action_scaler': action_scaler
        }
        
        print(f"✅ {name} training completed!")
    
    print("\n" + "=" * 60)
    print("🎉 All algorithms training completed!")
    
    return results

# 사용 예시
if __name__ == "__main__":
    # 알고리즘들 정의
    algorithms = {
        'BC': BehaviorCloning,
        'CQL': ConservativeQL,
        'IQL': ImplicitQL,
        'TD3+BC': TD3BC,
        'DDPG': OfflineDDPG
    }
    
    print("Offline RL Algorithms Implementation Complete!")
    print("Available algorithms:", list(algorithms.keys()))