import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

# Creazione dell'ambiente MountainCarContinuous
env = gym.make("MountainCarContinuous-v0")

# Parametri per PPO
learning_rate = 0.0003
gamma = 0.6
epsilon_clip = 0.2
num_episodes = 1000
max_steps_per_episode = 500
ppo_epochs = 10
gae_lambda = 0.95

# Liste per i grafici
sum_rewards = []
steps_per_episode = []

# Rete di politica e valore combinata
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc_mu = nn.Linear(128, action_size)   # Per la media (azione continua)
        self.fc_std = nn.Linear(128, action_size)  # Per la deviazione standard
        self.softplus = nn.Softplus()  # Softplus come attivazione per la deviazione standard
        self.fc_v = nn.Linear(128, 1)  # Per il valore

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x))  # Azioni normalizzate tra -1 e 1
        std = self.softplus(self.fc_std(x))  # Deviazione standard positiva
        v = self.fc_v(x)
        return mu, std, v

# Inizializzazione del modello
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
policy = ActorCritic(state_size, action_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Funzione per raccogliere le esperienze e addestrare la politica
def compute_advantages(rewards, values, dones, gamma, gae_lambda):
    advantages = []
    advantage = 0
    for i in reversed(range(len(rewards))):
        td_error = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
        advantage = td_error + gamma * gae_lambda * (1 - dones[i]) * advantage
        advantages.insert(0, advantage)
    return advantages

# Funzione per calcolare la loss PPO e aggiornare la politica
def ppo_update(states, actions, old_log_probs, returns, advantages):
    for _ in range(ppo_epochs):
        # Calcolo della nuova politica e valore
        mu, std, values = policy(states)
        dist = Normal(mu, std)
        new_log_probs = dist.log_prob(actions).sum(axis=-1)
        
        # Calcolo del rapporto delle probabilità tra vecchia e nuova politica
        ratio = torch.exp(new_log_probs - old_log_probs.detach())

        # Obiettivo PPO con clipping
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Loss per il valore
        value_loss = (returns - values).pow(2).mean()

        # Loss totale (somma delle due loss)
        loss = policy_loss + 0.5 * value_loss

        # Aggiornamento del modello
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Algoritmo PPO
for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)

    episode_rewards = []
    episode_log_probs = []
    episode_values = []
    episode_dones = []
    episode_states = []
    episode_actions = []

    for step in range(max_steps_per_episode):
        # Forward pass: ottieni la media, deviazione standard e il valore stimato dallo stato
        mu, std, value = policy(state)
        dist = Normal(mu, std)
        action = dist.sample()  # Azione campionata dalla distribuzione gaussiana
        action_clipped = torch.clamp(action, float(env.action_space.low[0]), float(env.action_space.high[0]))

        # Memorizza i dati per l'aggiornamento
        episode_states.append(state)
        episode_actions.append(action_clipped)
        episode_values.append(value)
        episode_log_probs.append(dist.log_prob(action_clipped).sum())

        # Esegui l'azione nell'ambiente
        next_state, reward, done, _, _ = env.step(action_clipped.numpy())
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # Memorizza i dati dell'episodio
        episode_rewards.append(reward)
        episode_dones.append(done)

        # Aggiorna lo stato
        state = next_state

        if done:
            break

    # Valore finale per il calcolo degli advantages
    _, _, final_value = policy(state)
    episode_values.append(final_value)

    # Calcolo di vantaggi e ritorni
    advantages = compute_advantages(episode_rewards, episode_values, episode_dones, gamma, gae_lambda)
    returns = [adv + val for adv, val in zip(advantages, episode_values[:-1])]

    # Conversione in tensori
    states = torch.stack(episode_states)
    actions = torch.stack(episode_actions)
    old_log_probs = torch.stack(episode_log_probs)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)

    # Aggiornamento della politica tramite PPO
    ppo_update(states, actions, old_log_probs, returns, advantages)

    # Memorizza i dati per il grafico
    sum_rewards.append(sum(episode_rewards))
    steps_per_episode.append(len(episode_rewards))

# Grafico delle ricompense totali per episodio
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(sum_rewards)
plt.title('Total Rewards per Episode')
plt.xlabel('Episodes')
plt.ylabel('Rewards')

# Grafico del numero di passi per episodio
plt.subplot(122)
plt.plot(steps_per_episode)
plt.title('Steps per Episode')
plt.xlabel('Episodes')
plt.ylabel('Number of Steps')

plt.tight_layout()
plt.savefig('ppo_mountaincar_continuous.png')
plt.show()

# Chiusura dell'ambiente
env.close()
