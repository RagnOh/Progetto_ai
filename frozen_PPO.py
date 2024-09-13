import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Creazione dell'ambiente FrozenLake
grid_size = 4  # Puoi modificare la dimensione della griglia
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)

# Parametri per PPO
learning_rate = 0.001
gamma = 0.6  # Fattore di sconto
epsilon_clip = 0.2  # Parametro epsilon per il clipping
num_episodes = 4000  # Numero di episodi per il training
max_steps_per_episode = 100
batch_size = 32
ppo_epochs = 4  # Numero di aggiornamenti su ogni batch di dati
gae_lambda = 0.95  # Per il calcolo del Generalized Advantage Estimation

# Lista per i grafici
sum_rewards = []
steps_per_episode = []

# Rete di politica e valore combinata
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc_pi = nn.Linear(128, action_size)  # Per la politica (azione)
        self.fc_v = nn.Linear(128, 1)  # Per il valore

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        pi_logits = self.fc_pi(x)
        v = self.fc_v(x)
        return pi_logits, v

# Inizializzazione del modello
state_size = env.observation_space.n
action_size = env.action_space.n
policy = ActorCritic(state_size, action_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Funzione per trasformare lo stato in un vettore one-hot
def one_hot_state(state, state_size):
    state_vector = np.zeros(state_size)
    state_vector[state] = 1
    return torch.tensor(state_vector, dtype=torch.float32)

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
        pi_logits, values = policy(states)
        dist = Categorical(logits=pi_logits)
        new_log_probs = dist.log_prob(actions)
        
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
    state = one_hot_state(state, state_size)
    
    episode_rewards = []
    episode_log_probs = []
    episode_values = []
    episode_dones = []
    episode_states = []
    episode_actions = []

    for step in range(max_steps_per_episode):
        # Forward pass: ottieni le probabilità e il valore stimato dallo stato
        pi_logits, value = policy(state)
        dist = Categorical(logits=pi_logits)
        action = dist.sample()

        # Memorizza i dati per l'aggiornamento
        episode_states.append(state)
        episode_actions.append(action)
        episode_values.append(value)
        episode_log_probs.append(dist.log_prob(action))

        # Esegui l'azione nell'ambiente
        next_state, reward, done, _, _ = env.step(action.item())
        next_state = one_hot_state(next_state, state_size)

        # Memorizza i dati dell'episodio
        episode_rewards.append(reward)
        episode_dones.append(done)

        # Aggiorna lo stato
        state = next_state

        if done:
            break

    # Valore finale per il calcolo degli advantages
    _, final_value = policy(state)
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

sum_rewards2 = np.zeros(num_episodes)
for x in range(num_episodes):
        sum_rewards2[x] = np.sum(sum_rewards[max(0, x-100):(x+1)])
    
plt.subplot(121)
plt.plot(sum_rewards2)
plt.title(f'Average Rewards (Grid Size: {grid_size}x{grid_size})')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.tight_layout()
plt.savefig(f'frozen_lake_ppo_{grid_size}x{grid_size}.png')



# Grafico del numero di passi per episodio

plt.figure(figsize=(10, 5))
plt.plot(steps_per_episode)
plt.title(f'Steps per Episode (Grid Size: {grid_size}x{grid_size})')
plt.xlabel('Episodes')
plt.ylabel('Number of Steps')
plt.tight_layout()
plt.savefig(f'steps_per_episode_PPO_{grid_size}x{grid_size}.png')

#plt.tight_layout()
#plt.savefig(f'frozen_lake_ppo_{grid_size}x{grid_size}.png')
plt.show()

# Chiusura dell'ambiente
env.close()
