import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Creazione dell'ambiente FrozenLake
grid_size = 4  # Puoi modificare la dimensione della griglia
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)

# Parametri per DQN
alpha = 0.001  # Tasso di apprendimento
gamma = 0.6  # Fattore di sconto
epsilon = 1.0  # Parametro epsilon-greedy
min_epsilon = 0.01
epsilon_decay = 0.995  # Decadimento di epsilon
num_episodes = 4000  # Numero di episodi per il training
max_steps_per_episode = 100  # Numero massimo di passi per episodio
batch_size = 32  # Dimensione del mini-batch
buffer_size = 1000  # Dimensione massima del replay buffer
target_update = 10  # Frequenza di aggiornamento della rete target

# Replay Buffer
replay_buffer = deque(maxlen=buffer_size)

# Lista per i grafici
sum_rewards = []
epsilon_history = []
steps_per_episode = []

# Definizione della rete neurale per la stima della funzione Q
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        #self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Inizializzazione del modello e della rete target
state_size = env.observation_space.n
action_size = env.action_space.n
q_network = QNetwork(state_size, action_size)
target_network = QNetwork(state_size, action_size)
target_network.load_state_dict(q_network.state_dict())  # Inizializzazione con i pesi della rete Q

optimizer = optim.Adam(q_network.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

# Funzione per trasformare lo stato in un vettore one-hot
def one_hot_state(state, state_size):
    state_vector = np.zeros(state_size)
    state_vector[state] = 1
    return torch.tensor(state_vector, dtype=torch.float32)

# Funzione per scegliere un'azione secondo l'epsilon-greedy
def choose_action(state, q_network, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Esplorazione
    else:
        with torch.no_grad():
            state_tensor = one_hot_state(state, state_size)
            q_values = q_network(state_tensor)
            return torch.argmax(q_values).item()  # Sfruttamento

# Funzione per memorizzare esperienze nel replay buffer
def store_experience(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

# Funzione per addestrare il modello con mini-batch
def train_q_network():
    if len(replay_buffer) < batch_size:
        return

    # Preleva un mini-batch dal replay buffer
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Prepara i tensori
    states = torch.stack([one_hot_state(s, state_size) for s in states])
    next_states = torch.stack([one_hot_state(s, state_size) for s in next_states])
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Stima Q corrente
    q_values = q_network(states).gather(1, actions.view(-1, 1)).squeeze()

    # Stima Q target
    with torch.no_grad():
        q_next = target_network(next_states).max(1)[0]
        q_targets = rewards + gamma * q_next * (1 - dones)

    # Calcolo della loss
    loss = loss_fn(q_values, q_targets)

    # Aggiornamento dei pesi
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Algoritmo Deep Q-Learning
for episode in range(num_episodes):
    # Reset dell'ambiente
    state, _ = env.reset()
    total_reward = 0
    step_count = 0

    for step in range(max_steps_per_episode):
        # Selezione dell'azione con epsilon-greedy
        action = choose_action(state, q_network, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1

        # Memorizza l'esperienza nel replay buffer
        store_experience(state, action, reward, next_state, done)

        # Aggiorna lo stato
        state = next_state

        # Addestra il modello con il replay buffer
        train_q_network()

        if done:
            break

    # Aggiorna epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Aggiorna la rete target ogni 'target_update' episodi
    if episode % target_update == 0:
        target_network.load_state_dict(q_network.state_dict())

    # Memorizza ricompensa totale e passaggi
    sum_rewards.append(total_reward)
    epsilon_history.append(epsilon)
    steps_per_episode.append(step_count)

# Grafici per i risultati
#plt.figure(figsize=(12, 5))
sum_rewards2 = np.zeros(num_episodes)
for x in range(num_episodes):
        sum_rewards2[x] = np.sum(sum_rewards[max(0, x-100):(x+1)])
    
plt.subplot(121)
plt.plot(sum_rewards2)
plt.title(f'Average Rewards (Grid Size: {grid_size}x{grid_size})')
plt.xlabel('Episodes')
plt.ylabel('Rewards')

plt.subplot(122)
plt.plot(epsilon_history)
plt.title(f'Epsilon Decay (Grid Size: {grid_size}x{grid_size})')
plt.xlabel('Episodes')
plt.ylabel('Epsilon')

plt.tight_layout()
plt.savefig(f'frozen_lake_dqn_{grid_size}x{grid_size}.png')

# Grafico per il numero di passi per episodio
plt.figure(figsize=(10, 5))
plt.plot(steps_per_episode)
plt.title(f'Steps per Episode (Grid Size: {grid_size}x{grid_size})')
plt.xlabel('Episodes')
plt.ylabel('Number of Steps')
plt.tight_layout()
plt.savefig(f'steps_per_episode_dqn_{grid_size}x{grid_size}.png')
plt.show()

# Chiusura dell'ambiente
env.close()
