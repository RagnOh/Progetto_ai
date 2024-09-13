import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Creazione dell'ambiente FrozenLake
grid_size = 4  # Puoi modificare la dimensione della griglia se desideri un ambiente diverso
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)

# Parametri per SARSA
alpha = 0.001  # Tasso di apprendimento per l'ottimizzatore
gamma = 0.6  # Fattore di sconto
epsilon = 1.0  # Parametro epsilon-greedy per l'esplorazione iniziale
min_epsilon = 0.01
epsilon_decay = 0.995  # Fattore di decadimento dell'epsilon
num_episodes = 4000  # Numero di episodi per il training
max_steps_per_episode = 200  # Numero massimo di passi per episodio

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

# Inizializzazione del modello e dell'ottimizzatore
state_size = env.observation_space.n
action_size = env.action_space.n
q_network = QNetwork(state_size, action_size)
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

# Algoritmo SARSA con rete neurale
for episode in range(num_episodes):
    # Reset dell'ambiente
    state, _ = env.reset()
    action = choose_action(state, q_network, epsilon)
    total_reward = 0
    step_count = 0

    for step in range(max_steps_per_episode):
        # Esegui l'azione e ottieni il nuovo stato, la ricompensa, etc.
        next_state, reward, done, _, _ = env.step(action)
        next_action = choose_action(next_state, q_network, epsilon)
        total_reward += reward
        step_count += 1

        # Ottieni la stima Q per stato e azione corrente
        state_tensor = one_hot_state(state, state_size)
        next_state_tensor = one_hot_state(next_state, state_size)

        # Calcola il target di SARSA
        with torch.no_grad():
            target = reward + gamma * q_network(next_state_tensor)[next_action] * (1 - done)

        # Ottieni la stima Q per l'azione corrente
        q_value = q_network(state_tensor)[action]

        # Calcola la loss e aggiorna la rete
        loss = loss_fn(q_value, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Aggiorna lo stato e l'azione
        state = next_state
        action = next_action

        if done:
            break

    # Aggiorna epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Memorizza ricompensa totale e passaggi
    sum_rewards.append(total_reward)
    epsilon_history.append(epsilon)
    steps_per_episode.append(step_count)

# Grafici per i risultati
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
plt.savefig(f'frozen_lake_sarsa_{grid_size}x{grid_size}.png')

# Grafico per il numero di passi per episodio
plt.figure(figsize=(10, 5))
plt.plot(steps_per_episode)
plt.title(f'Steps per Episode (Grid Size: {grid_size}x{grid_size})')
plt.xlabel('Episodes')
plt.ylabel('Number of Steps')
plt.tight_layout()
plt.savefig(f'steps_per_episode_sarsa_{grid_size}x{grid_size}.png')
plt.show()

# Chiusura dell'ambiente
env.close()
