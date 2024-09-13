import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Creazione dell'ambiente MountainCarContinuous
env = gym.make("MountainCarContinuous-v0")

# Parametri per SARSA
learning_rate = 0.001
gamma = 0.6
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 1000
max_steps_per_episode = 1000

# Liste per i grafici
sum_rewards = []
steps_per_episode = []
epsilon_history = []

# Rete per il valore Q (stato, azione)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 128)  # Stato + azione come input
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Restituisce Q(s,a)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenazione di stato e azione
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# Rete per la politica (stato -> azione)
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc_mu = nn.Linear(128, action_size)   # Per la media (azione continua)
        self.fc_std = nn.Linear(128, action_size)  # Per la deviazione standard
        self.softplus = nn.Softplus()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        mu = torch.tanh(self.fc_mu(x))  # Azioni normalizzate tra -1 e 1
        std = self.softplus(self.fc_std(x))  # Deviazione standard positiva
        return mu, std

# Inizializzazione delle reti
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
q_network = QNetwork(state_size, action_size)
policy_network = PolicyNetwork(state_size, action_size)
optimizer_q = optim.Adam(q_network.parameters(), lr=learning_rate)
optimizer_policy = optim.Adam(policy_network.parameters(), lr=learning_rate)

# Funzione per selezionare l'azione con esplorazione epsilon-greedy
def select_action(state, epsilon):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Aggiungi una dimensione batch
    if np.random.rand() < epsilon:
        # Esplorazione: scegli un'azione casuale
        action = env.action_space.sample()
    else:
        # Sfruttamento: scegli un'azione dalla politica
        mu, std = policy_network(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample().numpy()[0]  # Campiona dalla distribuzione normale
    action = np.clip(action, env.action_space.low, env.action_space.high)
    return action

# Algoritmo SARSA
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    epsilon_history.append(epsilon)
    
    # Seleziona la prima azione
    action = select_action(state, epsilon)
    
    for step in range(max_steps_per_episode):
        # Esegui l'azione nell'ambiente
        next_state, reward, done, _, _ = env.step(action)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        # Seleziona la prossima azione
        next_action = select_action(next_state, epsilon)
        next_action_tensor = torch.tensor(next_action, dtype=torch.float32).unsqueeze(0)
        
        # Calcola il target Q con SARSA
        with torch.no_grad():
            target_q = reward + gamma * q_network(next_state_tensor, next_action_tensor) * (1 - done)
        
        # Calcola il valore Q per lo stato attuale
        current_q = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0), action_tensor)
        
        # Calcolo della loss e ottimizzazione del Q-network
        loss_q = (current_q - target_q).pow(2).mean()
        optimizer_q.zero_grad()
        loss_q.backward()
        optimizer_q.step()

        # Aggiorna lo stato e l'azione
        state = next_state
        action = next_action
        total_reward += reward
        steps += 1

        if done:
            break

    # Decadimento epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    # Salva ricompense e passi per l'episodio corrente
    sum_rewards.append(total_reward)
    steps_per_episode.append(steps)

    print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward}, Steps: {steps}, Epsilon: {epsilon}")

# Grafico delle ricompense totali per episodio
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(sum_rewards)
plt.title('Total Rewards per Episode')
plt.xlabel('Episodes')
plt.ylabel('Rewards')

# Grafico del decadimento epsilon
plt.subplot(122)
plt.plot(epsilon_history)
plt.title('Epsilon Decay')
plt.xlabel('Episodes')
plt.ylabel('Epsilon')

plt.tight_layout()
plt.savefig('sarsa_mountaincar_continuous.png')
plt.show()

# Grafico del numero di passi per episodio
plt.figure(figsize=(10, 5))
plt.plot(steps_per_episode)
plt.title('Steps per Episode')
plt.xlabel('Episodes')
plt.ylabel('Number of Steps')
plt.tight_layout()
plt.savefig('steps_per_episode_sarsa_mountaincar.png')
plt.show()

# Chiusura dell'ambiente
env.close()
