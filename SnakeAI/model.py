import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DeepQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(hidden_size, hidden_size*2)
        self.dropout2 = nn.Dropout(0.2)
        self.linear_out = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        x = self.dropout1(F.relu(self.linear1(x)))
        x = self.dropout2(F.relu(self.linear2(x)))
        out = self.linear_out(x)
        return out

    def save(self, file_name='./model.pt'):
        torch.save(self.state_dict(), file_name)


class QNetTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        self.model.train()
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

        
        