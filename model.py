import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, load_exists = True):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        # 尝试加载已存在的模型
        if load_exists:
            self.load_state_dict_if_exists()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load_state_dict_if_exists(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        
        if os.path.exists(file_name):
            try:
                self.load_state_dict(torch.load(file_name))
                print(f"成功加载已存在的模型: {file_name}")
                # 切换回训练模式
                self.train()
            except Exception as e:
                print(f"加载模型时出错: {e}")
                print("将使用新初始化的模型")
        else:
            print("未找到已存在的模型，将使用新初始化的模型")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)  # shape: (batch_size, 3)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]: # 终止状态只有即时奖励 # 非终止状态考虑未来奖励  
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # 翻译过来： 
            # 就是第idx个回放样本中的当前state  
            # 如果采用其中action的三选项之一之最佳(这其实正是这一次回放样本采取的决策) 
            # 那么真实target的Q应该是这一次的reward 加上 从下一次状态next_state之后的采取每个动作的预期累计奖励 注意这里的累计 我们采用一个network直接端到端输出了
            # 所以，这里的Q值并不是动作概率，而是在当前状态下采取某个动作预期能获得的累积奖励。模型通过不断学习来优化这些Q值的估计，从而做出更好的决策。
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done 这里有疑问  为什么把model输出的关于action的概率当成Q value？
        # 重点在于我们这里的target标签的值？  为什么能够这么设定 融合了TD的写法
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



