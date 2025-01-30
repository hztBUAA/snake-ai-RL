import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import json
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self, load_existing=True, only_eval=False):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3, load_exists=load_existing)
        if not load_existing:
            print("强制使用新初始化的模型")
            # 可以在这里添加重新初始化参数的代码
        # 只在训练模式下创建trainer
        self.only_eval = only_eval
        if not only_eval:
            self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        else:
            print("当前为评估模式，模型参数不会更新")
            self.model.eval()  # 设置为评估模式

        # 加载历史最高分
        self.record = self.load_record()
        print(f"历史最高分: {self.record}")
    
    def save_record(self, score):
        """保存最高分数到文件"""
        record_file = os.path.join('./model', 'record.json')
        data = {'record': score}
        with open(record_file, 'w') as f:
            json.dump(data, f)
        self.record = score

    def load_record(self):
        """从文件加载历史最高分"""
        record_file = os.path.join('./model', 'record.json')
        try:
            if os.path.exists(record_file):
                with open(record_file, 'r') as f:
                    data = json.load(f)
                return data.get('record', 0)
        except Exception as e:
            print(f"还没有历史最高分，默认为0")
        return 0


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # 在评估模式下，不使用探索策略
        if self.only_eval:
            state0 = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():  # 在评估模式下不计算梯度
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move = [0] * 3
            final_move[move] = 1
            return final_move

        # 训练模式下的原有逻辑
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    agent = Agent()
    record = agent.record
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                agent.save_record(score)  # 保存新的最高分
                print(f"新记录！分数: {score}, 已保存模型")

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


def evaluate():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    agent = Agent(load_existing=True, only_eval=True)  # 评估模式
    game = SnakeGameAI()
    
    while True:
        # 获取当前状态
        state_old = agent.get_state(game)

        # 获取动作
        final_move = agent.get_action(state_old)

        # 执行动作
        reward, done, score = game.play_step(final_move)

        if done:
            game.reset()
            agent.n_games += 1

            print('Game', agent.n_games, 'Score', score, 'Record:', agent.record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    # 根据需要选择模式
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        print("启动评估模式")
        evaluate()
    else:
        print("启动训练模式")
        train()