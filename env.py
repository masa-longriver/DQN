import os
import numpy as np
import pandas as pd
from gym import spaces

class KnapsackEnvironment:
    def __init__(self, config):
        self.n_items    = config['env']['n_items']
        self.n_cand     = config['env']['n_cand']
        self.capacity   = config['env']['capacity']
        self.mu_poisson = config['env']['mu_poisson']

        self.action_space = spaces.Discrete(self.n_cand)
        self.reset()

        # 候補アイテムの重量と価値の情報
        # shape: 500(items) * 2(weight + value)
        self.observation_space = spaces.Box(low=0, high=max(self.max_weight, self.max_value),
                                            shape=(self.n_cand * 2,), dtype=np.float32)
      
    def reset(self):
        file_path = os.path.join(os.getcwd(), 'data', 'items.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            self.weights = df['WEIGHT'].to_numpy()
            self.values  = df['VALUE'].to_numpy()
        else:
            file_dir = os.path.join(os.getcwd(), 'data')
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            self.weights = np.random.poisson(lam=1, size=self.n_items) * self.mu_poisson + 50
            self.values  = np.random.poisson(lam=1, size=self.n_items) * self.mu_poisson + 50
            df = pd.DataFrame({'ITEM_NO': [i for i in range(1, self.n_items+1)],
                               'WEIGHT' : self.weights,
                               'VALUE'  : self.values})
            df.to_csv(file_path, index=False)
        
        # 10000アイテムから候補500アイテムを選ぶ
        self.cand_indices = np.random.choice(self.n_items, self.n_cand, replace=False)
        self.max_weight = self.weights.max()
        self.max_value = self.values.max()

        # ナップサック状態を初期化
        self.current_capacity = self.capacity
        self.selected_items = np.zeros(self.n_cand, dtype=np.int32)

        return self.get_observation()
    
    def step(self, action):
        index = self.cand_indices[action]
        # 既に選ばれたアイテム、または容量オーバーのアイテムを選んだらペナルティ
        if self.selected_items[action] == 1 or self.weights[index] > self.current_capacity:
            reward = -self.values[index]
        else:
            self.selected_items[action] = 1
            self.current_capacity -= self.weights[index]
            reward = self.values[index]
        
        # 終了条件：すべてのアイテムを入れる、または容量がなくなる
        done = np.sum(self.selected_items) == self.n_cand or self.current_capacity <= 0
        info = {
            'selected_item_count': np.sum(self.selected_items),
            'remaining_capacity' : self.current_capacity
        }

        return self.get_observation(), reward, done, info
    
    def get_observation(self):
        # 候補アイテムの重量と価値の情報を結合して返す
        weight_values = np.vstack((self.weights[self.cand_indices], self.values[self.cand_indices])).T
        
        return weight_values.flatten()