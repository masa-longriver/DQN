import json
import numpy as np
import torch
import torch.optim as optim
import pfrl
from pfrl import agents, replay_buffers, explorers

from env import KnapsackEnvironment
from model import QFunction
from utils import save_agent, save_img

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    pfrl.utils.set_random_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu = 0 if device.type == 'cuda' else None

    env = KnapsackEnvironment(config)                  # 環境を定義・初期化
    q_func = QFunction(config)                         # Q関数を定義
    optimizer = optim.Adam(q_func.parameters())
    rbuf = replay_buffers.ReplayBuffer(5 * 10 ** 5)    # 経験を貯めるバッファ
    explorer = explorers.ConstantEpsilonGreedy(        # 探索アルゴリズム
        epsilon=config['param']['epsilon'],
        random_action_func=lambda: np.random.randint(0, config['env']['n_cand'])  # アクションの数
    )
    agent = agents.DQN(                                # エージェントの初期化
        q_func,
        optimizer,
        rbuf,
        gpu=gpu,
        gamma=config['param']['gamma'],
        explorer=explorer,
        replay_start_size=config['param']['replay_start_size'],
        update_interval=config['param']['update_interval'],
        target_update_interval=config['param']['target_update_interval'],
        clip_delta=True,
        phi=lambda x: x.astype(np.float32, copy=False)
    )

    print("=============== train ===============", flush=True)
    train_returns = []
    for i in range(1, config['run']['n_episodes'] + 1):
        obs = env.reset()
        reward = 0
        done = False
        R = 0
        t = 0
        while not done and t < config['run']['max_episode_len']:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            R += reward
            t += 1
            reset = t == config['run']['max_episode_len']
            agent.observe(obs, reward, done, reset)
        train_returns.append(R)
        if i % 10 == 0:
            print(f"Episode: {i}, length: {t}, return: {R}", flush=True)
        if i % 10000 == 0:
            save_agent(agent, i)
    print(f"last train info", flush=True)
    print(info, flush=True)
    
    print("=============== test ==============", flush=True)
    test_returns = []
    with agent.eval_mode():
        for i in range(1, config['run']['n_episodes_test']+1):
            obs = env.reset()
            done = False
            R = 0
            t = 0
            while not done and t < config['run']['max_episode_len']:
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                R += reward
                t += 1
            test_returns.append(R)
            print(f"Episode: {i}, length: {t}, return: {R}", flush=True)
        print(f"last test info", flush=True)
        print(info, flush=True)
    
    save_img(train_returns, test_returns)
