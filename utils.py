import os
import datetime as dt
import matplotlib.pyplot as plt

def save_agent(agent, i):
    today = dt.datetime.now().date().strftime("%Y%m%d")
    path = os.path.join(os.getcwd(), 'log/agents', today)
    if not os.path.exists(path):
        if not os.path.exists(os.path.join(os.getcwd(), 'log')):
            os.makedirs('log')
        if not os.path.exists(os.path.join(os.getcwd(), './log/agents')):
            os.makedirs('log/agents')
        os.makedirs(path)
    file_nm = os.path.join(path, f"agent_{i}")
    agent.save(file_nm)

def save_img(train_returns, test_returns):
    path = os.path.join(os.getcwd(), 'log/img')
    if not os.path.exists(path):
        os.makedirs(path)
    today = dt.datetime.now().date().strftime('%Y%m%d')
    train_file_nm = os.path.join(path, f"{today}_train.png")
    test_file_nm  = os.path.join(path, f"{today}_test.png")

    plt.figure(figsize=(10, 5))
    plt.plot(train_returns)
    plt.title('Train Return')
    plt.ylabel('Return')
    plt.xlabel('Episode')
    plt.legend()
    plt.savefig(train_file_nm)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(test_returns)
    plt.title('Test Return')
    plt.ylabel('Return')
    plt.xlabel('Episode')
    plt.legend()
    plt.savefig(test_file_nm)
    plt.close()