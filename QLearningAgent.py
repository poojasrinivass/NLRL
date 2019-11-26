from core.symbolicEnvironment import CliffWalking, WindyCliffWalking, Unstack, Stack, On
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class QL():
    def __init__(self, actions, action_space):
        self.nA = action_space
        self.actions = actions
        self.Q = dict()

    def act(self, state):
        if state not in self.Q:
            self.Q[state] = [1]*self.nA
        return self.actions[np.argmax(self.Q[state])]

    def action_index(self, action):
        return self.actions.index(action)

    def get_value(self, state, action):
        if state not in self.Q:
            self.Q[state] = [1]*self.nA
        return self.Q[state][self.action_index(action)]

    def update(self, state, action, update):
        if state not in self.Q:
            self.Q[state] = [1]*self.nA
        self.Q[state][self.action_index(action)] += update

def epsilon_greedy_policy(state, Q, env):
    A = np.ones(env.action_n, dtype=float) * 0.1 / env.action_n
    if state not in Q:
        Q[state] = [1]*env.action_n
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - 0.1)
    return env.all_actions[np.random.choice(np.arange(len(A)), p=A)]

def QL_learner(steps, env, agent, discount_factor=0.99, alpha=0.3):
    for i in range(steps):
        env.reset()
        state = env.state

        for t in range(100):
            action = epsilon_greedy_policy(state, agent.Q, env)
            reward, done = env.next_step(action)
            next_state = env.state
            best_action = agent.act(next_state)

            q_state = agent.get_value(state, action)
            if done:
                G = alpha * (reward - q_state)
            else:
                G = alpha * (reward + discount_factor * agent.get_value(next_state, best_action) - q_state)

            agent.update(state, action, G)

            state = next_state

            if done:
                break
    return agent

def test(env_class, name):
    env_tr = env_class()
    labels = list(env_tr.all_variations)
    labels.append("train")
    game_means = []

    for var in env_tr.all_variations:
        print var
        env = env_class()
        env = env.vary(var)
        means = []
        for agent_trs in range(20):
            print agent_trs
            vals = []
            agent = QL(env_tr.all_actions, env_tr.action_n)
            agent = QL_learner(1500, env_tr, agent)
            for _ in range(20):
                env.reset()
                state = env.state
                for t in range(2500):
                    action = agent.act(state)
                    reward, done = env.next_step(action)
                    state = env.state

                    if done:
                        vals.append(env.acc_reward)
                        break
            means.append(np.mean(vals))
        game_means.append(np.mean(means))

    env = env_tr
    means = []
    for agent_trs in range(20):
        print agent_trs
        vals = []
        agent = QL(env_tr.all_actions, env_tr.action_n)
        agent = QL_learner(1500, env_tr, agent)
        for _ in range(20):
            env.reset()
            state = env.state
            for t in range(2500):
                action = agent.act(state)
                reward, done = env.next_step(action)
                state = env.state

                if done:
                    vals.append(env.acc_reward)
                    break
        means.append(np.mean(vals))
    game_means.append(np.mean(means))


    x = np.arange(len(labels))
    width = 0.9

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, game_means, width, label='QL')

    ax.set_ylabel('Rewards')
    ax.set_title(name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(rects1, ax)

    fig.tight_layout()
    plt.show()
    return agent

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

if __name__ == "__main__":
    test(Unstack, "Unstack")
    test(Stack, "Stack")
    test(On, "ON")
    test(CliffWalking, "Cliff Walking")
    test(WindyCliffWalking, "Windy Cliff Walking")
