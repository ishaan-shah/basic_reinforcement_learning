'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning

Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA

        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import gym
import numpy
import random
import pandas
import matplotlib.pyplot as plt


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions


    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)


    def updateQBellman(self, currentState, action, reward, nextState):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''

        Qvalue = self.q.get((currentState, action), None)
        if Qvalue is None:
            self.q[(currentState, action)] = reward
        else:
            rvalue = reward + self.gamma*max([self.getQ(nextState, a) for a in self.actions])
            self.q[(currentState, action)] = Qvalue + self.alpha * (rvalue - Qvalue)

    def updateQConsistent(self, currentState, action, reward, nextState):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        Qvalue = self.q.get((currentState, action), None)
        if Qvalue is None:
            self.q[(currentState, action)] = reward
        else:
            if currentState != nextState:
                rvalue = reward + self.gamma*max([self.getQ(nextState, a) for a in self.actions])
            else:
                rvalue = reward + self.gamma*Qvalue
            self.q[(currentState, action)] = Qvalue + self.alpha * (rvalue - Qvalue)


    def updateQRSO(self, currentState, action, reward, nextState):
        Qvalue = self.q.get((currentState, action), None)
        if Qvalue is None:
            self.q[(currentState, action)] = reward
        else:
            beta = numpy.random.uniform(0,2)
            bellmanValue = reward + self.gamma*max([self.getQ(nextState, a) for a in self.actions])
            rvalue = bellmanValue - beta*(max([self.getQ(currentState, a) for a in self.actions]) - Qvalue)
            self.q[(currentState, action)] = Qvalue + self.alpha * (rvalue - Qvalue)


    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q: # if they want it, give it!
            return action, q
        return action

    """
    def learn(self, state1, action1, reward, state2):
        #maxqnew = max([self.getQ(state2, a) for a in self.actions])
        #self.updateQBellman(state1, action1, reward, reward + self.gamma*maxqnew)

        if self.operator == 1:
            self.updateQBellman(state1,action1,state2,reward)
        elif self.operator == 2:
            self.updateQConsistent(state1,action1,state2,reward)
        else:
            self.updateQRSO(state1,action1,state2,reward)
    """


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

def update(update, epochs):
    env = gym.make('MountainCar-v0')
    # env.monitor.start('/tmp/cartpole-experiment-1', force=True)
    # video_callable=lambda count: count % 10 == 0)

    goal_average_steps = 195
    max_number_of_steps = 200
    last_time_steps = numpy.ndarray(0)
    n_bins = 40

    number_of_features = env.observation_space.shape[0]
    last_time_steps = numpy.ndarray(0)

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    feature1_bins = pandas.cut([-1.2, 0.6], bins=n_bins, retbins=True)[1][1:-1]
    feature2_bins = pandas.cut([-0.07, 0.07], bins=n_bins, retbins=True)[1][1:-1]

    # The Q-learn algorithm
    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.1)

    award_arr = numpy.zeros(epochs)

    for i_episode in range(epochs):
        observation = env.reset()

        feature1, feature2 = observation
        state = build_state([to_bin(feature1, feature1_bins),
                         to_bin(feature2, feature2_bins)])

        #qlearn.epsilon = qlearn.epsilon * 0.999 # added epsilon decay
        cumulated_reward = 0

        for t in range(max_number_of_steps):
            # env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # print(action)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            feature1, feature2 = observation
            nextState = build_state([to_bin(feature1, feature1_bins),
                             to_bin(feature2, feature2_bins)])


            # qlearn.learn(state, action, reward, nextState)
            if (update == 0):
                qlearn.updateQBellman(state, action, reward, nextState)
            elif (update == 1):
                qlearn.updateQConsistent(state, action, reward, nextState)
            else:
                qlearn.updateQRSO(state, action, reward, nextState)


            state = nextState
            cumulated_reward += reward
            if done:
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                break
        award_arr[i_episode] = cumulated_reward

        print("Episode {:d} reward score: {:0.2f}".format(i_episode, cumulated_reward))

    moving_average_reward = numpy.zeros(epochs)
    for i in range(1, epochs):
        moving_average_reward[i] = numpy.mean(award_arr[max(i-1000, 0):i])
    moving_average_reward[0] = -200
    return moving_average_reward
    # env.monitor.close()
    # gym.upload('/tmp/cartpole-experiment-1', algorithm_id='vmayoral simple Q-learning', api_key='your-key')

if __name__ == '__main__':

    num_trials = 4
    epochs = 10000

    bellman_update_arr = numpy.zeros(epochs)
    for i in range(num_trials):
        bellman_update_arr = bellman_update_arr + update(0, epochs)
    bellman_update_arr = bellman_update_arr / float(num_trials)

    consistent_bellman_update_arr = numpy.zeros(epochs)
    for i in range(num_trials):
        consistent_bellman_update_arr = consistent_bellman_update_arr + update(1, epochs)
    consistent_bellman_update_arr = consistent_bellman_update_arr / float(num_trials)

    rso_update_arr = numpy.zeros(epochs)
    for i in range(num_trials):
        rso_update_arr = rso_update_arr + update(2, epochs)
    rso_update_arr = rso_update_arr / float(num_trials)


    """
    test_epochs = 1000
    bellman_update_arr = update(0, test_epochs)
    consistent_bellman_test_arr = update(1, test_epochs)
    rso_test_arr = update(2, test_epochs)
    """


    plt.plot(numpy.arange(epochs), bellman_update_arr, label='bellman')
    plt.plot(numpy.arange(epochs), consistent_bellman_update_arr, label='consistent')
    plt.plot(numpy.arange(epochs), rso_update_arr, label='rso')
    plt.legend(loc='upper left')
    plt.show()


    """
    env = gym.make('MountainCar-v0')
    # env.monitor.start('/tmp/cartpole-experiment-1', force=True)
        # video_callable=lambda count: count % 10 == 0)

    goal_average_steps = 195
    max_number_of_steps = 1000
    last_time_steps = numpy.ndarray(0)
    n_bins = 10

    number_of_features = env.observation_space.shape[0]
    last_time_steps = numpy.ndarray(0)

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    feature1_bins = pandas.cut([-1.2, 0.6], bins=n_bins, retbins=True)[1][1:-1]
    feature2_bins = pandas.cut([-0.07, 0.07], bins=n_bins, retbins=True)[1][1:-1]

    # The Q-learn algorithm
    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.1,operator=1)

    for i_episode in range(20000):
        observation = env.reset()

        feature1, feature2 = observation
        state = build_state([to_bin(feature1, feature1_bins),
                         to_bin(feature2, feature2_bins)])

        qlearn.epsilon = qlearn.epsilon * 0.999 # added epsilon decay
        cumulated_reward = 0

        for t in range(max_number_of_steps):
            # env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            feature1, feature2 = observation
            nextState = build_state([to_bin(feature1, feature1_bins),
                             to_bin(feature2, feature2_bins)])

            # TODO remove
            if reward != -1:
                print(reward)

            qlearn.learn(state, action, reward, nextState)
            state = nextState
            cumulated_reward += reward

            if done:
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                break

        print("Episode {:d} reward score: {:0.2f}".format(i_episode, cumulated_reward))
        """

    # env.monitor.close()
    # gym.upload('/tmp/cartpole-experiment-1', algorithm_id='vmayoral simple Q-learning', api_key='your-key')
