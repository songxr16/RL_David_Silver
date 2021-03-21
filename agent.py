from classes import Actions, State
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import cm
import random
import csv
import pickle
from mpl_toolkits.mplot3d import Axes3D

random.seed(1)


class Agent:
    def __init__(self, environment, n0):
        self.iteration = 0
        self.n0 = float(n0)
        self.env = environment
        self.method = ""
        self.dealer_edges = [[1, 4], [4, 7], [7, 10]]
        self.player_edges = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
        self.N = np.zeros((self.env.dealer_values, self.env.player_values, self.env.action_values))
        self.Q = np.zeros((self.env.dealer_values, self.env.player_values, self.env.action_values))
        self.E = np.zeros((self.env.dealer_values, self.env.player_values, self.env.action_values))
        self.E_linear = np.zeros(len(self.dealer_edges)*len(self.player_edges)*2)
        self.theta = np.zeros(len(self.dealer_edges)*len(self.player_edges)*2)
        self.my_lambda = 0.0
        self.V = np.zeros((self.env.dealer_values, self.env.player_values))

    def eps_greedy_choice(self, state):
        try:
            visits_to_state = sum(self.N[state.dealer_sum-1, state.player_sum-1, :])
        except:
            visits_to_state = 0

        current_eps = self.n0 / (self.n0 + visits_to_state)
        if random.random() < current_eps:
            if random.random() < 0.5:
                return Actions.hit
            else:
                return Actions.stick
        else:
            return Actions.get_action(np.argmax(self.Q[state.dealer_sum-1, state.player_sum-1, :]))

    def eps_greedy_choice_linear(self, state, epsilon):
        Qa = np.zeros(2)
        if random.random() > epsilon:
            for action in Actions.get_values():
                phi = self.feature_computation(state, action)
                Qa[action] = sum(phi*self.theta)
            action_next = Actions.get_action(np.argmax(Qa))
        else:
            if random.random() < 0.5:
                action_next = Actions.hit
            else:
                action_next = Actions.stick
        phi = self.feature_computation(state, action_next)
        return [action_next, sum(phi*self.theta)]

    def MC_control(self, iteration):
        self.iteration = iteration
        self.method = "MC_control"
        count_wins = 0
        for episode in range(self.iteration):
            episode_pairs = []
            my_state = self.env.get_initial_state()
            while not my_state.terminal:
                my_action = self.eps_greedy_choice(my_state)
                episode_pairs.append((my_state, my_action))
                self.N[my_state.dealer_sum-1, my_state.player_sum-1, Actions.get_value(my_action)] += 1
                my_state = self.env.round(my_state, my_action)
            for current_state, current_action in episode_pairs:
                step = 1.0/(self.N[current_state.dealer_sum-1, current_state.player_sum-1, Actions.get_value(current_action)])
                error = my_state.reward-self.Q[current_state.dealer_sum-1, current_state.player_sum-1, Actions.get_value(current_action)]
                self.Q[current_state.dealer_sum-1, current_state.player_sum-1, Actions.get_value(current_action)] += step*error
            if my_state.reward == 1:
                count_wins += 1
        print(float(count_wins)/self.iteration*100)
        for dealer in range(self.env.dealer_values):
            for player in range(self.env.player_values):
                self.V[dealer, player] = max(self.Q[dealer, player, :])

    def TD_control(self, iteration, my_lambda, avg_iteration):
        self.my_lambda = float(my_lambda)
        self.iteration = iteration
        self.method = "SARSA_control"
        l_mse = 0
        e_mse = np.zeros((avg_iteration, self.iteration))
        MC_Q = pickle.load(open("Q_1000000_MC_control.pkl", "rb"))
        n_elements = MC_Q.shape[0]*MC_Q.shape[1]*2
        for my_iteration in range(avg_iteration):
            self.N = np.zeros((self.env.dealer_values, self.env.player_values, self.env.action_values))
            self.Q = np.zeros((self.env.dealer_values, self.env.player_values, self.env.action_values))
            self.E = np.zeros((self.env.dealer_values, self.env.player_values, self.env.action_values))
            count_wins = 0
            for episode in range(self.iteration):
                self.E = np.zeros((self.env.dealer_values, self.env.player_values, self.env.action_values))
                state = self.env.get_initial_state()
                action = self.eps_greedy_choice(state)
                state_next = self.env.round(state, action)
                while not state.terminal:
                    self.N[state.dealer_sum-1, state.player_sum-1, Actions.get_value(action)] += 1
                    state_next = self.env.round(state, action)
                    action_next = self.eps_greedy_choice(state_next)
                    alpha = 1.0 / (self.N[state.dealer_sum-1, state.player_sum-1, Actions.get_value(action)])
                    try:
                        delta = state_next.reward +\
                                self.Q[state_next.dealer_sum-1, state_next.player_sum-1, Actions.get_value(action_next)] -\
                                self.Q[state.dealer_sum-1, state.player_sum-1, Actions.get_value(action)]
                    except:
                        delta = state_next.reward - self.Q[state.dealer_sum-1, state.player_sum-1, Actions.get_value(action)]
                    self.E[state.dealer_sum-1, state.player_sum-1, Actions.get_value(action)] += 1
                    update = alpha*delta*self.E
                    self.Q = self.Q+update
                    self.E = self.my_lambda*self.E
                    state = state_next
                    action = action_next
                if state_next.reward == 1:
                    count_wins += 1
                e_mse[my_iteration, episode] = np.sum(np.square(self.Q-MC_Q))/float(n_elements)
            print(float(count_wins)/self.iteration*100)
            l_mse += np.sum(np.square(self.Q-MC_Q))/float(n_elements)
        if my_lambda == 0 or my_lambda == 1:
            plot.plot(e_mse.mean(axis=0))
            plot.ylabel('mse vs episodes')
            plot.show()
        for dealer in range(self.env.dealer_values):
            for player in range(self.env.player_values):
                self.V[dealer, player] = max(self.Q[dealer, player, :])
        return float(l_mse)/avg_iteration

    def TD_control_linear(self, iteration, my_lambda, avg_iteration):
        self.my_lambda = float(my_lambda)
        self.iteration = iteration
        self.method = "SARSA_control_linear_approx"
        eps = 0.05
        alpha = 0.01
        l_mse = 0
        e_mse = np.zeros((avg_iteration, self.iteration))
        MC_Q = pickle.load(open("Q_1000000_MC_control.pkl", "rb"))
        n_elements = MC_Q.shape[0]*MC_Q.shape[1]*2
        for my_iteration in range(avg_iteration):
            self.Q = np.zeros((self.env.dealer_values, self.env.player_values, self.env.action_values))
            self.E_linear = np.zeros(len(self.dealer_edges)*len(self.player_edges)*2)
            self.theta = np.random.random(36)*0.2
            count_wins = 0
            for episode in range(self.iteration):
                phi = []
                self.E_linear = np.zeros(36)
                state = self.env.get_initial_state()
                if np.random.random() < 1-eps:
                    Qa = -100000
                    action = None
                    for my_action in Actions.get_values():
                        phi_current = self.feature_computation(state, my_action)
                        Q = sum(self.theta*phi_current)
                        if Q > Qa:
                            Qa = Q
                            action = my_action
                            phi = phi_current
                else:
                    if np.random.random() < 0.5:
                        action = Actions.stick
                    else:
                        action = Actions.hit
                    phi = self.feature_computation(state,action)
                    Qa = sum(self.theta*phi)
                while not state.terminal:
                    self.E_linear[phi == 1] += 1
                    state_next = self.env.round(state, action)
                    delta = state_next.reward - sum(self.theta*phi)
                    if np.random.random() < 1-eps:
                        Qa = -100000.0
                        action = None
                        for my_action in Actions.get_values():
                            phi_current = self.feature_computation(state_next, my_action)
                            Q = sum(self.theta*phi_current)
                            if Q > Qa:
                                Qa = Q
                                action = my_action
                                phi = phi_current
                    else:
                        if np.random.random() < 0.5:
                            action = Actions.stick
                        else:
                            action = Actions.hit
                        phi = self.feature_computation(state_next, action)
                        Qa = sum(self.theta*phi)
                    delta += Qa
                    self.theta += alpha*delta*self.E_linear
                    self.E_linear = self.my_lambda*self.E_linear
                    state = state_next
                if state_next.reward == 1:
                    count_wins += 1
                self.Q = self.deriveQ()
                e_mse[my_iteration, episode] = np.sum(np.square(self.Q-MC_Q))/float(n_elements)
            print(float(count_wins)/self.iteration*100)
            self.Q = self.deriveQ()
            l_mse += np.sum(np.square(self.Q-MC_Q))
        if my_lambda == 0 or my_lambda == 1:
            plot.plot(e_mse.mean(axis=0))
            plot.ylabel('mse vs episodes')
            plot.show()
        for dealer in range(self.env.dealer_values):
            for player in range(self.env.player_values):
                self.V[dealer, player] = max(self.Q[dealer, player, :])
        return l_mse/float(n_elements)

    def deriveQ(self):
        temp_Q = np.zeros((self.env.dealer_values, self.env.player_values, self.env.action_values))
        for i in range(self.env.dealer_values):
            for j in range(self.env.player_values):
                for k in range(self.env.action_values):
                    phi = self.feature_computation(State(i, j), k)
                    temp_Q[i, j, k] = sum(phi*self.theta)
        return temp_Q

    def feature_computation(self, state, action):
        feature_vector = []
        for i in range(len(self.player_edges)):
            for j in range(len(self.dealer_edges)):
                player = self.player_edges[i]
                dealer = self.player_edges[j]
                if player[0] <= state.player_sum <= player[1] and dealer[0] <= state.dealer_sum <= dealer[1]:
                    feature_vector.append(1)
                else:
                    feature_vector.append(0)
        if action == Actions.hit:
            return np.concatenate([np.array(feature_vector), np.zeros(18)])
        else:
            return np.concatenate([np.zeros(18), np.array(feature_vector)])

    def store_state_value_function(self):
        with open('results.csv', 'wb') as csv_out:
            write_out = csv.writer(csv_out, delimiter=',')
            for row in self.V:
                write_out.writerow(row)

    def store_Qvalue_function(self):
        pickle.dump(self.Q, open("Q_%s_%s.pkl" % (self.iteration, self.method), "wb"))

    def show_state_value_function(self):

        def get_state_value(x, y):
            return self.V[x, y]

        figure = plot.figure()
        axis = figure.add_subplot(111, projection='3d')
        X = np.arange(0, self.env.dealer_values, 1)
        Y = np.arange(0, self.env.player_values, 1)
        X, Y = np.meshgrid(X, Y)
        Z = get_state_value(X, Y)
        surface = axis.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plot.show()
        my_method = self.method
        my_iteration = str(self.iteration)
        pickle.dump(self.V, open("value_%s_%s.pkl" % (my_iteration, my_method), "wb"))

    def show_previous_state_value_function(self, path):
        V = pickle.load(open(path, "rb"))

        def get_state_value(x, y):
            return V[x, y]

        figure = plot.figure()
        axis = figure.add_subplot(111, projection='3d')
        X = np.arange(0, self.env.dealer_values-1, 1)
        Y = np.arange(0, self.env.player_values-1, 1)
        X, Y = np.meshgrid(X, Y)
        Z = get_state_value(X, Y)
        surface = axis.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plot.show()
