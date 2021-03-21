from environment import Environment
from agent import Agent
import numpy as np
import pickle
import matplotlib.pyplot as plot


def test_Monte_Carlo(iteration=1000000, n=100):
    print("\n-------------------")
    print("Monte_Carlo_control")
    print("run for n. iterations: "+str(iteration))
    print("win percentage: ")
    game = Environment()
    agent = Agent(game, n)
    agent.MC_control(iteration)
    agent.show_state_value_function()
    agent.store_Qvalue_function()


def test_SARSA(iteration=1000, my_lambda=None, n=100, avg_iteration=50):
    print("\n-------------------")
    print("TD_control_SARSA")
    print("run for n. iterations: "+str(iteration))
    print("plot graph mse vs episodes for lambda equal 0 and lambda equal 1")
    print("list (standard output) win percentage for values of lambda 0, 0.1, 0.2, ..., 0.9, 1")
    mse = []
    if not isinstance(my_lambda, list):
        if my_lambda == None:
            lambda_current = 0.5
        else:
            lambda_current = my_lambda
        game = Environment()
        agent = Agent(game, n)
        agent.TD_control(iteration, lambda_current, avg_iteration)
        agent.show_state_value_function()
    else:
        for lambda_current in my_lambda:
            game = Environment()
            agent = Agent(game, n)
            l_mse = agent.TD_control(iteration, lambda_current, avg_iteration)
            mse.append(l_mse)
        plot.plot(my_lambda, mse)
        plot.ylabel('mse')
        plot.show()


def test_linear_SARSA(iteration=1000, my_lambda=None, n=100, avg_iteration=100):
    print("\n-------------------")
    print("TD_control_SARSA, with Linear function approximation")
    print("run for n. iterations: "+str(iteration))
    print("plot graph mse vs episodes for lambda equal 0 and lambda equal 1")
    print("list (std output) win percentage for values of lambda 0, 0.1, 0.2, ..., 0.9, 1")
    mse = []
    if not isinstance(my_lambda, list):
        if my_lambda == None:
            lambda_current = 0.5
        else:
            lambda_current = my_lambda
        game = Environment()
        agent = Agent(game, n)
        agent.TD_control_linear(iteration, lambda_current, avg_iteration)
        agent.show_state_value_function()
    else:
        for lambda_current in my_lambda:
            game = Environment()
            agent = Agent(game, n)
            l_mse = agent.TD_control_linear(iteration, lambda_current, avg_iteration)
            mse.append(l_mse)
        plot.plot(my_lambda, mse)
        plot.ylabel('mse')
        plot.show()


if __name__ == '__main__':
    lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    iteration_MC = 1000000
    iteration_SARSA = 1000
    n0 = 500
    test_Monte_Carlo(iteration_MC, n0)
    test_SARSA(iteration_SARSA, lambdas, n0, avg_iteration=1)
    test_linear_SARSA(iteration_SARSA, lambdas, n0, avg_iteration=1)
