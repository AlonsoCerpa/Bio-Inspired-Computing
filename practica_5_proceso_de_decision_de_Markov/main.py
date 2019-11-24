import numpy as np

# Ejemplo:
# 5 estados
#
#            s0 s1 s2 s3 s4
# rewards = [0, 0, 0, 0, 1]
# 2 acciones: a0 (derecha) y a1 (izquierda)
#
#           s0  s1  s2  s3  s4 
# policy = [1, 0, 0, 1, 0] ---> 0 = a0; 1 = a1
#                   
#
#                          s'
#                   s0 s1 s2 s3 s4
# P(s'|s,a0) =  s0 [[0, 1, 0, 0, 0],
#               s1  [0, 0, 1, 0, 0],
#             s s2  [0, 0, 0, 1, 0],
#               s3  [0, 0, 0, 0, 1],
#               s4  [0, 0, 0, 0, 1]]
#
#
#                          s'
#                   s0 s1 s2 s3 s4
# P(s'|s,a1) =  s0 [[1, 0, 0, 0, 0],
#               s1  [1, 0, 0, 0, 0],
#             s s2  [0, 1, 0, 0, 0],
#               s3  [0, 0, 1, 0, 0],
#               s4  [0, 0, 0, 1, 0]]
#
#
# P = [ P(s'|s,a0), P(s'|s,a1) ] en una tercera dimension
# P.shape = (#acciones, #estados, #estados)

def iterative_policy_evaluation(rewards, discount_factor, policy, P, precision):
    num_states = rewards.shape[0]
    values_old = np.zeros(num_states)
    values_new = np.zeros(num_states)
    current_precision = 100.0
    while current_precision >= precision:
        for state in range(num_states):
            action = policy[state]
            values_new[state] = rewards[state] + discount_factor * np.dot(P[action, state], values_old)
        current_precision = np.sum(np.abs(values_old - values_new))/num_states
        values_old = values_new.copy()
    return values_new

def policy_improvement(rewards, discount_factor, P, values):
    num_states = rewards.shape[0]
    num_actions = P.shape[0]
    Q = np.zeros((num_states, num_actions))
    for state in range(num_states):
        for action in range(num_actions):
            Q[state, action] = rewards[state] + discount_factor * np.dot(P[action, state], values)
    policy = np.argmax(Q, axis=1)
    return policy

def main():
    rewards = np.array([0, 0, 0, 0, 1])
    discount_factor = 0.999
    old_policy = np.array([0, 1, 1, 0, 1])
    new_policy = np.array([0, 1, 1, 0, 1])
    P = np.array([[[0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1]],

                  [[1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0]]])
    precision = 0.00001
    convergence = False

    print("Politica = ", old_policy)
    while (convergence == False):
        old_policy = new_policy.copy()
        values = iterative_policy_evaluation(rewards, discount_factor, old_policy, P, precision)
        print("Valores = ", values, "\n")
        new_policy = policy_improvement(rewards, discount_factor, P, values)
        print("Politica = ", new_policy)
        convergence = np.array_equal(old_policy, new_policy)

main()