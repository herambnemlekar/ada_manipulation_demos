import numpy as np
from copy import deepcopy


def get_trajectories(states, demonstrations, transition_function):
    trajectories = []
    for demo in demonstrations:
        s = states[0]
        trajectory = []
        for action in demo:
            p, sp = transition_function(s, action)
            s_idx, sp_idx = states.index(s), states.index(sp)
            trajectory.append((s_idx, action, sp_idx))
            s = sp
        trajectories.append(trajectory)

    return trajectories

def rollout_trajectory(qf, states, transition_function, remaining_actions, start_state=0):

    s = start_state
    available_actions = deepcopy(remaining_actions)
    generated_sequence = []
    while len(available_actions) > 0:
        max_action_val = -np.inf
        candidates = []
        for a in available_actions:
            p, sp = transition_function(states[s], a)
            if sp:
                if qf[s][a] > max_action_val:
                    candidates = [a]
                    max_action_val = qf[s][a]
                elif qf[s][a] == max_action_val:
                    candidates.append(a)
                    max_action_val = qf[s][a]

        if not candidates:
            print(s)
        take_action = np.random.choice(candidates)
        generated_sequence.append(take_action)
        p, sp = transition_function(states[s], take_action)
        s = states.index(sp)
        available_actions.remove(take_action)

    return generated_sequence

def boltzman_likelihood(state_features, trajectories, weights, rationality=0.99):
    n_states, n_features = np.shape(state_features)
    likelihood, rewards = [], []
    for traj in trajectories:
        feature_count = deepcopy(state_features[traj[0][0]])
        for t in traj:
            feature_count += deepcopy(state_features[t[2]])
        total_reward = rationality * weights.dot(feature_count)
        rewards.append(total_reward)
        likelihood.append(np.exp(total_reward))

    return likelihood, rewards