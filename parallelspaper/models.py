import numpy as np
import pandas as pd
from scipy import random as sr

# a list of transition diagrams
transition_diagrams = {}

# From: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0024516
states = [
    "start",
    "a",
    "b_a",
    "c_b",
    "b_c",
    "d_b",
    "g",
    "j",
    "e",
    "c_f",
    "c_i",
    "c_j",
    "f",
    "i",
    "h",
    "d_c",
]

transition_diagram = pd.DataFrame(
    np.zeros((len(states), len(states))), columns=states, index=states
)
transition_diagram.loc["start", "a"] = 1
transition_diagram.loc["a", "a"] = 0.69
transition_diagram.loc["a", "b_a"] = 0.3
transition_diagram.loc["b_a", "c_b"] = 0.69
transition_diagram.loc["c_b", "b_c"] = 0.98
transition_diagram.loc["b_c", "d_b"] = 0.99
transition_diagram.loc["d_b", "g"] = 0.47
transition_diagram.loc["d_b", "e"] = 0.51
transition_diagram.loc["g", "j"] = 0.21
transition_diagram.loc["g", "e"] = 0.78
transition_diagram.loc["j", "c_j"] = 0.96
transition_diagram.loc["e", "f"] = 1.0
transition_diagram.loc["c_f", "b_c"] = 0.99
transition_diagram.loc["c_i", "a"] = 1.0
transition_diagram.loc["c_j", "d_c"] = 0.94
transition_diagram.loc["f", "c_f"] = 0.45
transition_diagram.loc["f", "h"] = 0.52
transition_diagram.loc["i", "c_i"] = 0.97
transition_diagram.loc["h", "a"] = 0.19
transition_diagram.loc["h", "i"] = 0.81
transition_diagram.loc["d_c", "g"] = 1.0
transition_diagram["end"] = 1 - np.sum(transition_diagram.values, axis=1)
transition_diagram = transition_diagram.T
states.append("end")
transition_diagrams["Okada"] = transition_diagram


# Bird 1 from: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001108
states = ["start", "E", "C", "D", "G", "F", "B", "A"]
transition_diagram = pd.DataFrame(
    np.zeros((len(states), len(states))), columns=states, index=states
)
transition_diagram.loc["start", "C"] = 0.04
transition_diagram.loc["start", "E"] = 0.62
transition_diagram.loc["start", "D"] = 0.34
transition_diagram.loc["E", "E"] = 0.15
transition_diagram.loc["E", "C"] = 0.7
transition_diagram.loc["C", "C"] = 0.64
transition_diagram.loc["C", "G"] = 0.13
transition_diagram.loc["C", "D"] = 0.1
transition_diagram.loc["D", "D"] = 0.56
transition_diagram.loc["D", "G"] = 0.064
transition_diagram.loc["D", "F"] = 0.33
transition_diagram.loc["D", "B"] = 0.029
transition_diagram.loc["G", "G"] = 0.029
transition_diagram.loc["G", "B"] = 0.76
transition_diagram.loc["F", "B"] = 1.0
transition_diagram.loc["B", "B"] = 0.38
transition_diagram.loc["B", "G"] = 0.2
transition_diagram.loc["B", "A"] = 0.39
transition_diagram.loc["A", "A"] = 0.74
transition_diagram.loc["A", "G"] = 0.07
transition_diagram["end"] = 1 - np.sum(transition_diagram.values, axis=1)
transition_diagram = transition_diagram.T
states.append("end")
transition_diagrams["Bird1"] = transition_diagram


def sample_sequence_MM(transition_diagram):
    """Samples a Markov model given a pandas dataframe of a transition diagram 
        where start and end refer to the start and end states
    """
    sr.seed()
    states = list(transition_diagram.columns) + ["end"]
    sequence = []
    state = "start"
    while state != "end":
        state = np.random.choice(states, p=transition_diagram[state].values)
        if state != "end":
            if len(state.split("_")) > 1:
                sequence.append(state.split("_")[0])
            else:
                sequence.append(state)
    return sequence


def gen_balanced_matrix(na=5, ps=[0.7, 0.2, 0.1]):
    """ Generates a balanced matrix in which every state can reach every other state
    for hierarchical and Markov models
    """
    for r in range(1000):
        breakme = False
        probs = np.zeros((na, na))
        for p in ps:
            for i in np.arange(na):
                ixloc = np.where(
                    (probs[i, :] == 0) & (np.sum(probs != p, axis=0) == na)
                )[0]
                if len(ixloc) > 0:
                    probs[i, np.random.permutation(ixloc)[0]] = p
                else:
                    # the initialization didn't work
                    breakme = True
        if breakme:
            continue
        probs = probs / np.sum(probs, axis=0)
        return probs
    return "Generation Failed"


def gen_seq_markov(alphabet, probs, seq_len):
    """ like sample_sequence_MM, but uses a numpy matrix, no start and end states, and a set sequence length
    """
    sequence = list(
        np.random.choice(alphabet, p=np.sum(probs, axis=0) / np.sum(probs), size=1)
    )
    for i in range(seq_len):
        sequence.append(np.random.choice(alphabet, p=probs[:, sequence[-1]], size=1)[0])
    return sequence


def gen_seq_hierarchical(alphabet, probs, depth, n_subsamples):
    """ generates a sequence via the Lin Tegmark recursive model
    Arguments:
        alphabet {[type]} -- [alphabet of states]
        probs {[type]} -- [probability matrix for recursive subsampling]
        depth {[type]} -- [how many times to recursively subsample]
        n_subsamples {[type]} -- [the number of new elements to recursively replace old elements with]
    
    Returns:
        sequence [type] -- [sequence of elements]
    """
    sequence = np.random.choice(
        alphabet, p=np.sum(probs, axis=1) / np.sum(probs), size=1
    )
    if type(depth) == list:
        depth = np.random.choice(depth)
    depth_list = range(depth)
    for i in depth_list:
        q = np.random.choice(n_subsamples)
        sequence = subsample_sequence(sequence, probs, q, alphabet)
    return sequence


def subsample_sequence(sequence, probs, q, alphabet):
    """ subsamples a sequence given a probability matrix
    
    given a sequence, resamples each element in that sequences given a probability matrix of sequence element to new elements
    
    Arguments:
        sequence {[type]} -- input sequence
        probs {[type]} -- the probability matrix
        q {[type]} -- the number of items to subsample
    """
    return [
        item
        for sublist in [
            np.random.choice(alphabet, p=probs[:, i], size=q) for i in sequence
        ]
        for item in sublist
    ]
