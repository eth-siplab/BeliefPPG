import numpy as np

def update_prob(prev_maxprod, curr, trans_prob):
    """
    Applies one timestep of the forward pass of Viterbi Decoding.
    :param prev_maxprod: np.ndarray of shape (n_bins,) containing probabilities of most probable path per HR bin
    :param curr: np.ndarray of shape (n_bins,) containing the raw model beliefs for the current timestep
    :param trans_prob: np.ndarray of shape (n_bins, n_bins) containing transition probability matrix expressing prior
    :return: tuple (new_maxprod, ixes) containing new best path scores and backpointers respectively
    """
    curr_maxprod = np.empty_like(prev_maxprod)
    ixes = np.empty_like(prev_maxprod, dtype=int)
    for i in range(len(curr)):
        curr_maxprod[i] = np.max(prev_maxprod*trans_prob[:,i]) # store best path score to this bin
        ixes[i] = np.argmax(prev_maxprod*trans_prob[:,i]) # store incoming edge for backtracing

    curr_maxprod *= curr
    return curr_maxprod / curr_maxprod.sum(), ixes


def decode_viterbi(raw_pred, prior_layer):
    """
    Performs Viterbi Decoding on the output probabilities.
    That is, uses max-product message passing to find the most likely HR trajectory according to the
    raw class probabilites.
    :param raw_pred: np.ndarray of shape (n_timesteps, n_bins) of raw HR probabilities
    :param prior_layer: PriorLayer that has already been fit to training data
    :return: np.ndarray of shape (n_timesteps, ) of predictions for each step
    """
    trans_probs = prior_layer.transition_prior.numpy()
    dim, dim = trans_probs.shape

    # forward pass
    best_paths = []
    prev_maxprod = np.full((dim,), 1/dim) # initialize with uniform distribution
    for j in range(len(raw_pred)):
        prev_maxprod, paths = update_prob(prev_maxprod, raw_pred[j], trans_probs)
        best_paths.append(paths)

    # backward pass
    best_path = []
    curr_ix = np.argmax(prev_maxprod).astype(int)
    for i in range(len(best_paths)-1):
        best_path.append(curr_ix)
        curr_ix = best_paths[-(i+1)][curr_ix]
    best_path.append(curr_ix)

    return np.array([prior_layer.hr(x) for x in reversed(best_path)])


