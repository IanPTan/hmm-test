import torch as pt


def settle_state(transition):

    eigvals, eigvecs = pt.linalg.eig(transition)
    stationary_idx = torch.argmin(torch.abs(eigvals - 1))
    stationary_vec = eigvecs[:, stationary_idx].real
    stationary_dist = stationary_vec / stationary_vec.sum()

    return stationary_dist



class HMM(pt.nn.Module):


    def __init__(self, transition=None, emission=None, state_amt=None, symbol_amt=None):

        super().__init__()

        transition_missing = transition == None
        emission_missing = emission == None
        state_amt_missing = state_amt == None
        symbol_amt_missing = symbol_amt == None

        if transition_missing and state_amt_missing:
            raise Exception("You must state either transition or state_amt.")
        if emission_missing and (state_amt_missing or symbol_amt_missing):
            raise Exception("You must state either transition or state_amt.")

        if transition == None:
            transition = pt.randn(state_amt, state_amt)
        if emission == None:
            emission = pt.randn(state_amt, symbol_amt)

        self.transition = pt.nn.Parameter(transition)
        self.emission = pt.nn.Parameter(emission)
        self.state_amt, self.symbol_amt = emission.shape
        self.initial_state = pt.eye(self.state_amt)


    def forward(self, x_seq, steps=1, state=None):

        if state == None:
            state = settle_state(self.transition)

        x = x_seq[0]
        symbol_probs = self.emission[:, x]
        state = state * symbol_probs)
        state /= state.sum()

        for x in x_seq[1:]:
            symbol_probs = self.emission[:, x]
            state = self.transition @ state * symbol_probs
            state /= state.sum()

        pred = np.zeros(steps, self.symbol_amt)
        pred[0] = self.emission @ state

        for i in range(1, steps):
            state = self.transition @ state
            state /= state.sum()
            pred[i] = self.emission @ state
        
        return pred
