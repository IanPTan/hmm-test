import torch as pt


def settle_state(transition):

    eigvals, eigvecs = pt.linalg.eig(transition)
    stationary_idx = pt.argmin(pt.abs(eigvals - 1))
    stationary_vec = eigvecs[:, stationary_idx].real
    stationary_dist = stationary_vec / stationary_vec.sum()

    return stationary_dist



class HMM(pt.nn.Module):


    def __init__(self, transition=None, emission=None, state_amnt=None, symbol_amnt=None):

        super().__init__()

        transition_missing = transition == None
        emission_missing = emission == None
        state_amnt_missing = state_amnt == None
        symbol_amnt_missing = symbol_amnt == None

        if transition_missing and state_amnt_missing:
            raise Exception("You must state either transition or state_amnt.")
        if emission_missing and (state_amnt_missing or symbol_amnt_missing):
            raise Exception("You must state either transition or state_amnt.")

        if transition == None:
            transition = pt.rand(state_amnt, state_amnt)
        if emission == None:
            emission = pt.rand(symbol_amnt, state_amnt)

        self.transition = pt.nn.Parameter(transition)
        self.emission = pt.nn.Parameter(emission)
        self.symbol_amnt, self.state_amnt = emission.shape

        self.softmax = pt.nn.Softmax(dim=0)
        self._normalize()


    def _normalize(self):
        with pt.no_grad():
            self.transition.copy_(self.softmax(self.transition))
            self.emission.copy_(self.softmax(self.emission))


    def forward(self, x_seq, steps=1, state=None):

        if state == None:
            state = settle_state(self.transition)[None, :]

        state = state.T

        for x in x_seq.T:
            state = self.emission[x].T * state
            state /= state.sum(dim=0)
            state = self.transition @ state

        pred = pt.zeros(len(x_seq), steps, self.symbol_amnt)
        pred[:, 0] = (self.emission @ state).T

        for i in range(1, steps):
            state = self.transition @ state
            pred[:, i] = (self.emission @ state).T
        
        return pred


if __name__ == "__main__":
    m = HMM(state_amnt=3, symbol_amnt=2)
    x = pt.randint(2, (5, 4))
    y = m(x)
