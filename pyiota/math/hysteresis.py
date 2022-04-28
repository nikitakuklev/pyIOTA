import numpy as np


class PreisachModel:
    """
    Simple hysteresis model based on discrete switches that are weighted by function mu
    that is specific to the material.
    References:
        Wiki: https://en.wikipedia.org/wiki/Preisach_model_of_hysteresis
        https://arxiv.org/pdf/1909.13148.pdf
        http://dx.doi.org/10.3390/en12234466
        http://cds.cern.ch/record/2725885/files/materials-13-02561.pdf
    """

    def __init__(self, mesh_size: int = 70, value=0):
        self.mesh = self._gen_mesh(mesh_size, 'sobol')

        # Hysteron density is flat at start
        self.mu = np.ones(self.mesh.shape[1])

        # Hysterons are off at start
        self.last_state = np.zeros_like(self.mu)
        self.last_input = value

        self.states = [self.last_state]
        self.inputs = [self.last_input]
        self.outputs = [0.0]

        self.initial_value = value

    def _gen_mesh(self, mesh_size, mode='sobol'):
        if mode == 'sobol':
            from scipy.stats.qmc import Sobol
            s = Sobol(d=2)
            # This is technically not balanced
            mesh = s.random(mesh_size)
        elif mode == 'grid':
            # Simple 2D grid
            mx, my = np.meshgrid(np.linspace(0, 1, mesh_size), np.linspace(0, 1, mesh_size))
            mx = mx.flatten()
            my = my.flatten()
            mask = mx < my
            mesh = np.vstack([mx[mask], my[mask]])
        else:
            raise ValueError(f"Unrecognized mode {mode}")
        return mesh

    def step(self, value):
        """
        Apply a new input to the model and take a step
        :param value:
        """
        assert 0 <= value <= 1
        self.inputs.append(value)
        print(value, self.last_input)
        print(np.sum(self.last_state))
        if value > self.last_input:
            # Sweep in y
            mask = self.mesh[1, :] < value
            state = self.last_state.copy()
            state[mask] = 1
        elif value < self.last_input:
            # Sweep in x
            mask = self.mesh[0, :] > value
            state = self.last_state.copy()
            print(np.sum(state))
            state[mask] = 0
            print(np.sum(state))
        else:
            state = self.last_state.copy()
        self.states.append(state)
        self.last_input = value
        self.output = np.sum(state * self.mu)
        self.outputs.append(self.output)

    def simulate_history(self, values):
        """
        Run the model for all values in series, storing history
        :param values:
        """
        for value in values:
            self.step(value)

    def reset(self):
        self.mu = np.ones(self.mesh.shape[1])
        self.last_state = np.zeros_like(self.mu)
        self.last_input = self.initial_value

        self.inputs = [self.last_input]
        self.outputs = [0.0]
        self.states = [self.last_state]

    def simulate_gaussian_hysteron_density(self):
        """
        Apply simple 2D Gaussian density to the hysterons
        """
        x = self.mesh[0,:] - 0.5
        y = self.mesh[1,:] - 0.5
        dst = np.sqrt(x * x + y * y)
        sigma = 0.3
        mu = 0
        gauss = np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2)))
        self.mu = gauss