import jax
import jax.numpy as jnp

class BaseLayer:
    def __init__(self, seed=0):
        self.key = jax.random.key(seed)

    def forward(self, x):
        raise NotImplementedError("Each layer must implement the forward method.")

class NeuronLayer(BaseLayer):
    def __init__(self, n_neurons, n_neurons_next, activation=None, seed=0):
        super().__init__(seed)
        self.n_neurons = n_neurons
        self.n_neurons_next = n_neurons_next
        _, subkey = jax.random.split(self.key)
        self.W = jax.random.normal(subkey, (self.n_neurons, self.n_neurons_next))
        _, subkey = jax.random.split(subkey)
        self.x = jax.random.normal(subkey, (self.n_neurons))
        self.e = jnp.zeros((self.n_neurons))  # Initialize error to zero
        self.activation = activation if activation is not None else jax.nn.identity

    def forward(self):
        # Simple linear transformation using JAX numpy
        x = self.W.T @ self.x
        return self.activation(x)
    
    def update_weights(self, error_next):
        # Update weights using gradient descent
        self.W += jnp.kron(self.x[:, None], error_next[None, :])

    def update_error(self, signal_prev):
        # Update error based on the next layer's error
        self.e = self.x - signal_prev
        print("Updated error:", self.e.max(), self.e.min())
    
    def update_activation(self, error_next):
        self.x -= .045 * (2 * self.e - 2 * self.W @ error_next)
        print("Updated activation:", self.x.max(), self.x.min())
    
    def get_energy(self):
        return self.e @ self.e.T

class PredictiveCodingNetwork:
    def __init__(self, layer_sizes, activation=None, seed=0):
        self.layers = []
        layer_sizes = layer_sizes + [0]  # Add a dummy layer size for the output layer
        for i in range(len(layer_sizes) - 1):
            self.layers.append(NeuronLayer(layer_sizes[i], layer_sizes[i + 1], activation, seed+i))

    def forward(self):
        return self.layers[-1].x

    def update_activations(self):
        for i in reversed(range(len(self.layers))):
            print(i)
            layer = self.layers[i]
            # layer.update_weights(error_next)
            if i > 0:
                signal_prev = self.layers[i - 1].forward()
                layer.update_error(signal_prev)
            if i < len(self.layers) - 1:
                error_next = self.layers[i+1].e
                layer.update_activation(error_next)
    
    def update_weights(self):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i < len(self.layers) - 1:
                error_next = self.layers[i+1].e
                layer.update_weights(error_next)
    
    def get_energy(self):
        return sum(layer.get_energy() for layer in self.layers)