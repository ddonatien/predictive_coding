import jax
import jax.numpy as jnp

class BaseLayer:
    def __init__(self, seed=0):
        self.key = jax.random.key(seed)

    def forward(self, x):
        raise NotImplementedError("Each layer must implement the forward method.")

class NeuronLayer(BaseLayer):
    def __init__(self, n_neurons, n_neurons_next, batch_size=1, activation=None, seed=0):
        super().__init__(seed)
        self.n_neurons = n_neurons
        self.n_neurons_next = n_neurons_next
        _, subkey = jax.random.split(self.key)
        self.W = 0.1 * jax.random.normal(subkey, (self.n_neurons, self.n_neurons_next))
        _, subkey = jax.random.split(subkey)
        self.x = 0.1 * jax.random.normal(subkey, (batch_size, self.n_neurons))
        self.e = jnp.zeros((batch_size, self.n_neurons))  # Initialize error to zero
        self.activation = activation if activation is not None else jax.nn.identity
        # Store activation derivative function; use jax.grad for automatic differentiation
        self._activation_grad = jax.vmap(jax.grad(lambda xi: jnp.sum(self.activation(xi))))

    def forward(self):
        # Simple linear transformation using JAX numpy
        return self.activation(self.x) @ self.W
    
    def update_weights(self, error_next, learning_rate=.15):
        # Update weights using gradient descent
        self.W += learning_rate * (self.activation(self.x)[:, :, None] * error_next[:, None, :]).mean(axis=0)

    def update_error(self, signal_prev):
        # Update error based on the next layer's error
        self.e = self.x - signal_prev
    
    def _get_activation_derivative(self):
        """Compute f'(x) for each neuron, where f is the activation function."""
        # vmap over batch dimension, grad computes d/dx sum(f(x))
        # For a single neuron, this gives f'(x)
        return self._activation_grad(self.x)

    def update_activation(self, error_next, learning_rate=0.15):
        # Apply activation derivative only to the backpropagated error term
        f_prime = self._get_activation_derivative()
        backprop_error = (error_next @ self.W.T) * f_prime
        self.x -= learning_rate * (self.e - backprop_error)
    
    def get_energy(self):
        return jax.lax.batch_matmul(self.e, self.e.T).sum()

class PredictiveCodingNetwork:
    def __init__(self, layer_sizes, batch_size=1, activation=None, seed=0):
        self.timestep = 0
        self.layers = []
        layer_sizes = layer_sizes + [0]  # Add a dummy layer size for the output layer
        for i in range(len(layer_sizes) - 1):
            self.layers.append(NeuronLayer(layer_sizes[i], layer_sizes[i + 1], batch_size, activation, seed+i))

    def forward(self):
        return self.layers[-1].x

    def update(self, activations: bool =True, weights: bool =True):
        if activations:
            self._update_activations()
        if weights:
            self._update_weights()
        self.timestep += 1
    
    def _update_activations(self):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            # layer.update_weights(error_next)
            if i > 0:
                signal_prev = self.layers[i - 1].forward()
                layer.update_error(signal_prev)
            if i < len(self.layers) - 1:
                error_next = self.layers[i+1].e
                layer.update_activation(error_next)
    
    def _update_weights(self):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i < len(self.layers) - 1:
                error_next = self.layers[i+1].e
                layer.update_weights(error_next)
    
    def get_energy(self):
        return sum(layer.get_energy() for layer in self.layers)