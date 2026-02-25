import time
import jax
import jax.numpy as jnp
from predictive_coding import PredictiveCodingNetwork
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Example usage of the PredictiveCodingNetwork
    layer_sizes = [2, 3, 4]  # Example layer sizes
    network = PredictiveCodingNetwork(layer_sizes)

    energies = []

    target_output = jax.random.normal(jax.random.PRNGKey(1), (layer_sizes[-1],))
    print("Target: ", target_output)
    strt_time = time.time()
    for i in range(100000):  # Run multiple update iterations
        print(f"Iteration {i + 1}")
        network.layers[-1].x = target_output # Set the input to the first layer
        # Forward pass
        output = network.forward()

        # Update the network based on the error
        network.update_activations()
        if i % 2 == 0:  # Update weights every 100 iterations
            network.update_weights()
        energy = network.get_energy()
        energies.append(energy)
    
    print(f"Avg loop time: {(time.time() - strt_time) / 100000 / 1000:.6f} ms")
    
    # Plot energy over iterations
    plt.plot(energies)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Energy over Iterations")
    plt.show()
    