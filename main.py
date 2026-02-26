import time
import jax
import jax.numpy as jnp
from predictive_coding import PredictiveCodingNetwork, NetworkVisualizer
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Example usage of the PredictiveCodingNetwork
    layer_sizes = [3, 4, 6, 4, 4]  # Example layer sizes
    batch_size = 128
    vis = NetworkVisualizer()
    network = PredictiveCodingNetwork(layer_sizes, batch_size=batch_size)
    vis.attach(network)

    energies = []

    target_output = jax.random.normal(jax.random.PRNGKey(1), (batch_size, layer_sizes[-1]))
    strt_time = time.time()
    for i in range(500):  # Run multiple update iterations
        print(f"Iteration {i + 1}")
        network.layers[-1].x = target_output # Set the input to the first layer
        vis.step()  # Update the visualization

        # Update the network based on the error
        network.update(activations=True, weights=(i % 100 == 0))  # Update weights every 100 iterations

        energy = network.get_energy()
        energies.append(energy)
        time.sleep(0.1)  # Sleep briefly to slow down the loop for visualization
    
    vis.step()  # Final visualization update
    vis.close()  # Close the visualization window after the loop
    print(f"Avg loop time: {(time.time() - strt_time) / 100000 / 1000:.6f} ms")