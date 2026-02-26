import time
import jax
from tqdm import tqdm
import jax.numpy as jnp
from predictive_coding import PredictiveCodingNetwork, NetworkVisualizer
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Example usage of the PredictiveCodingNetwork
    layer_sizes = [4, 6, 6, 6, 12, 4]  # Example layer sizes
    batch_size = 256
    vis = NetworkVisualizer()
    network = PredictiveCodingNetwork(layer_sizes, batch_size=batch_size, activation=jax.nn.leaky_relu)
    vis.attach(network)

    energies = []

    target_output = jax.random.normal(jax.random.PRNGKey(1), (batch_size, layer_sizes[-1]))
    target_output /= jnp.linalg.norm(target_output, axis=1, keepdims=True)  # Normalize target output
    strt_time = time.time()
    n_iterations = 100*10000 # Number of update iterations
    for i in tqdm(range(50)):  # Run multiple update iterations
        network.layers[-1].x = target_output # Set the input to the first layer
        vis.step()  # Update the visualization

        # Update the network based on the error
        network.update(activations=True, weights=False)  # Update weights every 10 iterations

    for i in tqdm(range(n_iterations)):  # Run multiple update iterations
        network.layers[-1].x = target_output # Set the input to the first layer
        vis.step()  # Update the visualization

        # Update the network based on the error
        network.update(activations=True, weights=(i % 10 == 0))  # Update weights every 10 iterations

        energy = network.get_energy()
        energies.append(energy)
        # time.sleep(0.01)  # Sleep briefly to slow down the loop for visualization
    
    vis.step()  # Final visualization update
    vis.close()  # Close the visualization window after the loop
    print(f"Avg loop time: {(time.time() - strt_time) / n_iterations / 1000:.6f} ms")
    print("Final energy:", energies[-1])