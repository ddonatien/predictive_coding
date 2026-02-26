"""Network visualizer for PredictiveCodingNetwork.

This visualizer opens a non-blocking matplotlib window on construction.
Call ``attach(network)`` to attach a ``PredictiveCodingNetwork`` instance.
Call ``step()`` repeatedly to read the current network state and update the rendering.

Neurons are drawn as circles (left: layer 0, right: last layer). Neuron
colors transition from orange (negative) -> white (zero) -> blue (positive).
Connections are colored from red (negative weights) -> white -> green (positive).
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


class NetworkVisualizer:
    def __init__(self, figsize=(8, 5)):
        plt.ion()
        # create two side-by-side axes: network (left) and energy plot (right)
        self.fig, (self.net_ax, self.energy_ax) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.18}
        )
        self.net_ax.set_aspect('equal')
        self.net_ax.axis('off')
        self.energy_ax.set_title('Energy')
        self.energy_ax.set_xlabel('Timestep')
        self.energy_ax.set_ylabel('Energy')
        self.energy_ax.grid(True)
        # use log scale for energy
        self.energy_ax.set_yscale('log')

        # energy history
        self.energy_times = []
        self.energy_values = []
        self.energy_line, = self.energy_ax.plot([], [], '-o', markersize=3)

        # text artist for timestep placed above the network axes (in axes fraction coords)
        self.timestep_text = self.net_ax.text(
            0.5,
            1.03,
            "",
            transform=self.net_ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=10,
        )

        self.network = None
        self.neuron_patches = []  # list of lists: per-layer neuron Circle artists
        self.connection_collections = []  # list of tuples: (n_src, n_dst, LineCollection)

        # colormaps
        self.act_cmap = LinearSegmentedColormap.from_list('act_orange_blue', ['#FFA500', '#FFFFFF', '#1f77b4'])
        self.w_cmap = LinearSegmentedColormap.from_list('weight_red_green', ['red', '#FFFFFF', 'green'])

        # store neuron positions to draw connections
        self.positions = []  # list of lists of (x,y)

        # a small padding so nodes don't touch borders
        self._pad = 0.08

    def attach(self, network) -> None:
        """Attach a PredictiveCodingNetwork and build the static layout.

        The method creates Circle artists for neurons and Line2D artists for
        all connections between consecutive layers. It does NOT assume weights
        or activations are plain numpy arrays (will convert when reading).
        """
        self.network = network
        self._build_layout()
        # initial render
        self.step()

    def _build_layout(self):
        # clear and reset network axes
        self.net_ax.cla()
        self.net_ax.set_aspect('equal')
        self.net_ax.axis('off')
        # recreate timestep text after clearing the axes
        self.timestep_text = self.net_ax.text(
            0.5,
            1.03,
            "",
            transform=self.net_ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=10,
        )
        # clear and reset energy axes
        self.energy_ax.cla()
        self.energy_ax.set_title('Energy')
        self.energy_ax.set_xlabel('Timestep')
        self.energy_ax.set_ylabel('Energy')
        self.energy_ax.grid(True)
        self.energy_ax.set_yscale('log')
        # reset energy history and line
        self.energy_times = []
        self.energy_values = []
        self.energy_line, = self.energy_ax.plot([], [], '-o', markersize=3)
        self.neuron_patches = []
        self.connection_collections = []
        self.positions = []

        if self.network is None:
            return

        n_layers = len(self.network.layers)
        if n_layers == 0:
            return

        # x positions from left (0) to right (1)
        xs = np.linspace(self._pad, 1 - self._pad, n_layers)

        # determine max neurons in a layer to set vertical spacing
        max_neurons = max(int(layer.n_neurons) for layer in self.network.layers)

        for i, layer in enumerate(self.network.layers):
            n = int(layer.n_neurons)
            # center vertically
            if n == 1:
                ys = np.array([0.5])
            else:
                ys = np.linspace(self._pad, 1 - self._pad, n)

            layer_positions = [(xs[i], float(y)) for y in ys]
            self.positions.append(layer_positions)

            # draw neurons
            patches = []
            for (x, y) in layer_positions:
                c = Circle((x, y), 0.02 + 0.1 / max_neurons, facecolor='#ffffff', edgecolor='none', lw=0)
                self.net_ax.add_patch(c)
                patches.append(c)

            self.neuron_patches.append(patches)

        # draw connections between layers i -> i+1 using a LineCollection per layer
        for i in range(n_layers - 1):
            src_pos = self.positions[i]
            dst_pos = self.positions[i + 1]
            segments = []
            src_n = len(src_pos)
            dst_n = len(dst_pos)
            for si, (sx, sy) in enumerate(src_pos):
                for di, (dx, dy) in enumerate(dst_pos):
                    segments.append([(sx, sy), (dx, dy)])

            # initial neutral colors
            init_colors = ['#cccccc'] * len(segments)
            lc = LineCollection(segments, linewidths=1.0, colors=init_colors, alpha=0.9)
            self.net_ax.add_collection(lc)
            self.connection_collections.append((src_n, dst_n, lc))

        self.net_ax.set_xlim(0 - 0.02, 1 + 0.02)
        self.net_ax.set_ylim(0 - 0.02, 1 + 0.02)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _map_activation_colors(self, activations: np.ndarray):
        # map activations (1d array) to colors using act_cmap centered at 0
        if activations.size == 0:
            return []
        max_abs = float(np.max(np.abs(activations)))
        if max_abs == 0:
            normed = np.zeros_like(activations, dtype=float)
        else:
            normed = (activations + max_abs) / (2 * max_abs)
        return [self.act_cmap(float(v)) for v in normed]

    def _map_neuron_colors(self, layer_idx: int, activations: np.ndarray):
        """Map neuron activations using the same colormap as weights.

        Prefer using the maximum absolute outgoing weight for normalization so
        neuron colors and connection colors share a common scale where possible.
        """
        if activations.size == 0:
            return []
        try:
            W = np.array(self.network.layers[layer_idx].W)
        except Exception:
            W = np.array([])

        max_abs_w = float(np.max(np.abs(W))) if W.size else 0.0
        max_abs_a = float(np.max(np.abs(activations))) if activations.size else 0.0

        ref = max_abs_w if max_abs_w > 0 else (max_abs_a if max_abs_a > 0 else 1.0)
        normed = (activations + ref) / (2 * ref)
        return [self.w_cmap(float(v)) for v in normed]

    def _map_weight_color(self, w: float, wmax: float):
        if wmax == 0:
            return self.w_cmap(0.5)
        norm = (w + wmax) / (2 * wmax)
        return self.w_cmap(float(norm))

    def step(self) -> None:
        """Read the attached network state and update the visualization.

        This method is non-blocking and will update the figure in-place.
        """
        if self.network is None:
            return

        # update neurons (use weight colormap for neurons)
        for li, layer in enumerate(self.network.layers):
            activations = np.array(layer.x[0])
            colors = self._map_neuron_colors(li, activations)
            for ni, patch in enumerate(self.neuron_patches[li]):
                if ni >= len(colors):
                    continue
                patch.set_facecolor(colors[ni])

        # update connections colors based on weights (LineCollection)
        for li in range(len(self.network.layers) - 1):
            try:
                W = np.array(self.network.layers[li].W)
            except Exception:
                W = np.array([])
            if W.size == 0:
                continue
            wmax = float(np.max(np.abs(W)))
            flat = W.flatten()
            colors = [self._map_weight_color(float(w), wmax) for w in flat]
            # set colors on LineCollection
            if li < len(self.connection_collections):
                _, _, lc = self.connection_collections[li]
                try:
                    lc.set_colors(colors)
                except Exception:
                    # fallback: set single color
                    pass

        # update energy history
        try:
            energy = getattr(self.network, 'get_energy', None)
            if callable(energy):
                e = float(np.array(self.network.get_energy()))
            else:
                e = None
        except Exception:
            e = None

        try:
            ts = getattr(self.network, 'timestep', None)
        except Exception:
            ts = None

        if ts is None:
            # if no timestep, use incremental index
            ts_val = len(self.energy_times)
        else:
            ts_val = ts

        if e is not None:
            # ensure positive for log scale
            eps = 1e-12
            try:
                e = float(e)
            except Exception:
                e = None
            if e is not None:
                if e <= 0:
                    e = eps
                self.energy_times.append(ts_val)
                self.energy_values.append(e)
                # update energy line data
                self.energy_line.set_data(self.energy_times, self.energy_values)
                # adjust axes limits
                if len(self.energy_times) > 0:
                    xmin = min(self.energy_times)
                    xmax = max(self.energy_times)
                    if xmin == xmax:
                        xmin -= 0.5
                        xmax += 0.5
                    self.energy_ax.set_xlim(xmin, xmax)
                if len(self.energy_values) > 0:
                    ymin = min(self.energy_values)
                    ymax = max(self.energy_values)
                    if ymin == ymax:
                        ymin = max(ymin * 0.9, eps)
                        ymax = ymax * 1.1 + eps
                    # ensure strictly positive limits for log scale
                    ymin = max(ymin, eps)
                    ymax = max(ymax, ymin * 1.000001)
                    self.energy_ax.set_ylim(ymin, ymax)

        # update timestep text (safe-get in case network has no attribute)
        try:
            if ts is None:
                txt = "Timestep: ?"
            else:
                txt = f"Timestep: {ts}"
            self.timestep_text.set_text(txt)
        except Exception:
            pass

        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            plt.pause(0.001)

    def close(self) -> None:
        try:
            plt.close(self.fig)
        except Exception:
            pass
