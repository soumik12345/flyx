"""Execute a selected circuit as a sparse recurrent rate network.

`FlyModel` accepts locally indexed graph arrays, applies constant stimulation
to selected input neurons, and returns the final activity of selected readout
neurons. Each call starts with zero activity. One scalar represents each
neuron's activity for each batch example.

Connection counts determine fixed incoming-normalized weight proportions.
Training adjusts a gain per source neuron and a bias per neuron. The supplied
signs remain fixed; the model does not infer signs from neurotransmitter data.
These dynamics are a modeling choice, not a simulation of biological spikes
or a pretrained task model.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def _indices(values, *, name, size, nonempty=False):
    """Validate local neuron indices and convert them to NumPy int32.

    Args:
        values (numpy.typing.ArrayLike): One-dimensional indices. Nonempty
            arrays must have an integer dtype; floating-point and boolean
            indices are rejected before conversion.
        name (str): Array name used in validation error messages.
        size (int): Exclusive upper bound for valid indices.
        nonempty (bool): Whether an empty array should raise an error.

    Returns:
        numpy.ndarray: One-dimensional int32 indices in the supplied order.

    Raises:
        ValueError: The array is not one-dimensional, is empty when forbidden,
            has a noninteger dtype when nonempty, or contains an index outside
            `[0, size)`.
    """
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if nonempty and values.size == 0:
        raise ValueError(f"{name} must not be empty")

    if values.size:
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must contain integers")

        if np.any(values < 0) or np.any(values >= size):
            raise ValueError(f"{name} must contain indices in [0, {size})")

    return values.astype(np.int32)


@dataclass(frozen=True)
class FlyConfig:
    """Immutable configuration for recurrent activity updates.

    Attributes:
        propagation_steps (int): Positive Python integer specifying the
            number of internal updates per call. Defaults to `8`.
        update_rate (float): Fraction of the new target activity used in each
            update, in `(0, 1]`. The remaining fraction retains the previous
            activity. Defaults to `0.5`.
        max_gain (float): Upper bound used to scale sigmoid-transformed
            source-neuron gains, in `(0, 1)`. Defaults to `0.9`; zero-initialized
            gain logits therefore give initial gains of `0.45`.

    Raises:
        ValueError: The step count is not a positive Python integer, is a
            boolean, or either numerical setting is outside its valid range.

    Examples:
        >>> from flyx.model import FlyConfig
        >>> config = FlyConfig(propagation_steps=3, update_rate=1.0)
        >>> config.max_gain
        0.9
    """

    propagation_steps: int = 8
    update_rate: float = 0.5
    max_gain: float = 0.9

    def __post_init__(self):
        """Validate configuration values after dataclass initialization.

        Raises:
            ValueError: A setting violates the constraints of `FlyConfig`.
        """
        if isinstance(self.propagation_steps, bool) or not isinstance(
            self.propagation_steps, int
        ):
            raise ValueError("propagation_steps must be an integer")

        if self.propagation_steps < 1:
            raise ValueError("propagation_steps must be positive")

        if not 0 < self.update_rate <= 1:
            raise ValueError("update_rate must be in (0, 1]")

        if not 0 < self.max_gain < 1:
            raise ValueError("max_gain must be in (0, 1)")


class CircuitData(nnx.Variable):
    """Store circuit arrays separately from trainable parameters.

    Use for connection indices, port indices, and fixed base weights.
    These values are excluded from optimization when the optimizer
    selects only `nnx.Param`.

    Note:
        This type identifies graph data; it does not make arrays immutable.

    Examples:
        Wrap an array and retrieve its value:

        >>> import jax.numpy as jnp
        >>> from flyx.model.fly_model import CircuitData
        >>> indices = CircuitData(jnp.array([0, 2], dtype=jnp.int32))
        >>> indices[...].tolist()
        [0, 2]

        Use `nnx.state(model, CircuitData)` to select graph buffers and
        `nnx.state(model, nnx.Param)` to select learned parameters.
    """


class FlyModel(nnx.Module):
    """Propagate stimulation through a fixed directed graph.

    For each edge, divide its count by the total incoming count at its
    destination and multiply by its supplied sign. During execution, multiply
    this base weight by `max_gain * sigmoid(gain_logits[source])`.

    Starting from zero activity, each update computes
    `h_next = (1 - alpha) * h + alpha * relu(recurrent + external + bias)`,
    where `alpha` is `config.update_rate`. External stimulation remains
    constant throughout the updates. Only the final readout activity is
    returned; no activity is retained between calls.

    Attributes:
        config (FlyConfig): Recurrent dynamics configuration.
        num_neurons (int): Number of locally indexed neurons, `N`.
        num_inputs (int): Number of input neurons, `S`.
        num_outputs (int): Number of readout neurons, `R`, rather than the
            number of task classes.
        source_indices (CircuitData): int32 source indices with shape `[E]`.
        target_indices (CircuitData): int32 destination indices with shape `[E]`.
        input_indices (CircuitData): int32 input indices with shape `[S]`.
        output_indices (CircuitData): int32 readout indices with shape `[R]`.
        base_weights (CircuitData): Fixed float32 signed, incoming-normalized
            weights with shape `[E]`.
        gain_logits (nnx.Param): Learned float32 source-neuron gain logits
            with shape `[N]`, initialized to zero.
        bias (nnx.Param): Learned float32 neuron biases with shape `[N]`,
            initialized to zero.

    Note:
        Input-dependent activity requires at least `d + 1` updates to travel
        along a path of `d` edges from an input neuron. Construction does not
        validate reachability or choose a suitable propagation depth.

    Note:
        Execution materializes messages of shape `[batch, E]` and activity of
        shape `[batch, N]`. Differentiating through the updates can require
        additional memory. No dense `[N, N]` adjacency matrix is constructed.

    Examples:
        A three-neuron chain reaches its readout after three updates:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from flyx.model import FlyConfig, FlyModel
        >>> core = FlyModel(
        ...     num_neurons=3,
        ...     source_indices=[0, 1],
        ...     target_indices=[1, 2],
        ...     synapse_counts=[4, 9],
        ...     edge_signs=[1, 1],
        ...     input_indices=[0],
        ...     output_indices=[2],
        ...     config=FlyConfig(propagation_steps=3, update_rate=1.0),
        ... )
        >>> readout = core(jnp.array([[1.0], [2.0]]))
        >>> readout.shape
        (2, 1)
        >>> np.testing.assert_allclose(
        ...     readout, [[0.2025], [0.405]], atol=1e-6
        ... )
    """

    def __init__(
        self,
        *,
        num_neurons: int,
        source_indices,
        target_indices,
        synapse_counts,
        edge_signs,
        input_indices,
        output_indices,
        config: FlyConfig = FlyConfig(),
    ):
        """Validate graph arrays and initialize buffers and learned parameters.

        Args:
            num_neurons: Positive neuron count representable as int32.
                All indices refer to positions in `[0, num_neurons)`, not
                biological body IDs.
            source_indices (numpy.typing.ArrayLike): One-dimensional integer
                source indices, with shape `[E]`.
            target_indices (numpy.typing.ArrayLike): One-dimensional integer
                destination indices, with shape `[E]`.
            synapse_counts (numpy.typing.ArrayLike): Finite positive counts
                with shape `[E]`. Positive noninteger values are also accepted.
            edge_signs (numpy.typing.ArrayLike): Explicit signs with shape
                `[E]`, each exactly `-1` or `+1`. Resolve uncertain signs before
                construction.
            input_indices (numpy.typing.ArrayLike): Nonempty, unique integer
                indices with shape `[S]`. Their order defines drive columns.
            output_indices (numpy.typing.ArrayLike): Nonempty, unique integer
                indices with shape `[R]`. Their order defines output columns.
                Input and output sets may overlap.
            config: Configuration controlling the recurrent updates.

        Raises:
            ValueError: The neuron count is invalid; an index array has an
                invalid shape, dtype, or range; a port is empty or has duplicate
                indices; edge-array shapes differ; counts are nonfinite or
                nonpositive; a sign is invalid; or incoming totals overflow.

        Note:
            Empty edge sets, self-connections, and repeated edges are allowed.
            Repeated edges contribute separately to incoming totals and message
            sums. Normalization uses only the supplied graph. Construction is
            deterministic and requires no random-number generator.
        """
        if isinstance(num_neurons, bool) or not isinstance(
            num_neurons, (int, np.integer)
        ):
            raise ValueError("num_neurons must be an integer")

        if not 1 <= num_neurons <= np.iinfo(np.int32).max:
            raise ValueError("num_neurons must fit in a positive int32")

        n = int(num_neurons)

        src = _indices(source_indices, name="source_indices", size=n)
        dst = _indices(target_indices, name="target_indices", size=n)
        inputs = _indices(
            input_indices,
            name="input_indices",
            size=n,
            nonempty=True,
        )
        outputs = _indices(
            output_indices,
            name="output_indices",
            size=n,
            nonempty=True,
        )

        for name, indices in [
            ("input_indices", inputs),
            ("output_indices", outputs),
        ]:
            if np.unique(indices).size != indices.size:
                raise ValueError(f"{name} must not contain duplicates")

        counts = np.asarray(synapse_counts, dtype=np.float64)
        signs = np.asarray(edge_signs, dtype=np.float64)

        if not (src.shape == dst.shape == counts.shape == signs.shape):
            raise ValueError("All edge arrays must have the same one-dimensional shape")

        if not np.all(np.isfinite(counts) & (counts > 0)):
            raise ValueError("synapse_counts must be finite and positive")

        if not np.all(np.isin(signs, [-1.0, 1.0])):
            raise ValueError("edge_signs must contain only -1 or +1")

        incoming = np.bincount(
            dst,
            weights=counts,
            minlength=n,
        )

        if not np.all(np.isfinite(incoming)):
            raise ValueError("Incoming synapse totals overflowed")

        base = signs * counts / incoming[dst]

        self.config = config
        self.num_neurons = n
        self.num_inputs = len(inputs)
        self.num_outputs = len(outputs)

        self.source_indices = CircuitData(jnp.asarray(src))
        self.target_indices = CircuitData(jnp.asarray(dst))
        self.input_indices = CircuitData(jnp.asarray(inputs))
        self.output_indices = CircuitData(jnp.asarray(outputs))
        self.base_weights = CircuitData(jnp.asarray(base, dtype=jnp.float32))

        self.gain_logits = nnx.Param(jnp.zeros(n, dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros(n, dtype=jnp.float32))

    def __call__(self, drive: jax.Array) -> jax.Array:
        """Return final readout activity for independent batch examples.

        Args:
            drive: Stimulation with shape `[batch, num_inputs]`, converted to
                float32. Column `j` stimulates `input_indices[j]` at every
                update. Negative stimulation is accepted; values are not
                checked for finiteness.

        Returns:
            Float32 activity with shape `[batch, num_outputs]`, in the order
            specified by `output_indices`.

        Raises:
            ValueError: The drive is not two-dimensional or its feature
                dimension differs from `num_inputs`.

        Note:
            Each call allocates zero initial activity. Learned parameters
            persist, but neural activity does not. The computation is
            differentiable through all configured updates and can be called
            inside `nnx.jit`.
        """
        drive = jnp.asarray(drive, dtype=jnp.float32)

        if drive.ndim != 2 or drive.shape[1] != self.num_inputs:
            raise ValueError(f"Expected [batch, {self.num_inputs}], got {drive.shape}")

        # Extract NNX values before entering the JAX loop.
        src = self.source_indices[...]
        dst = self.target_indices[...]
        inputs = self.input_indices[...]
        outputs = self.output_indices[...]
        base = self.base_weights[...]
        bias = self.bias[...]

        gains = self.config.max_gain * jax.nn.sigmoid(self.gain_logits[...])
        weights = base * gains[src]

        n = self.num_neurons
        alpha = self.config.update_rate

        # Activity belongs to this call, not to persistent model state.
        h0 = jnp.zeros(
            (drive.shape[0], n),
            dtype=jnp.float32,
        )

        external = jnp.zeros_like(h0).at[:, inputs].set(drive)

        def step(activity, _):
            # Gather source activity for every retained edge.
            messages = activity[:, src] * weights[None, :]

            # Sum messages at their destination neurons.
            recurrent = jax.ops.segment_sum(
                messages.T,
                dst,
                num_segments=n,
            ).T

            target = jax.nn.relu(recurrent + external + bias)
            activity = (1.0 - alpha) * activity + alpha * target

            return activity, None

        final_activity, _ = jax.lax.scan(
            step,
            h0,
            xs=None,
            length=self.config.propagation_steps,
        )

        return final_activity[:, outputs]
