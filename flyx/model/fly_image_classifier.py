"""Classify images through spatially mapped stimulation of a fly circuit.

Image coordinates are supplied explicitly: anatomical column coordinates do
not establish an image's orientation or visual-field calibration. The encoder
samples a shared image for one or both eyes without mixing spatial locations.
Preprocessing, circuit selection, label encoding, and training are caller-owned.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from numpy.typing import ArrayLike

from flyx.model.fly_data_classifier import ClassifierOutput
from flyx.model.fly_model import CircuitData, FlyModel


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _body_ids(values: ArrayLike) -> np.ndarray:
    ids = np.asarray(values)
    if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
        raise ValueError("body_ids must be a nonempty one-dimensional integer array")
    if np.any(ids < 0) or np.any(ids > np.iinfo(np.int64).max):
        raise ValueError("body_ids must be nonnegative int64 values")
    ids = np.array(ids, dtype=np.int64, copy=True)
    if np.unique(ids).size != ids.size:
        raise ValueError("body_ids must be unique")
    ids.setflags(write=False)
    return ids


@dataclass(frozen=True, eq=False)
class VisualInputMap:
    """Host metadata linking biological input neurons to image locations.

    Arrays are copied and marked read-only. Identity equality allows the map
    to remain static NNX metadata. Original int64 body IDs never enter JAX.

    Args:
        body_ids: Unique biological IDs, shape `[S]`, in mapping order.
        sampling_coordinates: Finite normalized `(y, x)` locations, shape
            `[S, 2]`, in `[0, 1]`. Zero is the top/left pixel center; one is
            the bottom/right pixel center. These are image coordinates, not
            raw anatomical hex coordinates.
        neuron_types: Nonempty cell-type labels, one per neuron. Response
            parameters are shared by neurons with the same label.
        eyes: Eye labels, `"left"` or `"right"`, one per neuron. Both eyes
            sample the same image at their explicitly supplied coordinates.
        column_coordinates: Optional finite anatomical coordinates `[S, 2]`,
            retained as provenance and not used directly for sampling.
        provenance: Description of the dataset, mapping source, and image
            coordinate transform, including orientation and scale.

    Raises:
        ValueError: IDs, coordinates, or labels violate the mapping contract.
    """

    body_ids: ArrayLike
    sampling_coordinates: ArrayLike
    neuron_types: tuple[str, ...]
    eyes: tuple[str, ...]
    column_coordinates: ArrayLike | None = None
    provenance: str = ""

    def __post_init__(self) -> None:
        ids = _body_ids(self.body_ids)
        object.__setattr__(self, "body_ids", ids)
        for name in ("sampling_coordinates", "column_coordinates"):
            values = getattr(self, name)
            if values is None and name == "column_coordinates":
                continue
            coordinates = np.array(values, dtype=np.float64, copy=True)
            if coordinates.shape != (len(ids), 2) or not np.isfinite(coordinates).all():
                raise ValueError(f"{name} must be finite with shape [{len(ids)}, 2]")
            if name == "sampling_coordinates" and np.any(
                (coordinates < 0) | (coordinates > 1)
            ):
                raise ValueError("sampling_coordinates must be in [0, 1]")
            coordinates.setflags(write=False)
            object.__setattr__(self, name, coordinates)
        for name in ("neuron_types", "eyes"):
            raw = getattr(self, name)
            if isinstance(raw, str):
                raise ValueError(f"{name} must contain one label per neuron")
            labels = tuple(raw)
            if len(labels) != len(ids) or any(
                not isinstance(label, str) or not label.strip() for label in labels
            ):
                raise ValueError(f"{name} must contain one nonempty label per neuron")
            object.__setattr__(self, name, labels)
        if not set(self.eyes) <= {"left", "right"}:
            raise ValueError("eyes must contain only 'left' or 'right'")
        if not isinstance(self.provenance, str):
            raise ValueError("provenance must be a string")

    def aligned_to(self, body_ids: ArrayLike) -> VisualInputMap:
        """Select and reorder records to match explicit circuit input IDs.

        Extra mapping records are ignored. Missing IDs raise `ValueError`;
        no stimulation location is inferred or silently filled in.
        """
        requested = _body_ids(body_ids)
        stored_ids = np.asarray(self.body_ids)
        sampling_coordinates = np.asarray(self.sampling_coordinates)
        column_coordinates = (
            None
            if self.column_coordinates is None
            else np.asarray(self.column_coordinates)
        )
        lookup = {int(body_id): i for i, body_id in enumerate(stored_ids)}
        missing = [int(body_id) for body_id in requested if int(body_id) not in lookup]
        if missing:
            raise ValueError(f"Missing visual mappings for input body IDs: {missing}")
        indices = np.array([lookup[int(body_id)] for body_id in requested])
        return VisualInputMap(
            body_ids=requested,
            sampling_coordinates=sampling_coordinates[indices],
            neuron_types=tuple(self.neuron_types[i] for i in indices),
            eyes=tuple(self.eyes[i] for i in indices),
            column_coordinates=(
                None if column_coordinates is None else column_coordinates[indices]
            ),
            provenance=self.provenance,
        )


class RetinotopicEncoder(nnx.Module):
    """Sample images locally and produce nonnegative input-neuron activity.

    Bilinear sampling preserves the supplied mapping order. A learned channel
    mixture and bias are shared by cell type, followed by softplus. This is
    an engineered response function, not a measured physiological model.
    Initial channel weights average channels and initial biases are zero.
    Coordinates and type indices are nontrainable `CircuitData` buffers.

    Args:
        input_map: Ordered mapping to image coordinates for one or both eyes.
            Both eyes see the same image. Supply eye-specific coordinates;
            no mirroring or binocular image routing is inferred. Response
            parameters remain shared by neuron type across eyes.
        in_channels: Positive number of image channels. Multiple channels
            are mixed locally; RGB channels are not biological receptor types.
    """

    def __init__(self, input_map: VisualInputMap, *, in_channels: int = 1):
        if not isinstance(input_map, VisualInputMap):
            raise TypeError("input_map must be a VisualInputMap")
        self.in_channels = _positive_integer(in_channels, "in_channels")
        self.input_map = input_map
        self.neuron_types = tuple(sorted(set(input_map.neuron_types)))
        type_lookup = {label: i for i, label in enumerate(self.neuron_types)}
        self.coordinates = CircuitData(
            jnp.asarray(input_map.sampling_coordinates, dtype=jnp.float32)
        )
        self.type_indices = CircuitData(
            jnp.array([type_lookup[t] for t in input_map.neuron_types], dtype=jnp.int32)
        )
        self.channel_weights = nnx.Param(
            jnp.full(
                (len(self.neuron_types), self.in_channels),
                1.0 / self.in_channels,
                dtype=jnp.float32,
            )
        )
        self.bias = nnx.Param(jnp.zeros(len(self.neuron_types), dtype=jnp.float32))

    @property
    def num_inputs(self) -> int:
        """Return the number of input neurons in the retained mapping."""
        return int(np.asarray(self.input_map.body_ids).shape[0])

    def sample(self, images: jax.Array) -> jax.Array:
        """Return bilinear samples `[batch, num_inputs, in_channels]`.

        Images must have shape `[batch, height, width, in_channels]` with
        positive spatial dimensions. Values are cast to float32; scaling,
        normalization, and finite-value checks are the caller's responsibility.
        Singleton spatial dimensions are supported. Sampling at endpoints
        returns the corresponding edge pixel, without padding or extrapolation.

        Raises:
            ValueError: Image rank, spatial dimensions, or channels are invalid.
        """
        images = jnp.asarray(images, dtype=jnp.float32)
        if (
            images.ndim != 4
            or images.shape[-1] != self.in_channels
            or images.shape[1] < 1
            or images.shape[2] < 1
        ):
            raise ValueError(
                f"Expected [batch, height > 0, width > 0, {self.in_channels}], "
                f"got {images.shape}"
            )
        height, width = images.shape[1:3]
        positions = self.coordinates[...] * jnp.array([height - 1, width - 1])
        low = jnp.floor(positions).astype(jnp.int32)
        high = jnp.minimum(low + 1, jnp.array([height - 1, width - 1]))
        dy, dx = (positions - low).T
        dy, dx = dy[None, :, None], dx[None, :, None]
        top = (1 - dx) * images[:, low[:, 0], low[:, 1], :] + dx * images[
            :, low[:, 0], high[:, 1], :
        ]
        bottom = (1 - dx) * images[:, high[:, 0], low[:, 1], :] + dx * images[
            :, high[:, 0], high[:, 1], :
        ]
        return (1 - dy) * top + dy * bottom

    def __call__(self, images: jax.Array) -> jax.Array:
        """Return stimulation `[batch, num_inputs]` in mapping order."""
        samples = self.sample(images)
        types = self.type_indices[...]
        response = jnp.sum(samples * self.channel_weights[...][types], axis=-1)
        return jax.nn.softplus(response + self.bias[...][types])


class ImageClassificationHead(nnx.Module):
    """Decode circuit activity into unnormalized class logits.

    Args:
        num_features: Positive number of circuit readout neurons.
        num_classes: Number of classes, at least two; binary tasks use two logits.
        rngs: Random streams for initializing the linear decoder.
    """

    def __init__(self, *, num_features: int, num_classes: int, rngs: nnx.Rngs):
        self.num_features = _positive_integer(num_features, "num_features")
        self.num_classes = _positive_integer(num_classes, "num_classes")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.linear = nnx.Linear(self.num_features, self.num_classes, rngs=rngs)

    def __call__(self, activity: jax.Array) -> jax.Array:
        """Map `[batch, num_features]` to `[batch, num_classes]`.

        Raises:
            ValueError: Activity rank or feature count is invalid.
        """
        activity = jnp.asarray(activity, dtype=jnp.float32)
        if activity.ndim != 2 or activity.shape[1] != self.num_features:
            raise ValueError(
                f"Expected [batch, {self.num_features}], got {activity.shape}"
            )
        return self.linear(activity)


class FlyImageClassifier(nnx.Module):
    """Compose a spatial encoder, supplied fly circuit, and linear decoder.

    All image information passes through the circuit. The supplied core is
    retained directly and resets activity on every call. Its graph and the
    encoder's geometry remain separate from trainable `nnx.Param` state.
    Circuit selection, transmitter signs, and sufficient propagation depth
    are the caller's responsibility; this class does not select visual neurons.

    Args:
        core: Initialized fly circuit with ordered input and readout neurons.
        input_map: Visual metadata, automatically aligned to core input IDs.
        num_classes: Number of classes, at least two.
        rngs: Random streams for the classification head.
        in_channels: Number of image channels, default one (grayscale).
        input_body_ids: Biological IDs in core input order, required for a core
            built directly from arrays. For a core carrying a `Circuit`, IDs
            are inferred; an explicit argument must match that circuit exactly.

    Raises:
        ValueError: Input IDs are unavailable, inconsistent, or lack mappings;
            dimensions are invalid.
    """

    # Keep construction explicit and compatible; optional settings are keyword-only.
    def __init__(  # pylint: disable=too-many-arguments
        self,
        core: FlyModel,
        input_map: VisualInputMap,
        *,
        num_classes: int,
        rngs: nnx.Rngs,
        in_channels: int = 1,
        input_body_ids: ArrayLike | None = None,
    ):
        if core.circuit is not None:
            circuit_ids = core.circuit.input_body_ids
            if input_body_ids is not None and not np.array_equal(
                _body_ids(input_body_ids), circuit_ids
            ):
                raise ValueError("input_body_ids do not match core.circuit input order")
            input_body_ids = circuit_ids
        elif input_body_ids is None:
            raise ValueError("input_body_ids are required for a core without a Circuit")
        ids = _body_ids(input_body_ids)
        if len(ids) != core.num_inputs:
            raise ValueError("input_body_ids length must equal core.num_inputs")
        self.encoder = RetinotopicEncoder(
            input_map.aligned_to(ids), in_channels=in_channels
        )
        self.core = core
        self.head = ImageClassificationHead(
            num_features=core.num_outputs, num_classes=num_classes, rngs=rngs
        )
        self.num_classes = self.head.num_classes
        self.in_channels = self.encoder.in_channels

    def stimulate(self, images: jax.Array) -> jax.Array:
        """Return `[batch, core.num_inputs]` stimulation in core input order."""
        return self.encoder(images)

    def encode(self, images: jax.Array) -> jax.Array:
        """Return final circuit readout activity `[batch, core.num_outputs]`."""
        return self.core(self.stimulate(images))

    def __call__(self, images: jax.Array) -> ClassifierOutput:
        """Return `ClassifierOutput` with logits `[batch, num_classes]`."""
        return ClassifierOutput(logits=self.head(self.encode(images)))

    def predict_proba(self, images: jax.Array) -> jax.Array:
        """Return class probabilities `[batch, num_classes]`."""
        return jax.nn.softmax(self(images).logits, axis=-1)

    def predict(self, images: jax.Array) -> jax.Array:
        """Return integer class indices `[batch]`, choosing the first tied max."""
        return jnp.argmax(self(images).logits, axis=-1)
