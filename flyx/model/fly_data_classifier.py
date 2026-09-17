"""Classify feature vectors with learned adapters around a recurrent core.

The input projection maps dataset features to nonnegative neuron stimulation.
The core maps that stimulation to readout activity, and a linear head maps the
readout to class logits. Preprocessing, label encoding, loss computation, and
optimization are handled by the caller.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from flyx.model.fly_model import FlyModel


class ClassifierOutput(NamedTuple):
    """JAX-compatible named tuple returned by a classification forward pass.

    Attributes:
        logits: Unnormalized class scores with shape `[batch, num_classes]`.
            Apply softmax to obtain probabilities, or pass logits directly to
            a classification loss that accepts unnormalized scores.
    """

    logits: jax.Array


class FlyDataClassifier(nnx.Module):
    """Classify feature vectors using a fly-circuit recurrent core.

    Data flows through a learned linear input projection, ReLU, the supplied
    core, and a learned linear classification head. There is no direct
    connection from dataset features to the head.

    Attributes:
        num_features (int): Number of dataset features, `F`.
        num_classes (int): Number of output classes, `C`, at least two.
        core (nnx.Module): Supplied core, called with stimulation of shape
            `[batch, core.num_inputs]` and expected to return activity of
            shape `[batch, core.num_outputs]`.
        input_projection (nnx.Linear): Learned projection from `F` features
            to `core.num_inputs` stimulation values, followed by ReLU in
            `encode`.
        classifier (nnx.Linear): Learned projection from `core.num_outputs`
            readout values to `C` class logits.

    Note:
        A `FlyModel` core resets neural activity on every call, so examples
        are independent. An alternative core must provide equivalent behavior
        if independent-example classification is required; this wrapper does
        not reset arbitrary stateful cores.

    Examples:
        Build a classifier around a small circuit:

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from flax import nnx
        >>> from flyx.model import FlyConfig, FlyDataClassifier, FlyModel
        >>> core = FlyModel(
        ...     num_neurons=3,
        ...     source_indices=[0, 1],
        ...     target_indices=[1, 2],
        ...     synapse_counts=[4, 9],
        ...     edge_signs=[1, 1],
        ...     input_indices=[0],
        ...     output_indices=[2],
        ...     config=FlyConfig(propagation_steps=3),
        ... )
        >>> model = FlyDataClassifier(
        ...     core, num_features=2, num_classes=2, rngs=nnx.Rngs(42)
        ... )
        >>> x = jnp.array([[0.2, 0.8], [0.7, -0.1]])
        >>> model(x).logits.shape
        (2, 2)
        >>> model.encode(x).shape
        (2, 1)
        >>> np.testing.assert_allclose(
        ...     model.predict_proba(x).sum(axis=-1), 1.0, atol=1e-6
        ... )
        >>> model.predict(x).shape
        (2,)
    """

    def __init__(
        self,
        core: FlyModel,
        *,
        num_features: int,
        num_classes: int,
        rngs: nnx.Rngs,
    ):
        """Attach learned input and classification projections to a core.

        Args:
            core: NNX module exposing positive integer `num_inputs` and
                `num_outputs` attributes and accepting stimulation through
                `__call__`. The instance is retained directly, without copying;
                its learned parameters are part of this model's state.
            num_features: Positive integer number of input dataset features.
            num_classes: Integer number of classes, at least two. Binary
                classification uses two logits.
            rngs: NNX random-number streams used to initialize both linear
                projections. The core must already be initialized.

        Raises:
            ValueError: `num_features` is less than one or `num_classes` is
                less than two.

        Note:
            Construction does not comprehensively validate the supplied core
            interface. Its dimensions and forward-output shape must satisfy
            the documented contract.
        """
        if num_features < 1:
            raise ValueError("num_features must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.num_features = num_features
        self.num_classes = num_classes
        self.core = core

        self.input_projection = nnx.Linear(
            num_features,
            core.num_inputs,
            rngs=rngs,
        )

        self.classifier = nnx.Linear(
            core.num_outputs,
            num_classes,
            rngs=rngs,
        )

    def encode(self, x: jax.Array) -> jax.Array:
        """Project features into the core and return its readout activity.

        Args:
            x: Feature matrix with shape `[batch, num_features]`, converted
                to float32. Apply any dataset scaling before calling this
                method; values are not checked for finiteness.

        Returns:
            Readout activity with shape `[batch, core.num_outputs]`. With
            `FlyModel`, the returned array has dtype float32.

        Raises:
            ValueError: `x` is not two-dimensional or its feature dimension
                differs from `num_features`.

        Note:
            Returns the final core activity before the classification head.
            ReLU is applied to the input projection before the core is called.
        """
        x = jnp.asarray(x, dtype=jnp.float32)

        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(f"Expected [batch, {self.num_features}], got {x.shape}")

        drive = jax.nn.relu(self.input_projection(x))
        return self.core(drive)

    def __call__(self, x: jax.Array) -> ClassifierOutput:
        """Compute class logits from a batch of feature vectors.

        Args:
            x: Feature matrix with shape `[batch, num_features]`.

        Returns:
            A `ClassifierOutput` containing logits with shape
            `[batch, num_classes]`. No softmax or loss is applied.

        Raises:
            ValueError: `x` has an invalid rank or feature dimension.
        """
        features = self.encode(x)
        logits = self.classifier(features)
        return ClassifierOutput(logits=logits)

    def predict_proba(self, x: jax.Array) -> jax.Array:
        """Compute softmax probabilities over the class dimension.

        Args:
            x: Feature matrix with shape `[batch, num_features]`.

        Returns:
            Probabilities with shape `[batch, num_classes]`. For finite logits,
            each row sums to one up to numerical precision. Column `j`
            corresponds to class index `j`.

        Raises:
            ValueError: `x` has an invalid rank or feature dimension.
        """
        return jax.nn.softmax(self(x).logits, axis=-1)

    def predict(self, x: jax.Array) -> jax.Array:
        """Select the class index with the largest logit for each example.

        Args:
            x: Feature matrix with shape `[batch, num_features]`.

        Returns:
            Integer class indices with shape `[batch]`, in the range
            `[0, num_classes)`. Ties select the first maximum. The caller is
            responsible for mapping indices back to original dataset labels.

        Raises:
            ValueError: `x` has an invalid rank or feature dimension.
        """
        return jnp.argmax(self(x).logits, axis=-1)
