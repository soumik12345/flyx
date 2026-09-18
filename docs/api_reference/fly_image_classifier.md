# Fly Image Classifier

`FlyImageClassifier` maps images to spatially localized neuron stimulation,
propagates that activity through a supplied `FlyModel`, and decodes the final
readout into class logits. All image information passes through the circuit.

| Component | Responsibility |
| --- | --- |
| `VisualInputMap` | Immutable host metadata linking biological IDs to image locations |
| `RetinotopicEncoder` | Bilinear sampling and learned cell-type-shared local responses |
| `ImageClassificationHead` | Linear decoding of circuit activity into logits |
| `FlyImageClassifier` | Composition, input-ID alignment, and prediction helpers |

## Image and mapping contract

Images have shape `[batch, height, width, channels]`. Normalize images before
calling the model; casting to float32 does not rescale integer pixels.
The default is one grayscale channel. Multiple channels are mixed locally by
learned weights shared across neurons of the same type.

Supply normalized `(y, x)` image coordinates in `[0, 1]`: `(0, 0)` is the top-left
pixel center and `(1, 1)` is the bottom-right pixel center. The encoder uses
bilinear interpolation and supports singleton height or width. It learns a
channel mixture and bias per neuron type, followed by softplus; initial weights
average channels and initial biases are zero. Thus zero-valued pixels still
produce a positive baseline drive. This is a modeling choice, not a measured
physiological response.

Each encoder supports one or both eyes viewing the same image. Choose each
eye's circuit inputs and image-coordinate transform explicitly;
the model does not infer mirroring, project anatomical hex coordinates, select
visual neurons, or synthesize missing mappings. Optional anatomical column
coordinates and a provenance string document the supplied projection. Response
parameters are shared by neuron type across eyes. Separate left/right images
per example are not supported by this shared-image interface.

The classifier aligns mapping records to `core.circuit.input_body_ids` when the
core carries a circuit. Directly constructed cores require explicit
`input_body_ids` in core input order. Extra mapping records are ignored; missing
input mappings are errors. Biological IDs remain int64 host data. Sampling
coordinates and type indices are nontrainable `CircuitData` buffers.

## Example

This synthetic circuit demonstrates the API without requiring a dataset:

```python
import jax.numpy as jnp
from flax import nnx

from flyx.model import FlyConfig, FlyImageClassifier, FlyModel, VisualInputMap

core = FlyModel(
    num_neurons=4,
    source_indices=[0, 1],
    target_indices=[2, 3],
    synapse_counts=[1, 1],
    edge_signs=[1, 1],
    input_indices=[0, 1],
    output_indices=[2, 3],
    config=FlyConfig(propagation_steps=2),
)
input_map = VisualInputMap(
    body_ids=[101, 102],
    sampling_coordinates=[[0.25, 0.25], [0.75, 0.75]],
    neuron_types=("L1", "L1"),
    eyes=("left", "left"),
    provenance="Synthetic locations; y downward and x rightward.",
)
model = FlyImageClassifier(
    core,
    input_map,
    input_body_ids=[101, 102],
    num_classes=10,
    rngs=nnx.Rngs(42),
)
images = jnp.ones((8, 28, 28, 1), dtype=jnp.float32)
logits = model(images).logits  # (8, 10)
activity = model.encode(images)  # (8, 2)
drive = model.stimulate(images)  # (8, 2)
```

For anatomical experiments, select mapped visual inputs and downstream readouts
with a `CircuitSpec`. Use induced selection to retain intermediate neurons.
Choose sufficient propagation depth: an input needs at least `d + 1` updates to
reach a readout `d` edges away. Connectivity alone does not guarantee activity;
inhibitory paths and rectification can suppress responses.

## Training and prediction

The API mirrors `FlyDataClassifier`: `model(images)` returns `ClassifierOutput`,
`encode` returns circuit activity, `predict_proba` returns class probabilities,
and `predict` returns integer class indices. `stimulate` additionally exposes
ordered input-neuron stimulation. Each call resets core activity.

Use logits directly with a classification loss and optimize `nnx.Param` to train
the encoder response, core gains/biases, and head while excluding anatomical
buffers. Preprocessing, augmentation, labels, and optimization remain external.

::: flyx.model.fly_image_classifier
    options:
      merge_init_into_class: true
      filters:
        - "!^_"
        - "^__call__$"
