"""Portable spatial mapping and end-to-end classifier regression tests."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from flyx.model import (
    FlyConfig,
    FlyImageClassifier,
    FlyModel,
    ImageClassificationHead,
    RetinotopicEncoder,
    VisualInputMap,
)
from flyx.model.fly_model import CircuitData


def make_map(**overrides):
    arguments = dict(
        body_ids=[2**40 + 2, 2**40 + 1],
        sampling_coordinates=[[0, 0], [1, 1]],
        neuron_types=("L1", "L2"),
        eyes=("left", "left"),
        column_coordinates=[[10, 20], [11, 21]],
        provenance="Synthetic left eye; y increases downward, x rightward.",
    )
    arguments.update(overrides)
    return VisualInputMap(**arguments)


def make_core():
    return FlyModel(
        num_neurons=4,
        source_indices=[0, 1],
        target_indices=[2, 3],
        synapse_counts=[1, 1],
        edge_signs=[1, 1],
        input_indices=[0, 1],
        output_indices=[2, 3],
        config=FlyConfig(propagation_steps=2, update_rate=1),
    )


def make_model():
    return FlyImageClassifier(
        make_core(),
        make_map(),
        input_body_ids=[2**40 + 1, 2**40 + 2],
        num_classes=3,
        rngs=nnx.Rngs(0),
    )


def test_mapping_copies_arrays_and_aligns_all_metadata():
    coordinates = np.array([[0.0, 0.0], [1.0, 1.0]])
    mapping = make_map(sampling_coordinates=coordinates)
    coordinates[:] = 0.5
    aligned = mapping.aligned_to([2**40 + 1, 2**40 + 2])
    np.testing.assert_array_equal(aligned.sampling_coordinates, [[1, 1], [0, 0]])
    np.testing.assert_array_equal(aligned.column_coordinates, [[11, 21], [10, 20]])
    assert aligned.neuron_types == ("L2", "L1")
    assert aligned.provenance == mapping.provenance
    assert aligned.body_ids.dtype == np.int64
    with pytest.raises(ValueError):
        mapping.sampling_coordinates[0, 0] = 1
    with pytest.raises(FrozenInstanceError):
        mapping.provenance = "changed"
    assert len(mapping.aligned_to([2**40 + 1]).body_ids) == 1
    with pytest.raises(ValueError, match="Missing visual mappings"):
        mapping.aligned_to([99])


@pytest.mark.parametrize(
    "overrides",
    [
        {"body_ids": []},
        {"body_ids": [1, 1]},
        {"body_ids": [1.0, 2.0]},
        {"body_ids": [True, False]},
        {"body_ids": [-1, 2]},
        {"body_ids": np.array([2**63, 2**63 + 1], dtype=np.uint64)},
        {"sampling_coordinates": [[0, 0]]},
        {"sampling_coordinates": [[np.nan, 0], [1, 1]]},
        {"sampling_coordinates": [[0, -0.1], [1, 1]]},
        {"sampling_coordinates": [[0, 0], [1.1, 1]]},
        {"column_coordinates": [[0, 0], [np.inf, 1]]},
        {"neuron_types": ("L1", "")},
        {"neuron_types": "L1"},
        {"eyes": ("left",)},
        {"eyes": ("left", "unknown")},
    ],
)
def test_mapping_rejects_invalid_metadata(overrides):
    with pytest.raises(ValueError):
        make_map(**overrides)


def test_bilinear_sampling_matches_coordinate_ramp_and_edges():
    mapping = VisualInputMap(
        body_ids=[1, 2, 3, 4],
        sampling_coordinates=[[0, 0], [1, 1], [0.25, 0.75], [1, 0]],
        neuron_types=("L1",) * 4,
        eyes=("left",) * 4,
    )
    encoder = RetinotopicEncoder(mapping)
    # Intensity = 10*y + x in pixel coordinates on a nonsquare image.
    ramp = (10 * np.arange(3)[:, None] + np.arange(5)[None, :])[None, ..., None]
    expected = np.array([0, 24, 8, 20])
    np.testing.assert_allclose(encoder.sample(ramp)[0, :, 0], expected)
    np.testing.assert_allclose(encoder(ramp)[0], jax.nn.softplus(expected), rtol=1e-6)


@pytest.mark.parametrize("shape", [(2, 1, 1, 1), (2, 1, 5, 1), (2, 5, 1, 1)])
def test_sampling_supports_singleton_spatial_dimensions(shape):
    encoder = RetinotopicEncoder(make_map())
    np.testing.assert_allclose(encoder.sample(jnp.full(shape, 0.7)), 0.7)


def test_local_bright_spot_and_type_shared_channel_response():
    mapping = make_map(neuron_types=("L1", "L1"))
    encoder = RetinotopicEncoder(mapping, in_channels=3)
    images = jnp.zeros((1, 2, 2, 3)).at[0, 0, 0, :].set(jnp.array([1, 2, 3]))
    np.testing.assert_allclose(encoder.sample(images), [[[1, 2, 3], [0, 0, 0]]])
    assert encoder.channel_weights[...].shape == (1, 3)
    np.testing.assert_allclose(encoder(images), jax.nn.softplus(jnp.array([[2, 0]])))


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (1, 0, 3, 1), (1, 3, 3, 2)])
def test_encoder_rejects_invalid_image_shapes(shape):
    with pytest.raises(ValueError, match="Expected"):
        RetinotopicEncoder(make_map())(jnp.zeros(shape))


def test_encoder_samples_shared_image_at_each_eyes_coordinates():
    encoder = RetinotopicEncoder(
        make_map(eyes=("left", "right"), neuron_types=("L2", "L2"))
    )
    images = jnp.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    np.testing.assert_allclose(encoder.sample(images), [[[1.0], [4.0]]])
    np.testing.assert_allclose(
        encoder(images), jax.nn.softplus(jnp.array([[1.0, 4.0]]))
    )
    assert encoder.channel_weights[...].shape == (1, 1)


def test_classifier_aligns_both_eyes_and_runs_under_jit():
    model = FlyImageClassifier(
        make_core(),
        make_map(eyes=("left", "right")),
        input_body_ids=[2**40 + 1, 2**40 + 2],
        num_classes=3,
        rngs=nnx.Rngs(0),
    )
    assert model.encoder.input_map.eyes == ("right", "left")
    images = jnp.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    np.testing.assert_allclose(
        model.stimulate(images), jax.nn.softplus(jnp.array([[4.0, 1.0]]))
    )
    logits = nnx.jit(lambda m, x: m(x).logits)(model, images)
    np.testing.assert_allclose(logits, model(images).logits)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_encoder_rejects_invalid_channel_counts(value):
    with pytest.raises(ValueError, match="positive integer"):
        RetinotopicEncoder(make_map(), in_channels=value)


def test_classifier_aligns_inputs_and_matches_hand_computed_core():
    model = make_model()
    images = jnp.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    stimulation = jax.nn.softplus(jnp.array([[4.0, 1.0]]))
    np.testing.assert_allclose(model.stimulate(images), stimulation)
    np.testing.assert_allclose(model.encode(images), stimulation * 0.45, rtol=1e-6)
    np.testing.assert_allclose(model(images).logits, model.head(stimulation * 0.45))
    assert model(images).logits.shape == (1, 3)
    np.testing.assert_allclose(model.predict_proba(images).sum(-1), 1, atol=1e-6)
    np.testing.assert_array_equal(
        model.predict(images), model(images).logits.argmax(-1)
    )


def test_classifier_jit_and_batch_independence():
    model = make_model()
    images = jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2, 1) / 12
    logits = model(images).logits
    np.testing.assert_allclose(nnx.jit(lambda m, x: m(x).logits)(model, images), logits)
    separately = jnp.concatenate([model(images[i : i + 1]).logits for i in range(3)])
    np.testing.assert_allclose(logits, separately)
    np.testing.assert_allclose(model(images).logits, logits)


def test_classifier_requires_matching_input_ids():
    core = make_core()
    with pytest.raises(ValueError, match="required"):
        FlyImageClassifier(core, make_map(), num_classes=2, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="length"):
        FlyImageClassifier(
            core, make_map(), input_body_ids=[1], num_classes=2, rngs=nnx.Rngs(0)
        )
    # Only the Circuit's public port metadata is needed to infer mapping order.
    core.circuit = SimpleNamespace(input_body_ids=np.array([2**40 + 1, 2**40 + 2]))
    model = FlyImageClassifier(core, make_map(), num_classes=2, rngs=nnx.Rngs(0))
    np.testing.assert_array_equal(
        model.encoder.input_map.body_ids, core.circuit.input_body_ids
    )
    with pytest.raises(ValueError, match="do not match"):
        FlyImageClassifier(
            core,
            make_map(),
            input_body_ids=make_map().body_ids,
            num_classes=2,
            rngs=nnx.Rngs(0),
        )


def test_head_validates_dimensions_and_activity():
    for features, classes in [(0, 2), (2, 1), (True, 2), (2, 2.5)]:
        with pytest.raises(ValueError):
            ImageClassificationHead(
                num_features=features, num_classes=classes, rngs=nnx.Rngs(0)
            )
    head = ImageClassificationHead(num_features=2, num_classes=3, rngs=nnx.Rngs(0))
    for activity in (jnp.ones(2), jnp.ones((1, 3))):
        with pytest.raises(ValueError, match="Expected"):
            head(activity)


def test_training_reaches_all_components_and_preserves_fixed_buffers():
    model = make_model()
    images = jnp.array(
        [[[[0.1], [0.2]], [[0.3], [0.8]]], [[[0.9], [0.3]], [[0.2], [0.1]]]]
    )
    labels = jnp.array([0, 1])
    fixed_before = [np.array(x) for x in jax.tree.leaves(nnx.state(model, CircuitData))]
    params_before = [np.array(x) for x in jax.tree.leaves(nnx.state(model, nnx.Param))]

    def loss_fn(m):
        return optax.softmax_cross_entropy_with_integer_labels(
            m(images).logits, labels
        ).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    assert np.isfinite(loss)
    for name in ("encoder", "core", "head"):
        leaves = jax.tree.leaves(grads[name])
        assert all(np.isfinite(x).all() for x in leaves)
        assert any(np.any(np.asarray(x) != 0) for x in leaves)
    optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)
    optimizer.update(model, grads)
    fixed_after = jax.tree.leaves(nnx.state(model, CircuitData))
    for before, after in zip(fixed_before, fixed_after, strict=True):
        np.testing.assert_array_equal(before, after)
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            params_before, jax.tree.leaves(nnx.state(model, nnx.Param)), strict=True
        )
    )
    assert float(loss_fn(model)) < float(loss)
