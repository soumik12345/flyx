from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest
from flax import nnx

from flyx.core import Circuit, CircuitSpec, Connectome, NeuronQuery as Q
from flyx.model import FlyConfig, FlyModel

FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "transmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "connections": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
OFFSET = 2**35
POLICY = {"acetylcholine": 1, "gaba": -1}


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    tables = {
        "annotations": pa.table(
            {
                "bodyId": [OFFSET + i for i in [40, 20, 60, 10, 50, 30]],
                "type": ["output", "input", "other", "input", "middle", "output"],
                "superclass": [
                    "visual",
                    "visual",
                    "other",
                    "visual",
                    "visual",
                    "visual",
                ],
            }
        ),
        "transmitters": pa.table(
            {
                "body": [OFFSET + i for i in [10, 20, 30, 40, 50, 60]],
                "consensus_nt": [
                    "acetylcholine",
                    "acetylcholine",
                    "gaba",
                    "acetylcholine",
                    "gaba",
                    "unclear",
                ],
            }
        ),
        "connections": pa.table(
            {
                "body_pre": [
                    OFFSET + i for i in [10, 20, 10, 20, 30, 10, 50, 40, 60, 10]
                ],
                "body_post": [
                    OFFSET + i for i in [30, 30, 40, 40, 10, 50, 40, 30, 10, 60]
                ],
                "weight": [5, 5, 5, 5, 2, 7, 3, 1, 99, 99],
            }
        ),
    }
    for name, table in tables.items():
        # Multiple record batches exercise the streaming loader.
        feather.write_feather(table, tmp_path / FILES[name], chunksize=2)
    return tmp_path


@pytest.fixture
def spec() -> CircuitSpec:
    return CircuitSpec(
        neurons=Q.field("superclass").eq("visual"),
        inputs=Q.field("type").eq("input"),
        outputs=Q.field("type").eq("output"),
    )


def resolved_circuit(model: FlyModel) -> Circuit:
    """Return the anatomical graph attached by a FlyModel factory."""
    circuit = model.circuit
    assert circuit is not None
    return circuit


def test_induced_circuit_preserves_intermediates_ids_counts_and_ports(
    dataset_dir, spec
):
    circuit = Connectome.from_directory(dataset_dir).select(spec)
    np.testing.assert_array_equal(
        circuit.neuron_ids, OFFSET + np.array([10, 20, 30, 40, 50])
    )
    np.testing.assert_array_equal(circuit.input_body_ids, OFFSET + np.array([10, 20]))
    np.testing.assert_array_equal(circuit.output_body_ids, OFFSET + np.array([30, 40]))
    actual = list(
        zip(
            circuit.neuron_ids[circuit.source_indices] - OFFSET,
            circuit.neuron_ids[circuit.target_indices] - OFFSET,
            circuit.synapse_counts,
        )
    )
    assert actual == [
        (10, 30, 5),
        (10, 40, 5),
        (10, 50, 7),
        (20, 30, 5),
        (20, 40, 5),
        (30, 10, 2),
        (40, 30, 1),
        (50, 40, 3),
    ]
    assert circuit.dataset_id == "male-cns:v1.0"
    assert circuit.directory == str(dataset_dir.resolve())
    assert circuit.spec is spec
    assert circuit.neuron_ids.dtype == np.int64
    assert circuit.source_indices.dtype == np.int32
    assert circuit.candidate_neurons == 5
    assert circuit.candidate_edges == 8
    with pytest.raises(ValueError):
        circuit.neuron_ids[0] = 0


def test_directory_and_circuit_factories_agree_and_apply_presynaptic_signs(
    dataset_dir, spec
):
    core = FlyModel.from_directory(dataset_dir, circuit=spec, sign_policy=POLICY)
    circuit = resolved_circuit(core)
    reused = FlyModel.from_circuit(circuit, sign_policy=POLICY)
    assert reused.circuit is circuit
    np.testing.assert_allclose(
        core.base_weights[...], [5 / 11, 5 / 13, 1, 5 / 11, 5 / 13, -1, 1 / 11, -3 / 13]
    )
    drive = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(core(drive), reused(drive))
    np.testing.assert_allclose(nnx.jit(lambda m, x: m(x))(core, drive), core(drive))
    # Training one instance must not modify another instance's parameters.
    core.gain_logits[...] = jnp.ones(core.num_neurons)
    np.testing.assert_array_equal(reused.gain_logits[...], 0.0)
    grads = nnx.grad(lambda m: jnp.sum(m(drive)))(reused)
    assert np.any(np.asarray(grads.gain_logits[...]) != 0)


def test_ranked_selection_breaks_ties_by_id_and_keeps_reverse_edges(dataset_dir, spec):
    ranked = replace(
        spec, selection="strongest_input_to_output", max_inputs=1, max_outputs=1
    )
    core = FlyModel.from_directory(dataset_dir, circuit=ranked, sign_policy=POLICY)
    circuit = resolved_circuit(core)
    np.testing.assert_array_equal(circuit.neuron_ids, OFFSET + np.array([10, 30]))
    np.testing.assert_array_equal(circuit.synapse_counts, [5, 2])
    np.testing.assert_array_equal(core.base_weights[...], [1.0, -1.0])


def test_ranked_selection_without_limits_keeps_only_ports(dataset_dir, spec):
    ranked = replace(spec, selection="strongest_input_to_output")
    circuit = Connectome.from_directory(dataset_dir).select(ranked)
    np.testing.assert_array_equal(
        circuit.neuron_ids, OFFSET + np.array([10, 20, 30, 40])
    )
    assert circuit.num_edges == 6


def test_joined_transmitter_query_filters_neurons_before_port_selection(
    dataset_dir, spec
):
    neurons = spec.neurons
    assert neurons is not None
    filtered = replace(
        spec, neurons=neurons & Q.field("consensus_nt").eq("acetylcholine")
    )
    core = FlyModel.from_directory(
        dataset_dir, circuit=filtered, sign_policy={"acetylcholine": 1}
    )
    circuit = resolved_circuit(core)
    np.testing.assert_array_equal(circuit.neuron_ids, OFFSET + np.array([10, 20, 40]))
    np.testing.assert_array_equal(circuit.output_body_ids, [OFFSET + 40])


def test_sign_policy_is_copied_and_configuration_is_used(dataset_dir, spec):
    policy = dict(POLICY)
    config = FlyConfig(propagation_steps=1)
    core = FlyModel.from_directory(
        dataset_dir, circuit=spec, sign_policy=policy, config=config
    )
    policy["gaba"] = 1
    assert core.sign_policy == (("acetylcholine", 1), ("gaba", -1))
    assert core.config is config
    np.testing.assert_array_equal(core(jnp.ones((1, 2))), [[0.0, 0.0]])


@pytest.mark.parametrize("policy", [{"acetylcholine": 1}, {}])
def test_unmapped_transmitters_raise(dataset_dir, spec, policy):
    with pytest.raises(ValueError, match="no mapping.*gaba"):
        FlyModel.from_directory(dataset_dir, circuit=spec, sign_policy=policy)


def test_missing_transmitter_annotation_is_not_silently_dropped(dataset_dir, spec):
    path = dataset_dir / FILES["transmitters"]
    table = feather.read_table(path)
    feather.write_feather(table.slice(1), path)
    with pytest.raises(ValueError, match="no mapping.*None"):
        FlyModel.from_directory(dataset_dir, circuit=spec, sign_policy=POLICY)


@pytest.mark.parametrize(
    "policy", [{"gaba": 0}, {"gaba": True}, {"gaba": float("nan")}, {None: 1}]
)
def test_invalid_sign_policy_is_rejected_before_io(tmp_path, spec, policy):
    with pytest.raises(ValueError, match="sign_policy"):
        FlyModel.from_directory(
            tmp_path / "absent",
            circuit=spec,
            sign_policy=cast(Mapping[str, int], policy),
        )


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"neurons": Q.field("type").eq("absent")}, "Neuron selection is empty"),
        ({"inputs": Q.field("type").eq("absent")}, "both be nonempty"),
        ({"outputs": Q.field("type").eq("other")}, "both be nonempty"),
        (
            {
                "selection": "strongest_input_to_output",
                "outputs": Q.field("type").eq("input"),
            },
            "disjoint",
        ),
        (
            {
                "selection": "strongest_input_to_output",
                "inputs": Q.field("type").eq("middle"),
                "outputs": Q.field("type").eq("input"),
            },
            "No direct",
        ),
    ],
)
def test_invalid_population_selection_raises(dataset_dir, spec, changes, error):
    with pytest.raises(ValueError, match=error):
        Connectome.from_directory(dataset_dir).select(replace(spec, **changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"selection": "random"},
        {"max_inputs": 2},
        {"selection": "strongest_input_to_output", "max_inputs": 0},
        {"selection": "strongest_input_to_output", "max_outputs": True},
    ],
)
def test_invalid_spec_options_are_rejected(spec, changes):
    with pytest.raises(ValueError):
        replace(spec, **changes)


def test_missing_query_column_raises(dataset_dir, spec):
    with pytest.raises(KeyError, match="absent"):
        Connectome.from_directory(dataset_dir).select(
            replace(spec, neurons=Q.field("absent").eq("value"))
        )


def test_induced_selection_allows_empty_edges_and_overlapping_ports(dataset_dir):
    query = Q.field("type").eq("middle")
    core = FlyModel.from_directory(
        dataset_dir,
        circuit=CircuitSpec(neurons=query, inputs=query, outputs=query),
        sign_policy=POLICY,
        config=FlyConfig(propagation_steps=1, update_rate=1.0),
    )
    assert resolved_circuit(core).num_edges == 0
    np.testing.assert_array_equal(core(jnp.array([[2.0]])), [[2.0]])


@pytest.mark.parametrize("file", ["annotations", "transmitters"])
def test_duplicate_body_ids_raise(dataset_dir, spec, file):
    path = dataset_dir / FILES[file]
    table = feather.read_table(path)
    feather.write_feather(pa.concat_tables([table, table.slice(0, 1)]), path)
    with pytest.raises(ValueError, match="unique"):
        Connectome.from_directory(dataset_dir).select(spec)


def test_nonpositive_connection_count_raises(dataset_dir, spec):
    path = dataset_dir / FILES["connections"]
    table = feather.read_table(path)
    table = table.set_column(
        2, "weight", pa.array([0] + table["weight"].to_pylist()[1:])
    )
    feather.write_feather(table, path)
    with pytest.raises(ValueError, match="positive"):
        Connectome.from_directory(dataset_dir).select(spec)
