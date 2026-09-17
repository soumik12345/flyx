"""Select reusable, locally indexed circuits from a connectome dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from flyx.core.connectome import Connectome
from flyx.core.neuron_query import NeuronQuery


@dataclass(frozen=True)
class CircuitSpec:
    """Describe anatomical selection and ordered input/readout populations.

    Args:
        inputs: Query identifying input neurons within the neuron selection.
        outputs: Query identifying readout neurons within the neuron selection.
        neurons: Query over annotations joined with `consensus_nt`. When omitted,
            all annotated bodies are eligible, including non-neuronal records.
        selection: `"induced"` retains all eligible neurons and their internal
            edges. `"strongest_input_to_output"` ranks inputs by total direct
            synapse count onto eligible outputs, then ranks outputs by total
            count from retained inputs. This mode retains only selected port
            neurons and every recorded edge between them; intermediates are
            excluded. Ties prefer the smaller biological body ID.
        max_inputs: Optional positive input limit for ranked selection.
        max_outputs: Optional positive output limit for ranked selection.

    Raises:
        TypeError: A selector is not a `NeuronQuery`.
        ValueError: The mode or a limit is invalid, or limits are used with
            induced selection.
    """

    inputs: NeuronQuery
    outputs: NeuronQuery
    neurons: NeuronQuery | None = None
    selection: Literal["induced", "strongest_input_to_output"] = "induced"
    max_inputs: int | None = None
    max_outputs: int | None = None

    def __post_init__(self):
        for name in ("inputs", "outputs", "neurons"):
            query = getattr(self, name)
            if name == "neurons" and query is None:
                continue
            if not isinstance(query, NeuronQuery):
                raise TypeError(f"{name} must be a NeuronQuery")
        if self.selection not in {"induced", "strongest_input_to_output"}:
            raise ValueError(f"Unsupported circuit selection: {self.selection}")
        for name in ("max_inputs", "max_outputs"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer or None")
                if self.selection == "induced":
                    raise ValueError(
                        "Port limits require strongest_input_to_output selection"
                    )


@dataclass(frozen=True, eq=False)
class Circuit:
    """A resolved anatomical graph with read-only host arrays and provenance.

    Construct through `Connectome.select` or `Circuit.from_connectome`. Biological
    IDs remain int64 NumPy arrays on the host; only local int32 indices enter the
    neural computation. Array inputs are copied and marked read-only. Circuit
    objects compare by identity, allowing them to remain static NNX metadata.

    Attributes:
        dataset_id: Source dataset identifier.
        directory: Resolved source dataset directory.
        spec: Selection recipe used to construct the graph.
        neuron_ids: Sorted biological body IDs, shape `[N]`.
        neurotransmitters: Consensus transmitter labels in neuron order;
            missing annotations are represented as `None`.
        source_indices: Local source indices, shape `[E]`.
        target_indices: Local destination indices, shape `[E]`.
        synapse_counts: Recorded positive synapse counts, shape `[E]`.
        input_indices: Input indices ordered by biological body ID.
        output_indices: Readout indices ordered by biological body ID.
        candidate_neurons: Eligible neuron count before ranked selection.
        candidate_edges: Eligible edge count before ranked selection.
    """

    dataset_id: str
    directory: str
    spec: CircuitSpec
    neuron_ids: np.ndarray
    neurotransmitters: tuple[str | None, ...]
    source_indices: np.ndarray
    target_indices: np.ndarray
    synapse_counts: np.ndarray
    input_indices: np.ndarray
    output_indices: np.ndarray
    candidate_neurons: int
    candidate_edges: int

    def __post_init__(self):
        for name in (
            "neuron_ids",
            "source_indices",
            "target_indices",
            "synapse_counts",
            "input_indices",
            "output_indices",
        ):
            array = np.array(getattr(self, name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "neurotransmitters", tuple(self.neurotransmitters))

    @property
    def num_neurons(self) -> int:
        """Return the number of retained neurons."""
        return len(self.neuron_ids)

    @property
    def num_edges(self) -> int:
        """Return the number of retained connection records."""
        return len(self.source_indices)

    @property
    def input_body_ids(self) -> np.ndarray:
        """Return original body IDs in input-column order."""
        return self.neuron_ids[self.input_indices]

    @property
    def output_body_ids(self) -> np.ndarray:
        """Return original body IDs in readout-column order."""
        return self.neuron_ids[self.output_indices]

    def to_arrays(self) -> dict:
        """Return constructor arrays without inferring numerical edge signs.

        Returns:
            A new dictionary containing `num_neurons`, local connection and
            port indices, and `synapse_counts`. Arrays are read-only host data.
        """
        return {
            "num_neurons": self.num_neurons,
            **{
                name: getattr(self, name)
                for name in (
                    "source_indices",
                    "target_indices",
                    "synapse_counts",
                    "input_indices",
                    "output_indices",
                )
            },
        }

    @classmethod
    def from_connectome(cls, connectome: Connectome, spec: CircuitSpec) -> Circuit:
        """Resolve queries and stream retained edges from local Feather files.

        Args:
            connectome: Validated dataset handle.
            spec: Anatomical and port selection recipe. All queries operate on
                the joined annotation table, restricted by `spec.neurons` first.

        Returns:
            A circuit preserving original IDs and every edge between the
            resolved neurons. No bypass edges are added.

        Raises:
            TypeError: `spec` is not a `CircuitSpec`.
            KeyError: A query references an absent annotation column.
            ValueError: IDs are missing or duplicated, annotation columns
                conflict, a population is empty, retained weights are invalid,
                or ranked ports overlap or have no direct connections.

        Note:
            Induced selection permits disconnected neurons, empty edge sets,
            and overlapping ports. Ranked selection requires disjoint ports and
            gives every retained port neuron a direct input-to-output connection.
            Edge scans are batched, but all retained edges are held in memory.
        """
        if not isinstance(spec, CircuitSpec):
            raise TypeError("spec must be a CircuitSpec")
        annotations = feather.read_table(connectome.annotations_path).to_pandas()
        if annotations.bodyId.isna().any() or not annotations.bodyId.is_unique:
            raise ValueError("Annotation bodyId values must be nonmissing and unique")
        if "consensus_nt" in annotations.columns:
            raise ValueError(
                "consensus_nt is reserved for neurotransmitter annotations"
            )
        nt = feather.read_table(
            connectome.neurotransmitters_path, columns=["body", "consensus_nt"]
        )
        nt = nt.filter(pc.is_in(nt["body"], value_set=pa.array(annotations.bodyId)))
        nt_frame = nt.to_pandas().rename(columns={"body": "bodyId"})
        if not nt_frame.bodyId.is_unique:
            raise ValueError("Neurotransmitter body IDs must be unique")
        neurons = annotations.merge(
            nt_frame, on="bodyId", how="left", validate="one_to_one"
        )
        if spec.neurons is not None:
            neurons = neurons.loc[spec.neurons.evaluate(neurons)]
        if neurons.empty:
            raise ValueError("Neuron selection is empty")
        input_ids = neurons.loc[spec.inputs.evaluate(neurons), "bodyId"].to_numpy()
        output_ids = neurons.loc[spec.outputs.evaluate(neurons), "bodyId"].to_numpy()
        if not len(input_ids) or not len(output_ids):
            raise ValueError("Input and output selections must both be nonempty")
        if (
            spec.selection == "strongest_input_to_output"
            and np.intersect1d(input_ids, output_ids).size
        ):
            raise ValueError("Ranked input and output populations must be disjoint")
        allowed = pa.array(neurons.bodyId, type=pa.int64())
        pieces = []
        with pa.memory_map(connectome.connections_path, "r") as source:
            reader = pa.ipc.open_file(source)
            for index in range(reader.num_record_batches):
                batch = reader.get_batch(index)
                mask = pc.and_(
                    pc.is_in(batch["body_pre"], value_set=allowed),
                    pc.is_in(batch["body_post"], value_set=allowed),
                )
                selected = batch.filter(mask)
                if selected.num_rows:
                    pieces.append(selected.to_pandas())
        edges = (
            pd.concat(pieces, ignore_index=True)
            if pieces
            else pd.DataFrame(
                {
                    name: pd.Series(dtype="int64")
                    for name in ("body_pre", "body_post", "weight")
                }
            )
        )
        if edges.weight.isna().any() or (edges.weight <= 0).any():
            raise ValueError("Retained synapse counts must be positive and nonmissing")
        candidate_neurons, candidate_edges = len(neurons), len(edges)
        if spec.selection == "strongest_input_to_output":
            forward = edges.loc[
                edges.body_pre.isin(input_ids) & edges.body_post.isin(output_ids)
            ]
            if forward.empty:
                raise ValueError("No direct input-to-output connections were found")
            input_ids = _rank(forward, "body_pre", spec.max_inputs)
            output_ids = _rank(
                forward.loc[forward.body_pre.isin(input_ids)],
                "body_post",
                spec.max_outputs,
            )
            linked = forward.loc[
                forward.body_pre.isin(input_ids) & forward.body_post.isin(output_ids)
            ]
            input_ids = input_ids[np.isin(input_ids, linked.body_pre)]
            neurons = neurons.loc[
                neurons.bodyId.isin(np.concatenate([input_ids, output_ids]))
            ]
            edges = edges.loc[
                edges.body_pre.isin(neurons.bodyId)
                & edges.body_post.isin(neurons.bodyId)
            ]
        neurons = neurons.sort_values("bodyId")
        neuron_ids = neurons.bodyId.to_numpy(dtype=np.int64)
        edges = edges.sort_values(["body_pre", "body_post"], kind="stable")
        return cls(
            dataset_id=connectome.dataset_id,
            directory=connectome.directory,
            spec=spec,
            neuron_ids=neuron_ids,
            neurotransmitters=tuple(
                None if pd.isna(value) else value for value in neurons.consensus_nt
            ),
            source_indices=np.searchsorted(neuron_ids, edges.body_pre).astype(np.int32),
            target_indices=np.searchsorted(neuron_ids, edges.body_post).astype(
                np.int32
            ),
            synapse_counts=edges.weight.to_numpy(dtype=np.int64),
            input_indices=np.searchsorted(neuron_ids, np.sort(input_ids)).astype(
                np.int32
            ),
            output_indices=np.searchsorted(neuron_ids, np.sort(output_ids)).astype(
                np.int32
            ),
            candidate_neurons=candidate_neurons,
            candidate_edges=candidate_edges,
        )


def _rank(edges: pd.DataFrame, column: str, limit: int | None) -> np.ndarray:
    """Rank direct synapse totals, breaking ties by ascending body ID."""
    totals = (
        edges.groupby(column, sort=True)
        .weight.sum()
        .sort_values(ascending=False, kind="stable")
    )
    return (totals if limit is None else totals.head(limit)).index.to_numpy()
