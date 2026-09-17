# Fly Model

`FlyModel` executes a selected graph as a sparse recurrent rate network.
Supply locally indexed connection arrays and ordered input and readout ports.
Each call starts from zero activity and returns the final readout activity.

## Load from a dataset directory

Use an explicit circuit specification to initialize a new model from local
MaleCNS data. Neuron queries can reference both annotation fields and the joined
`consensus_nt` column.

```python
from flyx.core import CircuitSpec, NeuronQuery as Q
from flyx.model import FlyConfig, FlyModel

spec = CircuitSpec(
    neurons=(
        Q.field("superclass").eq("ol_intrinsic")
        & Q.field("type").isin(["L2", "Tm1"])
        & Q.field("consensus_nt").eq("acetylcholine")
    ),
    inputs=Q.field("type").eq("L2"),
    outputs=Q.field("type").eq("Tm1"),
    selection="strongest_input_to_output",
    max_inputs=32,
    max_outputs=64,
)

core = FlyModel.from_directory(
    "data/male-cns-v1.0",
    circuit=spec,
    sign_policy={"acetylcholine": 1},
    config=FlyConfig(propagation_steps=8),
)

# Inspect original biological IDs and reuse the graph without scanning again.
body_ids = core.circuit.neuron_ids
fresh_core = FlyModel.from_circuit(
    core.circuit,
    sign_policy={"acetylcholine": 1},
    config=FlyConfig(propagation_steps=8),
)
```

`from_directory` initializes fresh parameters; it does not load a trained
checkpoint. The explicit sign mapping is an experiment assumption. Every selected
neuron must have a mapped transmitter label, including readout-only neurons.

For graph-first workflows, call `Connectome.from_directory(path).select(spec)`
and pass the resulting `Circuit` to `FlyModel.from_circuit`.

| Selection | Neurons retained | Port limits |
| --- | --- | --- |
| `induced` (default) | Every neuron matched by `neurons`, including intermediates and disconnected neurons | Not supported |
| `strongest_input_to_output` | Selected input/output neurons, ranked using direct connection counts | Optional `max_inputs` and `max_outputs` |

Both modes retain all recorded edges whose endpoints survive selection. Ranked
selection requires disjoint ports, resolves tied totals by smaller body ID, and
removes inputs that have no direct connection to a retained readout. Final neuron
and port ordering follows ascending body ID. Normalization in `FlyModel` uses the
retained graph's incoming counts.

!!! note "Counts and signs"
    Connection counts define relative incoming strengths. Signs are supplied
    explicitly; the model does not infer them from neurotransmitter annotations.
    The resulting network is an experimental model, not a pretrained classifier.

## Circuit selection

::: flyx.core.circuit
    options:
      members:
        - CircuitSpec
        - Circuit

## Model API

::: flyx.model.fly_model
    options:
      members:
        - FlyConfig
        - CircuitData
        - FlyModel
      merge_init_into_class: true
      filters:
        - "!^_"
        - "^__call__$"
