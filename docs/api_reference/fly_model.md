# Fly Model

`FlyModel` executes a selected graph as a sparse recurrent rate network.
Supply locally indexed connection arrays and ordered input and readout ports.
Each call starts from zero activity and returns the final readout activity.

!!! note "Counts and signs"
    Connection counts define relative incoming strengths. Signs are supplied
    explicitly; the model does not infer them from neurotransmitter annotations.
    The resulting network is an experimental model, not a pretrained classifier.

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
