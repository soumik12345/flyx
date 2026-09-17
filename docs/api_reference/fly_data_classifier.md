# Fly Data Classifier

`FlyDataClassifier` maps feature vectors to class logits through a learned
input projection, a recurrent core, and a learned classification head.

| Operation | Input shape | Output shape |
| --- | --- | --- |
| `model(x)` | `[batch, num_features]` | `ClassifierOutput` with logits `[batch, num_classes]` |
| `model.encode(x)` | `[batch, num_features]` | `[batch, core.num_outputs]` |
| `model.predict_proba(x)` | `[batch, num_features]` | `[batch, num_classes]` |
| `model.predict(x)` | `[batch, num_features]` | `[batch]` |

Use a [FlyModel](fly_model.md) core to reset neural activity for every call.
Preprocessing, label encoding, training, and evaluation remain caller-managed.
For an integer-label cross-entropy loss, encode labels as class indices from
`0` to `num_classes - 1` and pass `model(x).logits` directly to the loss.

::: flyx.model.fly_data_classifier
    options:
      members:
        - ClassifierOutput
        - FlyDataClassifier
      merge_init_into_class: true
      filters:
        - "!^_"
        - "^__call__$"
