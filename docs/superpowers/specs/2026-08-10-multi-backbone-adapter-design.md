# Multi-Backbone Adapter Refactor Design

## Context

The paper pipeline trains a causal knowledge-tracing (KT) backbone, optionally
shapes its timestep representation space with a clustering-oriented auxiliary
objective, discards the auxiliary module, and applies post-training k-means to
recover learning-state trajectories.

The paper formulation is backbone-independent, but the current implementation
is coupled to CL4KT in model construction, augmented data loading, training
losses, feature extraction, checkpoint paths, experiment orchestration, and
analysis entrypoints. AKT and simpleKT are present only as upstream reference
sources and use different input, output, loss, and checkpoint conventions.

## Objective

Refactor the repository so that CL4KT, AKT, and simpleKT run through one common
training, extraction, clustering, and analysis pipeline. A future KT backbone
must be addable by implementing its backbone or wrapper, one adapter, one model
configuration, and adapter contract tests. Adding a model must not require
changes to the generic trainer, extractor, clustering code, or analyses.

## Scope

The first implementation phase covers:

- CL4KT, preserving its contrastive learning behavior.
- AKT, including optional item-difficulty/Rasch regularization.
- simpleKT, including optional item-difficulty behavior supported by the
  reference implementation.
- The clustering-oriented joint-training configurations used in the paper.
- The existing three datasets and both official and five-fold split behavior.
- Existing extraction, pedagogical-profile, order-shuffle, stability,
  trajectory, and transition analyses.
- Backward-compatible thin entrypoints for established top-level commands where
  practical.

The refactor does not add new KT algorithms, change the paper's mathematical
objective, introduce a new configuration framework, or modify reference sources
under `KT_backbone_ref/`.

## Selected Approach

Use composition through a registered `BackboneAdapter` interface. The generic
pipeline depends only on canonical batches and outputs. Each adapter owns the
translation between this canonical contract and one model's native behavior.

Inheritance directly inside upstream models was rejected because it requires
invasive changes and makes comparison with original implementations difficult.
Using the full pyKT toolkit as the runtime backend was rejected because it would
also import its data format, trainer, configuration, and dependency conventions,
reducing control over the paper pipeline's reproducibility.

## Target Repository Structure

```text
.
├── src/
│   └── kt_state_pipeline/
│       ├── core/
│       │   ├── contracts.py
│       │   ├── capabilities.py
│       │   └── registry.py
│       ├── data/
│       │   ├── datasets.py
│       │   ├── splits.py
│       │   ├── dataloaders.py
│       │   ├── augmentations.py
│       │   └── preprocessing/
│       ├── backbones/
│       │   ├── cl4kt/
│       │   ├── akt/
│       │   └── simplekt/
│       ├── adapters/
│       │   ├── base.py
│       │   ├── cl4kt.py
│       │   ├── akt.py
│       │   └── simplekt.py
│       ├── objectives/
│       │   └── cluster_friendly.py
│       ├── engine/
│       │   ├── system.py
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   └── checkpoint.py
│       ├── discovery/
│       │   ├── extractor.py
│       │   ├── clustering.py
│       │   └── artifacts.py
│       ├── analysis/
│       │   ├── pedagogy.py
│       │   ├── trajectories.py
│       │   ├── transitions.py
│       │   ├── order_shuffle.py
│       │   └── stability.py
│       ├── experiments/
│       │   ├── runner.py
│       │   ├── stages.py
│       │   └── manifest.py
│       └── cli/
│           ├── train.py
│           ├── extract.py
│           ├── analyze.py
│           └── run_experiments.py
├── configs/
│   ├── base.yaml
│   ├── datasets/
│   ├── models/
│   ├── objectives/
│   └── experiments/
├── scripts/
├── tests/
│   ├── unit/
│   ├── adapters/
│   ├── integration/
│   ├── regression/
│   └── smoke/
├── KT_backbone_ref/
├── dataset/
├── outputs/
└── requirements.txt
```

The exact number of files may be reduced during implementation when two small
modules do not justify separate files. The dependency directions and component
boundaries are mandatory; the tree is not a requirement to create empty modules.

## Canonical Contracts

### KTBatch

The canonical batch carries padded `skills`, `items`, `responses`, and an
`attention_mask`. Padding uses skill/item ID zero and invalid responses use the
existing negative sentinel. Model-specific augmented views are optional and are
produced by a batch transform selected by adapter capabilities.

### KTOutput

Every adapter returns:

- `predictions`, `targets`, and `prediction_mask` for KT evaluation.
- Differentiable `sequence_states` with shape `[batch, time, hidden_dim]` when
  states are requested.
- `state_mask`, aligned with `sequence_states`.
- Named base-loss components, such as binary cross-entropy, CL4KT contrastive
  loss, or AKT difficulty regularization.
- Optional diagnostics such as attention weights.

Losses remain named until the system layer combines them. This makes experiment
logs auditable and avoids embedding clustering policy in a backbone.

### BackboneAdapter

Each adapter is an `nn.Module` wrapper and exposes:

```python
class BackboneAdapter(nn.Module):
    def forward(self, batch, return_states=False) -> KTOutput: ...
    def compute_base_loss(self, batch, output) -> LossBundle: ...
    def build_batch_transform(self, training: bool): ...
    def export_backbone_state(self) -> dict: ...

    @property
    def hidden_dim(self) -> int: ...
```

The registry maps a stable model name to an adapter constructor. The rest of the
pipeline must not branch on model names.

## Backbone-Specific Representation Boundaries

Representation extraction must preserve the paper's causal interpretation:
the state at timestep `t` can use the interaction history before the response at
`t` and the current skill/question context, but must not use the current response.

- CL4KT exposes the knowledge-retriever state `x` before concatenation with the
  question embedding and before the prediction MLP.
- AKT exposes the causal decoder output `d_output` before concatenation and the
  prediction MLP.
- simpleKT exposes its knowledge-retriever output `d_output` at the equivalent
  pre-head boundary.

Adapters are responsible for prediction/state offset alignment. Contract tests
must detect current-response leakage by perturbing a response and checking that
the aligned state used to predict that response remains unchanged.

## Training Data Flow

1. The data layer creates a canonical base dataset and deterministic split.
2. The registry creates the configured adapter.
3. Adapter capabilities select the train/evaluation batch transform. CL4KT uses
   augmented views during training; AKT and simpleKT use the canonical view.
4. The adapter performs its native forward pass and returns `KTOutput`.
5. The adapter computes named base KT losses.
6. When enabled, `ClusterFriendlyObjective` consumes only `sequence_states` and
   `state_mask` and returns named clustering losses.
7. The system layer combines configured weights and exposes one total loss to
   the generic trainer.
8. The evaluator consumes only canonical prediction fields.

The adapter should produce predictions and differentiable states in one forward
computation during joint training where the architecture permits it.

## Checkpoints and Artifacts

Training checkpoints record the backbone adapter state, optimizer state,
configuration snapshot, model name, dataset metadata, epoch, seed, and metric
values. The auxiliary clustering objective may be recorded for resuming training
but is excluded from the exported analysis backbone, matching the paper.

Post-training artifacts are organized as:

```text
outputs/<experiment>/<dataset>/<backbone>/<seed>/<stage>/
```

Extraction and analysis must validate the model name, hidden dimension, dataset
metadata, sequence length, and checkpoint schema before loading. Legacy CL4KT
checkpoints will be supported through an explicit compatibility loader where
their schema is unambiguous.

## Configuration

Continue using YAML and the packages already present in the repository. Split
configuration by concern:

- A base training configuration.
- Dataset-specific metadata and split configuration.
- Model-specific hyperparameters and capabilities.
- Clustering-objective weights and enabled loss terms.
- Paper experiment matrices defining datasets, models, seeds, and stages.

Configuration loading performs explicit validation and rejects unknown model
names, missing model fields, invalid loss names, and incompatible capability
requests. No Hydra or other new configuration dependency is required.

## Error Handling

Fail early with actionable errors for:

- Unknown or duplicate adapter registrations.
- Missing batch fields required by adapter capabilities.
- State and mask shape/alignment mismatches.
- Non-causal or non-timestep representations.
- Checkpoints created for a different backbone or dataset metadata.
- Unsupported legacy checkpoint layouts.
- Empty valid-state sets and too-few points for a requested cluster count.

Errors at command boundaries include the dataset, backbone, fold/seed, and stage
so failed experiment runs remain diagnosable from manifests and log files.

## Testing Strategy

### Unit tests

- Canonical batch validation and masks.
- Adapter registry behavior.
- Cluster objective terms and gradients.
- Checkpoint filtering and schema validation.
- Split reproducibility and artifact paths.

### Adapter contract tests

Run the same assertions for CL4KT, AKT, and simpleKT:

- Expected prediction, target, state, and mask shapes.
- Finite losses and gradients through both base and clustering objectives.
- Padded positions excluded from prediction and state losses.
- State alignment is causal and free of current-response leakage.
- Save/load preserves predictions and extracted states in evaluation mode.

### Regression and integration tests

- The CL4KT adapter matches a frozen baseline for output shapes, masks,
  checkpoint reload, and seeded numerical outputs within tolerance.
- A tiny end-to-end run trains, extracts states, fits k-means, and produces
  trajectory artifacts for each backbone.
- Official and five-fold dataset paths both work.
- Original and joint-training configurations both run.
- Order-shuffling reuses the frozen model and saved post-hoc k-means model.

### Full verification

After smoke tests, run the configured CL4KT/AKT/simpleKT matrix on a selected
dataset, then validate all three datasets as resources permit. Compare CL4KT AUC
and clustering artifacts against the pre-refactor baseline before declaring the
migration complete.

## Migration Sequence

1. Capture a deterministic CL4KT baseline and add regression fixtures.
2. Add canonical contracts, registry, and tests.
3. Move the clustering objective out of CL4KT and introduce the generic system
   loss composition.
4. Implement the CL4KT adapter and prove baseline equivalence.
5. Port minimally modernized AKT and simpleKT runtime implementations while
   leaving `KT_backbone_ref/` unchanged.
6. Add AKT and simpleKT adapters and causal-alignment tests.
7. Generalize training, checkpoint, extraction, and clustering entrypoints.
8. Generalize analyses and experiment orchestration, including artifact paths.
9. Add compatibility wrappers, configuration examples, and contributor docs for
   adding another backbone.
10. Run regression, adapter, integration, and smoke verification.

## Dependencies

The refactor is designed to use only the packages already listed in
`requirements.txt`. Tests will use Python's built-in `unittest` unless the
existing environment already provides another runner. If implementation reveals
a genuine runtime dependency, it must be added to `requirements.txt` with a
compatible lower bound and documented justification; development-only tooling
must not become an unnecessary runtime dependency.

## Completion Criteria

The refactor is complete when:

- The shared pipeline runs CL4KT, AKT, and simpleKT without model-name branches
  outside registration/configuration code.
- All three adapters satisfy the same causal-state and checkpoint contracts.
- The clustering objective is independent of backbone implementations.
- CL4KT regression checks pass within declared tolerances.
- A new adapter can be registered without modifying trainer, extractor,
  clustering, or analysis modules.
- Existing paper experiment stages can select a backbone through configuration
  or CLI and write collision-free artifacts.
- Documentation explains the adapter contract, representation boundary, and the
  steps required to add a fourth backbone.
