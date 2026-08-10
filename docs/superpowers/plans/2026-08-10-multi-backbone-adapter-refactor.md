# Multi-Backbone Adapter Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the paper pipeline so CL4KT, AKT, and simpleKT share one causal-state training, extraction, clustering, and analysis workflow through registered backbone adapters.

**Architecture:** A canonical `KTBatch`/`KTOutput` contract separates the generic pipeline from model-native inputs and losses. Registered adapters wrap minimally changed backbone implementations, while a `KTSystem` composes each adapter's base loss with a backbone-independent clustering objective; thin legacy entrypoints delegate to package CLI modules.

**Tech Stack:** Python 3, PyTorch, Accelerate, NumPy, pandas, scikit-learn, PyYAML, joblib, matplotlib, seaborn, and standard-library `unittest`.

## Global Constraints

- Runtime dependencies remain limited to the packages already listed in `requirements.txt` unless a verified implementation need requires an explicit compatible lower bound.
- Tests use standard-library `unittest`; do not add pytest solely for this refactor.
- Preserve the paper objective: the cluster-friendly module shapes training representations but is not the final post-training clusterer.
- A state at timestep `t` must not contain response `r_t`; it may contain earlier interactions and current question/skill context.
- Keep `KT_backbone_ref/` unchanged and never import it from the runtime package.
- Preserve user data and unrelated working-tree changes.
- Keep established top-level commands usable through compatibility wrappers during migration.

---

## File Map

The implementation creates `src/kt_state_pipeline/` as the runtime package. The
top-level scripts remain import-compatible wrappers until downstream users have
migrated.

- `core/contracts.py`: canonical batch, output, and named-loss value objects.
- `core/capabilities.py`: immutable declarations of adapter data requirements.
- `core/registry.py`: adapter registration and construction without model-name branches.
- `adapters/base.py`: abstract adapter interface and shared validation.
- `objectives/cluster_friendly.py`: independent clustering-oriented loss.
- `engine/system.py`: base-loss and joint-loss composition.
- `backbones/*`: runtime CL4KT, AKT, and simpleKT implementations.
- `adapters/*`: model-specific input/output/loss translation.
- `data/*`: canonical datasets, splits, CL4KT augmentation, and loader creation.
- `engine/trainer.py`, `engine/evaluator.py`, `engine/checkpoint.py`: generic execution.
- `discovery/*`: model-independent state extraction and post-hoc k-means.
- `analysis/*`: shared state inference used by trajectories and order shuffling.
- `experiments/*`: model-aware commands, manifests, and artifact paths.
- `cli/*`: package command entrypoints.
- `configs/models/*.yaml`: model defaults.
- `tests/*`: unit, adapter-contract, integration, regression, and smoke coverage.

### Task 1: Canonical contracts and adapter registry

**Files:**
- Create: `src/kt_state_pipeline/__init__.py`
- Create: `src/kt_state_pipeline/core/__init__.py`
- Create: `src/kt_state_pipeline/core/contracts.py`
- Create: `src/kt_state_pipeline/core/capabilities.py`
- Create: `src/kt_state_pipeline/core/registry.py`
- Create: `src/kt_state_pipeline/adapters/__init__.py`
- Create: `src/kt_state_pipeline/adapters/base.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/test_contracts.py`
- Create: `tests/unit/test_registry.py`

**Interfaces:**
- Produces: `KTBatch.from_mapping(mapping)`, `KTBatch.to(device)`, `KTOutput.validate()`, `LossBundle.total()`, `AdapterCapabilities`, `BackboneAdapter`, `register_adapter(name)`, `create_adapter(name, **kwargs)`, and `registered_adapters()`.
- Consumes: PyTorch tensors and modules only.

- [ ] **Step 1: Write failing contract tests**

```python
class ContractTests(unittest.TestCase):
    def test_batch_round_trip_and_output_validation(self):
        batch = KTBatch.from_mapping({
            "questions": torch.tensor([[1, 2]]),
            "skills": torch.tensor([[3, 4]]),
            "responses": torch.tensor([[0, 1]]),
            "attention_mask": torch.tensor([[1, 1]]),
        })
        self.assertEqual(tuple(batch.skills.shape), (1, 2))
        output = KTOutput(
            predictions=torch.tensor([[0.4]]),
            targets=torch.tensor([[1.0]]),
            prediction_mask=torch.tensor([[True]]),
            sequence_states=torch.zeros(1, 2, 4),
            state_mask=torch.tensor([[True, True]]),
        )
        output.validate()

    def test_output_rejects_misaligned_state_mask(self):
        with self.assertRaisesRegex(ValueError, "state_mask"):
            KTOutput(
                predictions=torch.tensor([[0.4]]),
                targets=torch.tensor([[1.0]]),
                prediction_mask=torch.tensor([[True]]),
                sequence_states=torch.zeros(1, 2, 4),
                state_mask=torch.tensor([[True]]),
            ).validate()
```

- [ ] **Step 2: Run tests and verify missing package failure**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_contracts tests.unit.test_registry -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'kt_state_pipeline'`.

- [ ] **Step 3: Implement immutable contracts and registry**

```python
@dataclass(frozen=True)
class KTBatch:
    questions: torch.Tensor
    skills: torch.Tensor
    responses: torch.Tensor
    attention_mask: torch.Tensor
    views: Mapping[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KTBatch":
        required = {"questions", "skills", "responses", "attention_mask"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"KTBatch is missing fields: {missing}")
        return cls(
            questions=value["questions"],
            skills=value["skills"],
            responses=value["responses"],
            attention_mask=value["attention_mask"].bool(),
            views=value.get("views", {}),
        )

    def to(self, device: torch.device) -> "KTBatch":
        return replace(
            self,
            questions=self.questions.to(device),
            skills=self.skills.to(device),
            responses=self.responses.to(device),
            attention_mask=self.attention_mask.to(device),
            views={name: tensor.to(device) for name, tensor in self.views.items()},
        )

@dataclass
class KTOutput:
    predictions: torch.Tensor
    targets: torch.Tensor
    prediction_mask: torch.Tensor
    sequence_states: Optional[torch.Tensor] = None
    state_mask: Optional[torch.Tensor] = None
    diagnostics: MutableMapping[str, Any] = field(default_factory=dict)
    def validate(self) -> None:
        if self.predictions.shape != self.targets.shape:
            raise ValueError("predictions and targets must have identical shapes")
        if self.prediction_mask.shape != self.targets.shape:
            raise ValueError("prediction_mask must align with targets")
        if (self.sequence_states is None) != (self.state_mask is None):
            raise ValueError("sequence_states and state_mask must be provided together")
        if self.sequence_states is not None:
            if self.sequence_states.shape[:2] != self.state_mask.shape:
                raise ValueError("state_mask must align with sequence_states")

@dataclass
class LossBundle:
    components: Mapping[str, torch.Tensor]
    def total(self) -> torch.Tensor:
        if not self.components:
            raise ValueError("LossBundle requires at least one component")
        return torch.stack([value.reshape(()) for value in self.components.values()]).sum()
```

The registry normalizes names to lowercase, rejects duplicate registrations,
and reports valid names in unknown-model errors.

- [ ] **Step 4: Run unit tests**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_contracts tests.unit.test_registry -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit the contract slice**

```bash
git add src/kt_state_pipeline tests/unit tests/__init__.py
git commit -m "refactor: add canonical KT adapter contracts"
```

### Task 2: Independent clustering objective and system composition

**Files:**
- Create: `src/kt_state_pipeline/objectives/__init__.py`
- Create: `src/kt_state_pipeline/objectives/cluster_friendly.py`
- Create: `src/kt_state_pipeline/engine/__init__.py`
- Create: `src/kt_state_pipeline/engine/system.py`
- Test: `tests/unit/test_cluster_objective.py`
- Test: `tests/unit/test_system.py`

**Interfaces:**
- Consumes: `KTBatch`, `KTOutput`, `LossBundle`, and `BackboneAdapter` from Task 1.
- Produces: `ClusterFriendlyObjective.forward(states, state_mask) -> LossBundle`, `KTSystem.forward(batch, return_states=None) -> KTOutput`, and `KTSystem.compute_loss(batch, output) -> LossBundle`.

- [ ] **Step 1: Write failing loss and gradient tests**

```python
def test_cluster_objective_ignores_padding_and_backpropagates(self):
    states = torch.randn(2, 4, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    objective = ClusterFriendlyObjective(input_dim=8, num_clusters=3,
                                         enabled_losses="soft,kmeans")
    losses = objective(states, mask)
    losses.total().backward()
    self.assertIsNotNone(states.grad)
    self.assertTrue(torch.isfinite(losses.total()))
```

Add a fake adapter whose `compute_base_loss` returns `{"kt": tensor(2.0)}` and
assert that system loss equals base loss plus weighted clustering loss.

- [ ] **Step 2: Verify tests fail before implementation**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_cluster_objective tests.unit.test_system -v`

Expected: FAIL because objective and system modules do not exist.

- [ ] **Step 3: Move and adapt the cluster-friendly implementation**

Port the mathematics from `cluster_friendly_module.py` without importing any
backbone. Return named weighted components (`cluster_soft`, `cluster_kmeans`,
`cluster_separation`, `cluster_temporal`) and preserve Student-t assignment,
nearest-center compactness, optional center separation, and temporal-neighbor
loss. Return a differentiable zero scalar when a batch has too few valid states.

Implement system composition as:

```python
def compute_loss(self, batch: KTBatch, output: KTOutput) -> LossBundle:
    components = dict(self.adapter.compute_base_loss(batch, output).components)
    if self.cluster_objective is not None:
        if output.sequence_states is None or output.state_mask is None:
            raise ValueError("Joint training requires timestep sequence states")
        components.update(
            self.cluster_objective(output.sequence_states, output.state_mask).components
        )
    return LossBundle(components)
```

- [ ] **Step 4: Run objective and system tests**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_cluster_objective tests.unit.test_system -v`

Expected: all tests PASS and gradients are finite.

- [ ] **Step 5: Commit the independent objective**

```bash
git add src/kt_state_pipeline/objectives src/kt_state_pipeline/engine tests/unit/test_cluster_objective.py tests/unit/test_system.py
git commit -m "refactor: separate clustering objective from backbones"
```

### Task 3: CL4KT runtime backbone and adapter regression

**Files:**
- Create: `src/kt_state_pipeline/backbones/__init__.py`
- Create: `src/kt_state_pipeline/backbones/cl4kt/__init__.py`
- Create: `src/kt_state_pipeline/backbones/cl4kt/model.py`
- Create: `src/kt_state_pipeline/backbones/cl4kt/modules.py`
- Create: `src/kt_state_pipeline/adapters/cl4kt.py`
- Modify: `src/kt_state_pipeline/adapters/__init__.py`
- Test: `tests/adapters/test_cl4kt_adapter.py`
- Test: `tests/regression/test_cl4kt_regression.py`

**Interfaces:**
- Consumes: Task 1 adapter contract and current `models/cl4kt.py` behavior.
- Produces: registered adapter name `cl4kt`; `CL4KTAdapter.forward()` returns shifted predictions/targets and causal knowledge-retriever states `x` before concatenation with question embeddings.

- [ ] **Step 1: Write failing CL4KT adapter contract tests**

Build a tiny `hidden_size=8`, `num_blocks=1`, `num_attn_heads=2` model and assert:

```python
output = adapter(batch, return_states=True)
self.assertEqual(output.predictions.shape, (2, 4))
self.assertEqual(output.targets.shape, (2, 4))
self.assertEqual(output.sequence_states.shape, (2, 5, 8))
self.assertEqual(output.state_mask.shape, (2, 5))
self.assertIn("kt", adapter.compute_base_loss(batch, output).components)
```

For causality, clone an evaluation batch, flip `responses[:, 2]`, and assert
`states_before[:, 2]` equals `states_after[:, 2]` within `atol=1e-6`.

- [ ] **Step 2: Run CL4KT tests and verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.adapters.test_cl4kt_adapter tests.regression.test_cl4kt_regression -v`

Expected: FAIL because the runtime backbone and adapter are absent.

- [ ] **Step 3: Port CL4KT without its clustering module**

Copy the current CL4KT architecture into the package, preserving parameter names
needed by legacy checkpoints. Refactor its native forward helper to return the
knowledge-retriever `x` once so prediction and state loss share one computation.
Do not instantiate or call `ClusterFriendlyModule` inside the backbone.

The adapter translates native output to `KTOutput`, computes BCE plus configured
contrastive loss, and declares `requires_augmented_views=True`.

- [ ] **Step 4: Add seeded regression assertions**

Load one state dict into the current native CL4KT and packaged CL4KT, switch both
to evaluation mode, and compare predictions and `x` tensors with `rtol=1e-5,
atol=1e-6`. Compare exported state-dict keys after excluding the legacy
`cluster_friendly_module.*` prefix.

- [ ] **Step 5: Run CL4KT and prior unit tests**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.adapters.test_cl4kt_adapter tests.regression.test_cl4kt_regression tests.unit.test_system -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit the CL4KT adapter slice**

```bash
git add src/kt_state_pipeline/backbones/cl4kt src/kt_state_pipeline/adapters tests/adapters/test_cl4kt_adapter.py tests/regression/test_cl4kt_regression.py
git commit -m "refactor: wrap CL4KT with causal-state adapter"
```

### Task 4: AKT runtime backbone and adapter

**Files:**
- Create: `src/kt_state_pipeline/backbones/akt/__init__.py`
- Create: `src/kt_state_pipeline/backbones/akt/model.py`
- Create: `src/kt_state_pipeline/backbones/akt/modules.py`
- Create: `src/kt_state_pipeline/adapters/akt.py`
- Modify: `src/kt_state_pipeline/adapters/__init__.py`
- Test: `tests/adapters/test_akt_adapter.py`
- Test: `tests/regression/test_reference_state_dicts.py`

**Interfaces:**
- Consumes: canonical questions, skills, responses, mask, and optional item IDs.
- Produces: registered adapter name `akt`; causal `d_output` states; BCE and named `rasch_regularization` when `num_questions > 0`.

- [ ] **Step 1: Write failing AKT shape, loss, and causality tests**

Instantiate both `n_pid=0` and `n_pid>0` variants. Assert finite predictions,
`[B,T,H]` states, correct masks, finite loss, and successful backward. Flip the
current response and assert the same-timestep state remains unchanged while a
later state is allowed to differ.

- [ ] **Step 2: Run AKT tests and verify missing implementation**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.adapters.test_akt_adapter -v`

Expected: FAIL on missing AKT modules.

- [ ] **Step 3: Port and minimally modernize AKT**

Use `KT_backbone_ref/pykt-toolkit-main/pykt/models/akt.py` and the original
`KT_backbone_ref/AKT-master/akt.py` as comparison sources. Preserve architecture
and initialization, but construct causal masks with `torch` on `query.device`
instead of a module-level device and NumPy conversion. Expose `d_output` from
the same native forward pass used for prediction.

Map padding responses to a safe embedding ID before embedding and exclude them
through masks. Keep item difficulty optional and report its regularizer as a
named loss component.

- [ ] **Step 4: Verify reference-compatible parameter shapes**

In `test_reference_state_dicts.py`, instantiate matching reference/runtime
configurations and assert corresponding embeddings, attention projections,
decoder blocks, and prediction head parameters have identical shapes. Load the
matching subset with `strict=False` and assert no unexpected runtime keys outside
documented wrapper fields.

- [ ] **Step 5: Run AKT contract and regression tests**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.adapters.test_akt_adapter tests.regression.test_reference_state_dicts -v`

Expected: all AKT tests PASS.

- [ ] **Step 6: Commit AKT support**

```bash
git add src/kt_state_pipeline/backbones/akt src/kt_state_pipeline/adapters/akt.py src/kt_state_pipeline/adapters/__init__.py tests/adapters/test_akt_adapter.py tests/regression/test_reference_state_dicts.py
git commit -m "feat: add AKT backbone adapter"
```

### Task 5: simpleKT runtime backbone and adapter

**Files:**
- Create: `src/kt_state_pipeline/backbones/simplekt/__init__.py`
- Create: `src/kt_state_pipeline/backbones/simplekt/model.py`
- Create: `src/kt_state_pipeline/backbones/simplekt/modules.py`
- Create: `src/kt_state_pipeline/adapters/simplekt.py`
- Modify: `src/kt_state_pipeline/adapters/__init__.py`
- Modify: `tests/regression/test_reference_state_dicts.py`
- Test: `tests/adapters/test_simplekt_adapter.py`

**Interfaces:**
- Consumes: canonical questions, skills, responses, mask, and optional item IDs.
- Produces: registered adapter name `simplekt`; causal knowledge-retriever `d_output`; BCE and optional named difficulty regularization.

- [ ] **Step 1: Write failing simpleKT adapter contract tests**

Repeat the common assertions from AKT using simpleKT's single knowledge-retriever
stack. Verify sinusoidal positional embeddings move with the input device and
that current-response perturbation does not change the same-timestep state.

- [ ] **Step 2: Run the tests and observe missing implementation**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.adapters.test_simplekt_adapter -v`

Expected: FAIL on missing simpleKT modules.

- [ ] **Step 3: Port simpleKT and expose its causal state**

Port the `qid` family needed by the supplied datasets from
`KT_backbone_ref/pykt-toolkit-main/pykt/models/simplekt.py`. Retain the reference
embedding, position encoding, attention, and prediction head behavior. Replace
module-level device assumptions with tensor-local devices and return
`d_output` before concatenation with the question embedding.

- [ ] **Step 4: Extend reference shape and state-load checks**

Assert reference/runtime parameter shapes for embeddings, positional weights,
retriever blocks, and prediction head. Load a compatible subset and verify the
runtime model produces finite outputs.

- [ ] **Step 5: Run all three adapter test modules**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.adapters.test_cl4kt_adapter tests.adapters.test_akt_adapter tests.adapters.test_simplekt_adapter -v`

Expected: all adapter contracts PASS.

- [ ] **Step 6: Commit simpleKT support**

```bash
git add src/kt_state_pipeline/backbones/simplekt src/kt_state_pipeline/adapters/simplekt.py src/kt_state_pipeline/adapters/__init__.py tests/adapters/test_simplekt_adapter.py tests/regression/test_reference_state_dicts.py
git commit -m "feat: add simpleKT backbone adapter"
```

### Task 6: Canonical data, model configuration, and loaders

**Files:**
- Create: `src/kt_state_pipeline/data/__init__.py`
- Create: `src/kt_state_pipeline/data/datasets.py`
- Create: `src/kt_state_pipeline/data/augmentations.py`
- Create: `src/kt_state_pipeline/data/splits.py`
- Create: `src/kt_state_pipeline/data/loaders.py`
- Create: `src/kt_state_pipeline/core/config.py`
- Create: `configs/models/cl4kt.yaml`
- Create: `configs/models/akt.yaml`
- Create: `configs/models/simplekt.yaml`
- Create: `configs/objectives/cluster_friendly.yaml`
- Modify: `configs/paper.yaml`
- Test: `tests/unit/test_data_pipeline.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: adapter `AdapterCapabilities` and existing preprocessed dataframe schema.
- Produces: `DatasetMetadata`, `load_preprocessed_dataset()`, `make_splits()`, `make_dataloaders()`, `load_experiment_config()`, and `model_config(config, model_name)`.

- [ ] **Step 1: Write failing split and loader tests**

Create an in-memory dataframe with six users. Assert official splits remain
unchanged, seeded five-fold splits repeat exactly, base batches expose canonical
fields, CL4KT train batches contain named augmented views, and AKT/simpleKT train
batches do not.

- [ ] **Step 2: Write failing configuration tests**

Assert that all three model files load, unknown model names list `akt`, `cl4kt`,
and `simplekt`, `hidden_dim` is positive, attention heads divide hidden size, and
invalid clustering loss names fail before training starts.

- [ ] **Step 3: Run data/config tests and verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_data_pipeline tests.unit.test_config -v`

Expected: FAIL because package data/config modules do not exist.

- [ ] **Step 4: Move canonical dataset and augmentation behavior**

Preserve recent/early window selection and padding conventions from
`data_loaders.py`. Implement a generic transformed dataset whose transform is
selected by adapter capabilities. Move CL4KT augmentation from
`SimCLRDatasetWrapper` without changing its seeded behavior.

- [ ] **Step 5: Add validated model/objective configuration**

Use existing PyYAML and `ConfigNode`; do not introduce Hydra. Support legacy
`cl4kt_config` while adding `models.cl4kt`, `models.akt`, and `models.simplekt`.
Centralize CLI overrides so model-specific flags are applied to the selected
model mapping rather than through `if model_name == "cl4kt"`.

- [ ] **Step 6: Run data/config tests**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_data_pipeline tests.unit.test_config -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit data and configurations**

```bash
git add src/kt_state_pipeline/data src/kt_state_pipeline/core/config.py configs tests/unit/test_data_pipeline.py tests/unit/test_config.py
git commit -m "refactor: add model-aware data and configuration layer"
```

### Task 7: Generic trainer, evaluator, and checkpoint schema

**Files:**
- Create: `src/kt_state_pipeline/engine/trainer.py`
- Create: `src/kt_state_pipeline/engine/evaluator.py`
- Create: `src/kt_state_pipeline/engine/checkpoint.py`
- Create: `src/kt_state_pipeline/cli/train.py`
- Modify: `main.py`
- Modify: `train.py`
- Test: `tests/unit/test_checkpoint.py`
- Test: `tests/integration/test_train_step.py`

**Interfaces:**
- Consumes: registry, `KTSystem`, model-aware dataloaders, Accelerate, and validated config.
- Produces: `train_epoch()`, `evaluate()`, `save_checkpoint()`, `load_checkpoint()`, `export_analysis_backbone()`, and CLI `train_main(argv=None)`.

- [ ] **Step 1: Write failing checkpoint schema tests**

Save a tiny adapter/system checkpoint and assert it records schema version,
backbone, dataset metadata, config, seed, epoch, metrics, model state, optimizer
state, and optional objective state. Assert `export_analysis_backbone()` omits
all `cluster_objective.*` keys and a backbone mismatch raises `ValueError`.

- [ ] **Step 2: Write a failing one-step integration test**

For each registered adapter, create a two-batch synthetic loader, run one
optimization step through `KTSystem`, and assert at least one backbone parameter
changes and evaluation returns finite AUC-compatible predictions and targets.

- [ ] **Step 3: Run engine tests and verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_checkpoint tests.integration.test_train_step -v`

Expected: FAIL because generic engine functions do not exist.

- [ ] **Step 4: Implement generic training and evaluation**

Use `Accelerator.backward`, configured gradient clipping, validation AUC early
stopping, and one consistent unwrapping helper for Accelerate/DataParallel.
Initialize `best_epoch=0` and `best_valid_auc=-inf` so the first finite epoch is
always checkpointed. Aggregate prediction tensors using `prediction_mask` only.

- [ ] **Step 5: Implement checkpoint compatibility**

Use schema version `1`. Accept old CL4KT dictionaries containing
`model_state_dict`; strip `module.` prefixes and ignore only the documented
`cluster_friendly_module.*` keys. Reject ambiguous legacy layouts for AKT and
simpleKT with an actionable error.

- [ ] **Step 6: Convert top-level training files to wrappers**

Keep `python main.py` arguments working and add `akt` and `simplekt` model choices.
The wrapper adds `src` to its import path and delegates to
`kt_state_pipeline.cli.train.train_main`. Leave import aliases in `train.py` for
callers of `model_train`, marking them as compatibility functions.

- [ ] **Step 7: Run engine and CL4KT regression tests**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_checkpoint tests.integration.test_train_step tests.regression.test_cl4kt_regression -v`

Expected: all tests PASS.

- [ ] **Step 8: Commit the generic training engine**

```bash
git add src/kt_state_pipeline/engine src/kt_state_pipeline/cli main.py train.py tests/unit/test_checkpoint.py tests/integration/test_train_step.py
git commit -m "refactor: make training engine backbone independent"
```

### Task 8: Generic state extraction and post-hoc clustering

**Files:**
- Create: `src/kt_state_pipeline/discovery/__init__.py`
- Create: `src/kt_state_pipeline/discovery/extractor.py`
- Create: `src/kt_state_pipeline/discovery/clustering.py`
- Create: `src/kt_state_pipeline/discovery/artifacts.py`
- Create: `src/kt_state_pipeline/cli/extract.py`
- Modify: `extract_and_cluster.py`
- Test: `tests/unit/test_extractor.py`
- Test: `tests/integration/test_discovery_pipeline.py`

**Interfaces:**
- Consumes: any loaded `BackboneAdapter`, canonical evaluation loader, dataset sequence metadata, and checkpoint metadata.
- Produces: `extract_timestep_states()`, `fit_state_clusters()`, `load_cluster_bundle()`, and CLI `extract_main(argv=None)`.

- [ ] **Step 1: Write failing alignment and clustering tests**

Use a fake adapter returning known `[B,T,H]` states. Assert padding is removed,
metadata rows retain user, sequence, token, item, skill, response, and feature
indices, and feature order matches chronological order within each selected
sequence. Fit k-means and assert the saved bundle records backbone, hidden
dimension, scaler, best `k`, and cluster unit `timestep`.

- [ ] **Step 2: Run discovery tests and verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_extractor tests.integration.test_discovery_pipeline -v`

Expected: FAIL because discovery modules do not exist.

- [ ] **Step 3: Move extraction and clustering logic**

Move generic functions from `extract_and_cluster.py`. Call the adapter contract
instead of `model.extract_features()`. Validate `state_mask` against dataset
padding and verify that extracted row count equals the mask sum. Retain standard
scaling, k sensitivity, silhouette, Davies-Bouldin, Calinski-Harabasz,
representatives, compressed latent arrays, and optional t-SNE.

- [ ] **Step 4: Make titles and paths model aware**

Replace CL4KT literals with checkpoint backbone metadata. Write outputs below
`<output>/<dataset>/<backbone>/<split>` while allowing the wrapper's legacy path
mode for existing experiment directories.

- [ ] **Step 5: Convert top-level extraction to a wrapper and test**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_extractor tests.integration.test_discovery_pipeline -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit discovery pipeline**

```bash
git add src/kt_state_pipeline/discovery src/kt_state_pipeline/cli/extract.py extract_and_cluster.py tests/unit/test_extractor.py tests/integration/test_discovery_pipeline.py
git commit -m "refactor: generalize state extraction and clustering"
```

### Task 9: Backbone-independent trajectory and order-shuffle inference

**Files:**
- Create: `src/kt_state_pipeline/analysis/__init__.py`
- Create: `src/kt_state_pipeline/analysis/inference.py`
- Create: `src/kt_state_pipeline/analysis/transitions.py`
- Modify: `trajectory_analysis.py`
- Modify: `order_shuffle_control.py`
- Test: `tests/unit/test_transitions.py`
- Test: `tests/integration/test_order_shuffle.py`

**Interfaces:**
- Consumes: adapter state output, saved cluster bundle, canonical dataset, and state metadata.
- Produces: `infer_states()`, `assign_saved_clusters()`, `labels_by_user()`, `transition_counts()`, and `transition_probabilities()`.

- [ ] **Step 1: Write failing transition and saved-cluster tests**

Assert transition counting for known user label sequences, zero-row handling,
and chronological sorting. Assert a saved scaler and k-means model are reused
without refitting for trajectory and shuffle inference.

- [ ] **Step 2: Write failing identity/order-shuffle integration test**

For a tiny causal adapter, verify identity inference reproduces saved features
within tolerance. Shuffle KC-response pairs within each student's valid window,
preserve per-student correctness totals, and assert output assignments have the
same number of valid timesteps.

- [ ] **Step 3: Run analysis tests and verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_transitions tests.integration.test_order_shuffle -v`

Expected: FAIL because shared analysis inference is absent.

- [ ] **Step 4: Extract shared inference and transition functions**

Replace both scripts' calls to `model.extract_features()` with
`infer_states(adapter, batch)`. Validate cluster-bundle backbone and hidden
dimension before prediction. Keep order-shuffle's no-retraining/no-refitting
control invariant.

- [ ] **Step 5: Remove model-name assignments from analysis CLIs**

Add `--model-name` with checkpoint metadata as the default authority. Update
descriptions, plot titles, and explanatory text so only paper-specific claims
that truly concern CL4KT retain its name.

- [ ] **Step 6: Run analysis tests**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_transitions tests.integration.test_order_shuffle -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit shared analysis inference**

```bash
git add src/kt_state_pipeline/analysis trajectory_analysis.py order_shuffle_control.py tests/unit/test_transitions.py tests/integration/test_order_shuffle.py
git commit -m "refactor: make trajectory inference backbone independent"
```

### Task 10: Multi-model experiment orchestration and remaining analyses

**Files:**
- Create: `src/kt_state_pipeline/experiments/__init__.py`
- Create: `src/kt_state_pipeline/experiments/artifact_paths.py`
- Create: `src/kt_state_pipeline/experiments/manifest.py`
- Create: `src/kt_state_pipeline/experiments/commands.py`
- Create: `configs/experiments/paper.yaml`
- Modify: `run_experiments.py`
- Modify: `run_all_datasets.py`
- Modify: `cluster_pedagogy_analysis.py`
- Modify: `generate_experiment3_visuals.py`
- Test: `tests/unit/test_experiment_commands.py`
- Test: `tests/integration/test_resume_manifest.py`

**Interfaces:**
- Consumes: generic train/extract/analyze CLIs and model-aware artifact paths.
- Produces: deterministic commands for every dataset/backbone/seed/configuration, resumable stage manifests, and collision-free output roots.

- [ ] **Step 1: Write failing command-matrix tests**

Load `configs/experiments/paper.yaml` and assert it expands exactly the requested
datasets, `cl4kt`, `akt`, `simplekt`, model seeds, and ablation configurations.
Assert every train/extract command carries the same backbone and every artifact
path contains experiment, dataset, backbone, seed, and stage.

- [ ] **Step 2: Write failing resume-manifest tests**

Create a temporary manifest, mark a stage completed with its required outputs,
and assert it is reused only when all outputs exist and its backbone/config hash
matches. A changed backbone or missing output must schedule the stage again.

- [ ] **Step 3: Run orchestration tests and verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_experiment_commands tests.integration.test_resume_manifest -v`

Expected: FAIL because orchestration modules do not exist.

- [ ] **Step 4: Move command and manifest responsibilities**

Replace hard-coded CL4KT checkpoint paths and commands in `run_experiments.py`.
Keep representation-quality configurations, k sensitivity, seed stability,
bootstrap stability, pedagogy, order-shuffle, and trajectory stages unchanged in
meaning. Record backbone and resolved config snapshot in every stage manifest.

- [ ] **Step 5: Generalize remaining analysis labels**

Pass backbone metadata into pedagogy reports and visual captions. Preserve text
that explains variables are post-hoc rather than explicit model inputs, but use
the selected backbone name instead of a CL4KT literal.

- [ ] **Step 6: Run orchestration tests and dry-run matrix**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_experiment_commands tests.integration.test_resume_manifest -v`

Run: `.venv/bin/python run_all_datasets.py --models cl4kt,akt,simplekt --datasets ASSISTMENT2009 --dry-run`

Expected: tests PASS and dry-run prints distinct commands/paths for all three
models without launching training.

- [ ] **Step 7: Commit orchestration changes**

```bash
git add src/kt_state_pipeline/experiments configs/experiments run_experiments.py run_all_datasets.py cluster_pedagogy_analysis.py generate_experiment3_visuals.py tests/unit/test_experiment_commands.py tests/integration/test_resume_manifest.py
git commit -m "refactor: orchestrate paper experiments across backbones"
```

### Task 11: Compatibility cleanup, documentation, and full verification

**Files:**
- Modify: `models/__init__.py`
- Modify: `models/cl4kt.py`
- Modify: `models/modules.py`
- Modify: `data_loaders.py`
- Modify: `cluster_friendly_module.py`
- Modify: `README.md`
- Modify: `THIRD_PARTY.txt`
- Modify: `.gitignore`
- Modify: `requirements.txt` only if verified runtime imports require a missing package
- Create: `docs/architecture/adding-a-backbone.md`
- Create: `tests/smoke/test_three_backbones.py`

**Interfaces:**
- Consumes: all prior package APIs.
- Produces: compatibility imports, contributor documentation, and a verified three-backbone pipeline.

- [ ] **Step 1: Write the three-backbone smoke test**

For each registered backbone, create a temporary tiny dataframe, train one
epoch, reload the exported analysis backbone, extract timestep states, standardize
them, fit `k=2`, and compute a transition matrix. Assert finite loss, matching
state/metadata counts, two cluster centers, and a square transition matrix.

- [ ] **Step 2: Run smoke test before cleanup**

Run: `PYTHONPATH=src .venv/bin/python -m unittest tests.smoke.test_three_backbones -v`

Expected: PASS only when all cross-component APIs are connected; otherwise fix
the first concrete integration failure before continuing.

- [ ] **Step 3: Replace legacy modules with compatibility re-exports**

Make old import paths re-export packaged CL4KT, data classes, and cluster
objective with deprecation docstrings. Do not maintain duplicate runtime
implementations. Search for and remove runtime branches or imports tied to a
specific model outside adapters, registration, configs, compatibility aliases,
and scientifically model-specific report text.

Run: `rg -n 'if .*model.*cl4kt|model_name = "cl4kt"|from models.cl4kt|extract_features' --glob '*.py' --glob '!KT_backbone_ref/**'`

Expected: no pipeline-control match; remaining matches are compatibility or
model-specific implementation references with explanatory comments.

- [ ] **Step 4: Document operation and extension**

Update README commands for all three backbones, configuration layering, output
paths, legacy checkpoint behavior, and environment installation. In
`adding-a-backbone.md`, document the canonical contract, causal-state boundary,
registration decorator, required config, adapter tests, and smoke command.
Update `THIRD_PARTY.txt` with the exact AKT/simpleKT source paths and retained
licenses.

- [ ] **Step 5: Audit dependencies**

Run: `PYTHONPATH=src .venv/bin/python -c "import kt_state_pipeline; import torch, numpy, pandas, sklearn, yaml, joblib"`

Expected: imports succeed using current requirements. Modify `requirements.txt`
only if a runtime package imported by committed code is absent; include its
minimum compatible version and repeat the import audit.

- [ ] **Step 6: Run the complete automated suite**

Run: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`

Expected: all unit, adapter, regression, integration, and smoke tests PASS.

- [ ] **Step 7: Run syntax and CLI verification**

Run: `.venv/bin/python -m compileall -q src main.py train.py extract_and_cluster.py trajectory_analysis.py order_shuffle_control.py run_experiments.py`

Run: `.venv/bin/python main.py --help`

Run: `.venv/bin/python extract_and_cluster.py --help`

Run: `.venv/bin/python run_experiments.py --help`

Expected: compilation succeeds and every help command exits zero with all three
backbone choices available where applicable.

- [ ] **Step 8: Run a real minimal dataset smoke command**

Run one epoch on the smallest available configured dataset for each backbone,
using separate temporary output roots under `/tmp/kt-state-pipeline-smoke`, then
run extraction with `k=3`. Confirm each checkpoint metadata file names the correct
backbone and each extraction produces `latent_features.npz`,
`latent_metadata.csv`, `cluster_metrics.csv`, and `kmeans_model.pkl`.

- [ ] **Step 9: Commit cleanup and documentation**

```bash
git add models data_loaders.py cluster_friendly_module.py README.md THIRD_PARTY.txt .gitignore requirements.txt docs/architecture tests/smoke
git commit -m "docs: complete multi-backbone pipeline migration"
```

## Plan Self-Review

- Spec coverage: contracts, three causal representation boundaries, independent
  joint objective, data capabilities, checkpoints, artifact layout, all analyses,
  experiment orchestration, compatibility, dependencies, and contributor docs
  each map to an explicit task.
- Scope sequencing: every task consumes interfaces defined by an earlier task;
  no task requires AKT/simpleKT behavior before their adapters exist.
- Type consistency: all pipeline stages use `KTBatch`, `KTOutput`, `LossBundle`,
  `BackboneAdapter`, and `AdapterCapabilities` with the signatures established in
  Task 1.
- Dependency consistency: all commands use the existing `.venv`; tests use
  `unittest`, and dependency changes are conditional on a verified missing
  runtime import.
