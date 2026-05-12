# RNA GUI — quick-start

## Run

```
cargo run --bin gui
```

---

## Directory layout

```
src/
  history/
    mod.rs              — module declarations + re-exports
    types.rs            — all snapshot / history structs + ActivationName + forward_snapshot()
    linear.rs           — LinearHistory  (re-impl of Linear::train with history capture)
    gradient.rs         — GradientHistory (re-impl of Gradient::train)
    adeline.rs          — AdelineHistory  (re-impl of Adeline::train)
    mlp.rs              — train_mlp_with_history()  (mirrors MLP::train backprop loop)
    single_layer.rs     — train_single_layer_with_history()  (mirrors bin-local SingleLayer)

  gui/
    mod.rs              — module declarations
    params.rs           — HyperParams, MlpActivation, defaults_for(name)
    app.rs              — GuiApp + eframe::App  (sidebar / central / snapshot panel)
    experiments.rs      — hardcoded experiment registry + run_experiment(name, verbose, params)
    plots.rs            — egui_plot helpers (loss, scatter, boundary, regression, sparklines)

  bin/
    gui.rs              — entry point; pulls in history/gui_module via #[path = "..."]
                          so src/lib.rs is never touched

GUI_README.md           — this file
```

---

## Using the Hyperparameters panel

A collapsible **Hyperparameters** section appears in the left sidebar before the
Run button.  Controls reset to the experiment's original defaults whenever you
switch experiments, and a **Reset to defaults** button is always available.

| Control | Type | Notes |
|---|---|---|
| Learning rate | `DragValue` (drag or type) | Range 1e-6 … 10, 6 decimal places |
| Tolerance | `DragValue` | Range 0 … 100 |
| Max epochs | `DragValue` | Range 1 … 100 000 |
| Class A / B labels | `DragValue` | Visible only for binary classification experiments |
| Threshold | `DragValue` | Decision threshold for classification |
| Error limit | checkbox + `DragValue` | Tick to enable early stopping on misclassification count |
| Activation | dropdown | **MLP experiments only** — Sigmoid / Tanh / Identity |
| Hidden layers | `DragValue` (1 … 5) | **MLP experiments only** |
| Layer N neurons | `DragValue` per layer (1 … 64) | Appears / disappears as layer count changes |

**Regression experiments** (`gradient_2_11`): class controls are hidden because
the experiment has no classification threshold.

**Multi-output SingleLayer experiments** (`*_singlelayer_3_1`): class controls
are hidden because the CSV dataset has 3 output columns and each neuron is
trained independently without a `ClassificationStop`.

The values actually used are stored in the exported JSON via the `max_epochs`,
`learning_rate`, `tolerance` fields of `History` / `MlpHistory` /
`SingleLayerHistory`, and the new `activation` field in `MlpHistory`.

---

## MLP decision boundary rendering

For `rna_xor_4_3` (and any future MLP with `input_dim == 2`):

- A **64 × 64 grid** is sampled over the dataset bounding box with a 0.5-unit
  margin on each side.
- For every grid point `(x, y)` a **standalone forward pass** is run through
  the snapshot's layer weights using `forward_snapshot()` in `history/types.rs`.
  This does not require the live model.
- Each cell is colored:
  - **Blue-tint** (rgba 80, 120, 220, 90) — output ≥ threshold (positive class)
  - **Red-tint**  (rgba 220, 80, 80, 90) — output < threshold (negative class)
- The colored grid is uploaded as an `egui::ColorImage` and rendered as a
  `PlotImage` behind the scatter.  Texture names include the epoch index so
  egui caches one texture per epoch — re-uploading on every slider move is
  avoided.
- Scatter points are overlaid with two visual channels:
  - **Color** — true label (green = positive, red = negative)
  - **Shape** — circle = correctly classified at this epoch, × = misclassified

The boundary updates live as the epoch slider is dragged.

---

## New fields added to history structs

| Struct | New field | Why |
|---|---|---|
| `History` | `max_epochs: usize` | Records the epoch cap actually passed, so JSON export is self-contained |
| `SingleLayerHistory` | `max_epochs: usize` | Same |
| `MlpHistory` | `max_epochs: usize` | Same |
| `MlpHistory` | `activation: ActivationName` | Required to replay standalone forward passes from snapshots for boundary rendering |
| `MlpHistory` | `input_dim: usize` | GUI needs to know input dimensionality to decide whether to show the 2D boundary view |

A new free function `forward_snapshot(layers, input, activation)` in
`history/types.rs` implements the forward pass over `&[LayerSnapshot]` without
the live model.

---

## Ambiguous-option choices documented

| Question | Choice |
|---|---|
| Neurons per hidden layer | **Per-layer individual DragValues** that grow/shrink as hidden-layer count changes |
| Activation | **Uniform across all layers** — one dropdown for the whole MLP |
| Activation choices for MLP | Identity, Sigmoid, Tanh (**Step excluded** — it doesn't implement `Derivative`) |
| Multi-class SingleLayer class controls | **Hidden** — those experiments use no `ClassificationStop` |

---

## Re-implemented types — what was blocked and why

| New type | Original blocked by | Specific reason |
|---|---|---|
| `LinearHistory` | `Linear::train` | Loop runs to completion, returns `()`. No callback or observer hook. |
| `GradientHistory` | `Gradient::train` | Same; `squarred_error_sum` and `delta_weights` are local loop variables. |
| `AdelineHistory` | `Adeline::train` | Same; online-update + loss-evaluation passes are fully internal. |
| `train_mlp_with_history` | `MLP::train` | `train` is a method on `MLP` with no per-epoch yield or return. Mirror uses public fields of `MLP` / `Layer` / `Perceptron`. |
| `train_single_layer_with_history` | bin-local `SingleLayer` | Defined as a private struct inside each binary; not importable as a library type. |

### What to merge back later

1. **`Train::train`** — add an optional `FnMut(usize, f64)` callback (epoch, loss).
2. **`MLP::train`** — same, or split into `train_one_epoch` + `train`.
3. **`SingleLayer`** — move to `src/layer.rs` or a new `src/multi_layer.rs`.

---

## Per-epoch verbose output

Every history trainer has a `verbose: bool` flag.  Enable via the
**Verbose (terminal)** checkbox in the sidebar before pressing Run.

```
[gradient] epoch     7  MSE = 0.124800  w = [0.34, 0.21]  b = -0.05  misclassified = 1
[mlp]      epoch   150  MSE = 0.003200
[adeline]  epoch    42  MSE = 0.098700  w = [0.11, 0.89]  b = 0.03
```

---

## Export

After training, **💾 Export JSON** saves `<dataset_name>.json` next to the
executable.  The file contains the full serialised history including all new
metadata fields.
