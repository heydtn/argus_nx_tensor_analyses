# Changelog

## 0.1.0 (unreleased)

- `ArgusNxTensorAnalyses.TensorShapes`, modeling Nx 1.0:
  - calls whose operand shapes Nx rejects (`tensor_shape_mismatch`), and
    calls Nx accepts where the code does not line its axes up
    (`tensor_axis_misalignment`);
  - results the code's own math can make infinite or NaN, and operands
    nothing checks (`tensor_nonfinite_result`);
  - operands, literals and types Nx rejects or reads differently than
    written (`tensor_type_error`);
  - calls Nx rejects or computes wrongly for other reasons
    (`tensor_call_error`): options, indices and slices, tensor access and
    tuples, traced code and `defn` control flow, gradients, containers,
    random keys and spent tensors, servings and batches.
- `ArgusNxTensorAnalyses.EMLX` (`tensor_emlx`): calls EMLX computes
  differently from BinaryBackend and EXLA, and tensors of two backends
  meeting.
- `mix argus_nx_tensor_analyses`: `mix argus` with these analyses beside
  Argus's own. A run that names none runs the default analyses
  (`tensor_shapes`); `analyses:`, `unsupported_types:` and `float_types:`
  under `argus_nx_tensor_analyses:` in `mix.exs` configure them.
