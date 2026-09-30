# Changelog

## 0.1.0 (unreleased)

- Analyses of code that uses Nx 1.0, each a category of finding reported
  under its name (`error[argus.nx_shapes]`):
  - `nx_shapes`: shapes Nx rejects where tensors meet, tensor access and
    tuples;
  - `nx_names`: axes Nx accepts where the code does not line them up;
  - `nx_math`: results the code's own math can make infinite or NaN, and
    operands nothing checks;
  - `nx_types`: operands, literals and types Nx rejects or reads
    differently than written;
  - `nx_options`: options Nx rejects;
  - `nx_indices`: indices, slices and ranges;
  - `nx_traced`: traced code and `defn` control flow;
  - `nx_gradients`: gradients;
  - `nx_containers`: containers;
  - `nx_random`: random keys and seeds;
  - `nx_freed`: tensors used after they are freed;
  - `nx_serving`: servings and batches;
  - `emlx`: calls EMLX computes differently from BinaryBackend and EXLA,
    and tensors of two backends meeting.
- The engines that find them: `ArgusNxTensorAnalyses.TensorShapes` for
  the `nx_` analyses, and `ArgusNxTensorAnalyses.EMLX` for `emlx`, whose
  program includes the other's; `ArgusNxTensorAnalyses.run/3` solves one
  program for any set of analyses.
- `mix argus_nx_tensor_analyses`: `mix argus` with these analyses beside
  Argus's own. A run that names none runs the default analyses (every one
  but `emlx`); `analyses:`, `unsupported_types:` and `float_types:` under
  `argus_nx_tensor_analyses:` in `mix.exs` configure them.
