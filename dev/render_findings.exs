# mix run dev/render_findings.exs [OUTPUT_DIRECTORY]
#
# Renders every finding both analyses place over the test suite's
# fixtures, as `mix argus` shows them (pentiment frames, without color),
# one file per fixture set and analysis under OUTPUT_DIRECTORY (default
# `_build/render`). Each finding is headed by the function it is in, so a
# lint case's findings are found by its function's name (`lint_N`).
#
# The fixtures are the ones the suite last compiled, beams and sources,
# under `_build/test/suite/fixtures`: run `mix test` first.
output = List.first(System.argv(), "_build/render") |> Path.expand()
fixtures = Path.expand("_build/test/suite/fixtures")
cwd = File.cwd!()

if not File.dir?(fixtures), do: Mix.raise("no fixtures under #{fixtures}: run mix test first")

# Each set with the options its tests solve it with.
runs = [
  {"tensor_shapes", ArgusNxTensorAnalyses.TensorShapes, [unsupported_types: [:f64]]},
  {"tensor_shapes_configured", ArgusNxTensorAnalyses.TensorShapes,
   [unsupported_types: [:f64], float_types: [:f16, :bf16, :f32]]},
  {"tensor_shapes", ArgusNxTensorAnalyses.EMLX, [unsupported_types: [:f64]]},
  {"emlx", ArgusNxTensorAnalyses.TensorShapes, [unsupported_types: [:c128]]},
  {"emlx", ArgusNxTensorAnalyses.EMLX, [unsupported_types: [:c128]]},
  {"emlx_on_exla", ArgusNxTensorAnalyses.EMLX, []}
]

File.rm_rf!(output)
File.mkdir_p!(output)
config = Argus.Config.load()

for {label, analysis, options} <- runs do
  set = String.replace_suffix(label, "_configured", "")
  beams = fixtures |> Path.join("#{set}/*.beam") |> Path.wildcard()
  {:ok, placed} = analysis.run(beams, options)

  # One finding at a time, to head each with the function it is in.
  entries =
    for %{finding: %{mfa: {module, name, arity}}} = one <- placed,
        [entry] <- [Argus.Report.build(%{analysis.name() => {:ok, [one]}}, config, cwd)],
        do: {Exception.format_mfa(module, name, arity), entry}

  rendered =
    entries
    |> Enum.sort_by(fn {function, entry} -> {function, entry.line, entry.title} end)
    |> Enum.map(fn {function, entry} ->
      ["## ", function, "\n", Argus.Report.Pentiment.format(entry, cwd), "\n\n"]
    end)

  file = Path.join(output, "#{label}_#{analysis.name()}.txt")
  File.write!(file, rendered)
  IO.puts("#{Path.relative_to(file, cwd)}: #{length(entries)} findings")
end
