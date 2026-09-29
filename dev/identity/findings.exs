# mix run dev/identity/findings.exs NAME
#
# Runs both analyses of the checkout it runs in (`TensorShapes.run/2` and
# `EMLX.run/2`) over every set of beams under the data directory's
# `beams/`, and keeps every finding they place, one per line with every
# field of the placed finding and of the finding itself (severity, title,
# detail, help, label, frames, file and line), sorted, under
# `findings/NAME`. Compare two runs with compare_findings.sh: a change to
# how findings are built or worded that should change none must change no
# line.
#
# The data directory is the one identity.exs uses; the README next to this
# script says how to fill it.
[name] = System.argv()

{common, 0} =
  System.cmd("git", [
    "-C",
    Path.dirname(__ENV__.file),
    "rev-parse",
    "--path-format=absolute",
    "--git-common-dir"
  ])

root =
  System.get_env("ARGUS_NX_IDENTITY_DIR") ||
    common |> String.trim() |> Path.dirname() |> Path.join("_build/identity")

analyses = [
  {"shapes", ArgusNxTensorAnalyses.TensorShapes},
  {"emlx", ArgusNxTensorAnalyses.EMLX}
]

# The test suite's fixtures, run with the options their tests use; any
# other directory of beams is a set of its own, run with the defaults.
fixture_sets = %{
  "tensor_shapes" => {"fixtures", [unsupported_types: [:f64]]},
  "emlx" => {"emlx_fixtures", [unsupported_types: [:c128]]},
  "emlx_on_exla" => {"emlx_on_exla", []}
}

sets =
  for directory <- root |> Path.join("beams/*") |> Path.wildcard() |> Enum.sort(),
      beams = Path.wildcard(Path.join(directory, "*.beam")),
      beams != [] do
    {label, options} =
      Map.get(fixture_sets, Path.basename(directory), {Path.basename(directory), []})

    {label, beams, options}
  end

directory = Path.join([root, "findings", name])
File.rm_rf!(directory)
File.mkdir_p!(directory)

runs =
  for {set, beams, options} <- sets,
      {label, analysis} <- analyses,
      do: {"#{set}_#{label}", beams, analysis, options}

# A placed finding as one line: every field, maps written with their keys
# in order, so two runs that place the same finding write the same line.
spell = fn located ->
  located
  |> Map.from_struct()
  |> inspect(
    limit: :infinity,
    printable_limit: :infinity,
    width: :infinity,
    custom_options: [sort_maps: true]
  )
end

runs
|> Task.async_stream(
  fn {label, beams, analysis, options} ->
    {microseconds, {:ok, placed}} = :timer.tc(fn -> analysis.run(beams, options) end)
    lines = placed |> Enum.map(spell) |> Enum.sort()
    File.write!(Path.join(directory, "#{label}.txt"), Enum.map(lines, &[&1, "\n"]))
    "#{label}: #{Float.round(microseconds / 1_000_000, 1)}s, #{length(lines)} findings"
  end,
  max_concurrency: 4,
  timeout: :infinity
)
|> Enum.each(fn {:ok, line} -> IO.puts(line) end)
