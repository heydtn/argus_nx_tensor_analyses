# mix run dev/identity/findings.exs NAME
#
# Runs every category of the checkout it runs in
# (`ArgusNxTensorAnalyses.run/3` of `:all`) over every set of beams under
# the data directory's `beams/`, and keeps every finding it places, one
# per line with every field of the placed finding and of the finding
# itself (category, severity, title, detail, help, label, frames, file and
# line), sorted, under `findings/NAME`: a file for each set and category
# that finds any. Compare two runs with compare_findings.sh: a change to
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

sets
|> Task.async_stream(
  fn {set, beams, options} ->
    {microseconds, {:ok, placed}} =
      :timer.tc(fn -> ArgusNxTensorAnalyses.run(beams, [:all], options) end)

    counts =
      for category <- ArgusNxTensorAnalyses.analyses(),
          found = Map.fetch!(placed, category),
          found != [] do
        lines = found |> Enum.map(spell) |> Enum.sort()
        File.write!(Path.join(directory, "#{set}_#{category}.txt"), Enum.map(lines, &[&1, "\n"]))
        "#{category} #{length(lines)}"
      end

    total = placed |> Map.values() |> Enum.map(&length/1) |> Enum.sum()

    "#{set}: #{Float.round(microseconds / 1_000_000, 1)}s, #{total} findings " <>
      "(#{Enum.join(counts, ", ")})"
  end,
  max_concurrency: 4,
  timeout: :infinity
)
|> Enum.each(fn {:ok, line} -> IO.puts(line) end)
