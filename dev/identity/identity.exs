# mix run dev/identity/identity.exs NAME
#
# Solves the rules of the checkout it runs in (tensor_shapes.dl and
# emlx.dl) over every set of beams under the data directory's `beams/`,
# and keeps every output and the internal lattice, demand and detection
# relations, each sorted, under `snapshots/NAME`. Compare two snapshots
# with compare.sh: a change to the rules that should change no finding
# must change no row.
#
# The data directory is `_build/identity` of the main checkout of the
# repository this script is in, shared by its worktrees
# (ARGUS_NX_IDENTITY_DIR overrides it). The README next to this script
# says how to fill it.
#
# A change that renames or merges one of the internal relations keeps it
# comparable with a view in the checkout's `_build/probe/identity_views.dl`,
# included after the rules when it exists.
[name] = System.argv()
checkout = File.cwd!()

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

internal = ~w(result_value return_value can_be_negative can_be_positive can_be_zero in_region
  value_class value_dtype can_go_negative sign_demand dtype_demand negative_asked call_error
  nonfinite violation misalignment holds carries_gradient comes_from)
emlx_internal = ~w(placed_on made_on_default carries_mix default_demand)

views = Path.join(checkout, "_build/probe/identity_views.dl")

program = fn rules, relations ->
  file =
    Path.join(
      System.tmp_dir!(),
      "identity_#{Base.url_encode64(:crypto.strong_rand_bytes(9), padding: false)}.dl"
    )

  includes =
    [~s(.include "#{Path.join(checkout, rules)}"\n)] ++
      if(File.exists?(views), do: [~s(.include "#{views}"\n)], else: [])

  File.write!(file, [includes | Enum.map(relations, &".output #{&1}\n")])
  file
end

programs = [
  {"shapes", program.("priv/tensor_shapes.dl", internal)},
  {"emlx", program.("priv/emlx.dl", internal ++ emlx_internal)}
]

# The test suite's fixtures (ARGUS_NX_FIXTURE_BEAMS), solved with the
# options their tests solve them with; any other directory of beams is a
# set of its own, solved with the defaults.
fixture_sets = %{
  "tensor_shapes" => [
    {"fixtures", [unsupported_types: [:f64]]},
    {"fixtures_configured", [unsupported_types: [:f64], float_types: [:f16, :bf16, :f32]]}
  ],
  "emlx" => [{"emlx_fixtures", [unsupported_types: [:c128]]}],
  "emlx_on_exla" => [{"emlx_on_exla", []}]
}

sets =
  for directory <- root |> Path.join("beams/*") |> Path.wildcard() |> Enum.sort(),
      beams = Path.wildcard(Path.join(directory, "*.beam")),
      beams != [],
      {label, options} <-
        Map.get(fixture_sets, Path.basename(directory), [{Path.basename(directory), []}]),
      do: {label, beams, options}

directory = Path.join([root, "snapshots", name])
File.rm_rf!(directory)

solves =
  for {set, beams, options} <- sets,
      {label, file} <- programs,
      do: {"#{set}_#{label}", beams, file, options}

solves
|> Task.async_stream(
  fn {label, beams, file, options} ->
    {microseconds, {:ok, rows}} =
      :timer.tc(fn -> ArgusNxTensorAnalyses.TensorShapes.solve(beams, file, options) end)

    target = Path.join(directory, label)
    File.mkdir_p!(target)

    for {relation, relation_rows} <- rows do
      lines = relation_rows |> Enum.map(&Enum.join(&1, "\t")) |> Enum.sort()
      File.write!(Path.join(target, "#{relation}.tsv"), Enum.map(lines, &[&1, "\n"]))
    end

    row_count =
      rows |> Enum.map(fn {_relation, relation_rows} -> length(relation_rows) end) |> Enum.sum()

    "#{label}: #{Float.round(microseconds / 1_000_000, 1)}s, #{row_count} rows"
  end,
  max_concurrency: 5,
  timeout: :infinity
)
|> Enum.each(fn {:ok, line} -> IO.puts(line) end)
