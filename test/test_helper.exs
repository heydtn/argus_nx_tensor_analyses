ExUnit.start()

defmodule ArgusNxTensorAnalyses.FixtureBeams do
  @moduledoc false

  # With ARGUS_NX_FIXTURE_BEAMS set, a test's compiled fixtures are copied
  # to a directory of that name under it, to solve again outside the suite,
  # as a check that a change to the rules changes no row does.
  def keep(directory, name) do
    if root = System.get_env("ARGUS_NX_FIXTURE_BEAMS") do
      target = Path.join(root, name)
      File.mkdir_p!(target)

      for beam <- Path.wildcard(Path.join(directory, "*.beam")) do
        File.cp!(beam, Path.join(target, Path.basename(beam)))
      end
    end

    :ok
  end
end
