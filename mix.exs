defmodule ArgusNxTensorAnalyses.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/heydtn/argus_nx_tensor_analyses"

  def project do
    [
      app: :argus_nx_tensor_analyses,
      version: @version,
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      description:
        "Argus analyses of Nx code: shapes, values, types and uses of Nx that are wrong, found in compiled modules before the code runs.",
      package: package(),
      docs: docs(),
      source_url: @source_url
    ]
  end

  def application do
    [
      extra_applications: [:logger, :crypto]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_env), do: ["lib"]

  defp deps do
    [
      {:argus_beam, "~> 0.20.1"},
      {:nx, "~> 1.0", only: :test},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib priv docs mix.exs README.md CHANGELOG.md LICENSE .formatter.exs)
    ]
  end

  defp docs do
    [
      main: "readme",
      source_ref: "v#{@version}",
      extras: ["README.md", "docs/checks.md", "docs/how-it-works.md", "CHANGELOG.md"]
    ]
  end
end
