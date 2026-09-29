defmodule ArgusNxTensorAnalyses.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/heydtn/argus_nx_tensor_analyses"

  def project do
    [
      app: :argus_nx_tensor_analyses,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description:
        "Argus analyses of Nx code: tensor shapes that Nx rejects, or that the code does not line up, found in compiled modules.",
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

  defp deps do
    [
      {:argus_beam, "~> 0.20.0"},
      {:jason, "~> 1.4"},
      {:nx, "~> 1.0", only: :test},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib priv mix.exs README.md CHANGELOG.md LICENSE .formatter.exs)
    ]
  end

  defp docs do
    [
      main: "readme",
      source_ref: "v#{@version}",
      extras: ["README.md", "CHANGELOG.md"]
    ]
  end
end
