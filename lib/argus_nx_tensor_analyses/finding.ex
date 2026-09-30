defmodule ArgusNxTensorAnalyses.Finding do
  @moduledoc false
  # A finding of either analysis as its wording says it (`title`, `detail`,
  # `label`, `help`, `frame`), at the call it is about, and how severe each
  # relation's rows are.

  alias Argus.Findings

  # The kinds of an operand nothing checks, which may still never reach
  # where its call is not defined.
  @unchecked ["unchecked_divisor", "unchecked_logarithm", "unchecked_root", "unchecked_domain"]

  # The finding of a row laid out as `[id, func, operation, ..., origin,
  # origin_operation]`, at the severity `severity/3` gives it.
  @spec build(atom(), [String.t()], map()) :: Findings.attrs()
  def build(relation, [id, _func, operation | _rest] = row, wording) do
    [origin, shown] = Enum.take(row, -2)
    build(wording, id, operation, severity(relation, row, wording), {origin, shown})
  end

  # A finding at the call `id` of `operation`, titled by what the wording
  # says the call does, with a frame at the call it comes from: `{origin,
  # origin_operation}` under the wording's frame, or frames already made
  # (`origin_frame/3`).
  @spec build(map(), String.t(), String.t(), Findings.severity(), tuple() | list()) ::
          Findings.attrs()
  def build(wording, id, operation, severity, {origin, shown}),
    do: build(wording, id, operation, severity, origin_frame(wording.frame, origin, shown))

  def build(wording, id, operation, severity, related) when is_list(related) do
    Findings.new(severity, String.trim("#{operation} #{wording.title}"), wording.detail,
      at: Findings.at_instr(id),
      at_label: wording.label,
      help: [wording.help],
      related: related
    )
  end

  # The frame at the call a finding comes from, none for a written value.
  @spec origin_frame(String.t(), String.t(), String.t()) :: [Findings.related()]
  def origin_frame(_frame, "", _shown), do: []

  # A call with no name to show (a fun's call, a child specification's
  # tuple) leaves the frame's text without the colon that would lead to it.
  def origin_frame(frame, origin, ""),
    do: [
      Findings.related(
        frame |> String.trim() |> String.trim_trailing(":"),
        Findings.at_instr(origin)
      )
    ]

  def origin_frame(frame, origin, shown),
    do: [Findings.related(String.trim("#{frame} #{shown}"), Findings.at_instr(origin))]

  # How severe a row of `relation` is: as its wording says, where it says
  # so; else an error where Nx raises whatever path the operands take, or
  # for an operand that is never of a class the call takes, and a warning
  # where it raises only on some; an error for any other call Nx rejects; a
  # warning for what computes otherwise than the code means; and info where
  # only an operand nothing checks, or an input the analysis does not
  # follow, may cause it, or where EMLX rounds a half otherwise.
  @spec severity(atom(), [String.t() | integer()], map()) :: Findings.severity()
  def severity(_relation, _row, %{severity: severity}), do: severity

  def severity(
        :tensor_shape_mismatch,
        [_id, _func, _operation, _kind, _detail, certainty | _rest],
        _wording
      ),
      do: if(certainty == "always", do: :error, else: :warning)

  def severity(:tensor_axis_misalignment, _row, _wording), do: :warning

  def severity(:tensor_nonfinite_result, [_id, _func, _operation, kind | _rest], _wording),
    do: if(kind in @unchecked, do: :info, else: :warning)

  def severity(
        :tensor_type_error,
        [_id, _func, _operation, _kind, _subject, _position, certain | _rest],
        _wording
      ),
      do: if(number(certain) == 1, do: :error, else: :warning)

  def severity(:tensor_call_error, _row, _wording), do: :error

  def severity(
        :tensor_emlx_divergence,
        [_id, _func, _operation, kind, _detail, certain | _rest],
        _wording
      ),
      do: if(kind == "round_half" or number(certain) == 0, do: :info, else: :warning)

  def severity(:tensor_emlx_mixed_backends, _row, _wording), do: :warning

  defp number(value) when is_integer(value), do: value
  defp number(value) when is_binary(value), do: String.to_integer(value)
end
