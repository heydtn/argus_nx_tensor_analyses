defmodule ArgusNxTensorAnalyses.TensorShapesTest do
  # Nx is the oracle: each case below is compiled into a module, run for
  # the shape Nx gives it or the error Nx raises, and the analysis of the
  # compiled module has to agree.
  #
  # Not async: `setup_all` loads the compiled fixtures into the VM, and
  # silences the compiler by capturing `:stderr`, which every process
  # shares.
  use ExUnit.Case, async: false

  import ExUnit.CaptureIO

  alias ArgusNxTensorAnalyses.TensorShapes

  unless Argus.Souffle.available?() do
    @moduletag skip: "souffle is not on PATH"
  end

  @fixtures ArgusNxTensorAnalyses.TensorShapesTest.Fixtures
  @lint_fixtures ArgusNxTensorAnalyses.TensorShapesTest.LintFixtures

  @helpers """
    defp local_helper(tensor), do: Nx.add(tensor, Nx.iota({4}))
    defp make_mask, do: Nx.iota({1, 3})
    defp grow(tensor, 0), do: tensor
    defp grow(tensor, times), do: grow(Nx.concatenate([tensor, tensor]), times - 1)
    def pick(0), do: Nx.iota({3})
    def pick(_), do: Nx.iota({4})
    defn defn_kernel(left, right), do: left * right + 1
    defn defn_reshape(tensor), do: Nx.reshape(tensor, {:auto, 2})
    defn defn_numbers(tensor), do: tensor + 2 * 3
    defn defn_outer(tensor), do: defn_kernel(tensor, Nx.iota({5}))
    defn defn_compare(left, right), do: left > right and -right <= 3
  """

  # Each case is a function body. A tagged case expects less than full
  # agreement: `:some_path`, only findings marked on_some_path, for branches
  # the analysis does not tell apart; `:unknown`, no shape and no finding;
  # `:no_finding`, no finding, whatever shapes are derived.
  @cases [
    # element-wise
    "Nx.add(Nx.iota({2, 3}), Nx.iota({3}))",
    "Nx.add(Nx.iota({2, 3}), Nx.iota({2}))",
    "Nx.multiply(Nx.iota({2, 1, 4}), Nx.iota({3, 1}))",
    "Nx.subtract(Nx.iota({2, 3}, names: [:a, :b]), Nx.iota({2, 3}, names: [:a, :c]))",
    "Nx.divide(Nx.iota({2, 3}, names: [:a, nil]), Nx.iota({3}, names: [:b]))",
    "Nx.add(Nx.iota({2, 3}, names: [:a, :b]), Nx.iota({2, 1}, names: [:a, :c]))",
    "Nx.pow(Nx.iota({4}), 2)",
    "Nx.add(Nx.iota({2, 3}), 1.5)",
    "Nx.exp(Nx.iota({2, 3}, names: [:x, :y]))",
    "Nx.negate(Nx.iota({2, 2}, names: [:p, :q]))",
    "Nx.equal(Nx.tensor([1, 2, 3]), Nx.tensor([[1], [2]]))",
    "Nx.max(Nx.tensor([[1, 2]]), Nx.tensor([1, 2, 3]))",
    "Nx.bitwise_and(Nx.iota({4}), Nx.iota({2, 4}))",
    "Nx.logical_or(Nx.iota({2, 3}), Nx.iota({3, 2}))",
    "Nx.log(Nx.iota({2, 2}), 2)",
    "Nx.complex(Nx.iota({3}), Nx.iota({2, 1}))",
    "Nx.as_type(Nx.iota({2, 5}), :f32)",
    "Nx.donatable(Nx.iota({2, 3}))",
    "Nx.select(Nx.iota({2, 3}), Nx.iota({3}), 0)",
    "Nx.select(Nx.iota({3}), Nx.iota({2, 3}), 0)",
    "Nx.select(1, Nx.iota({2, 3}), Nx.iota({2, 1}))",
    "Nx.select(1, Nx.iota({2, 3}), Nx.iota({4}))",
    "Nx.clip(Nx.iota({2, 3}), 0, 4)",
    "Nx.clip(Nx.iota({2, 3}), Nx.iota({3}), 4)",
    "Nx.clip(Nx.iota({2, 3}), 0, Nx.vectorize(Nx.iota({2}), :x))",
    "Nx.fill(Nx.iota({2, 3}), 7)",
    "Nx.sort(Nx.iota({2, 3}), axis: 1)",
    "Nx.sort(Nx.iota({2, 3}), axis: 2)",
    "Nx.argsort(Nx.iota({2, 3}, names: [:a, :b]), axis: :b)",
    "Nx.argsort(Nx.iota({2, 3}, names: [:a, :b]), axis: :c)",
    "Nx.cumulative_sum(Nx.iota({4, 2}), axis: -1)",
    "Nx.cumulative_max(Nx.iota({2, 3}), axis: 3)",
    "Nx.reverse(Nx.iota({2, 3}), axes: [0, 1])",
    "Nx.reverse(Nx.iota({2, 3}), axes: [3])",
    "Nx.tril(Nx.iota({3, 3}))",
    "Nx.tril(Nx.iota({3}))",
    "Nx.all_close(Nx.iota({2, 3}), Nx.iota({3}))",
    "Nx.all_close(Nx.iota({2, 3}), Nx.iota({2}))",
    # reductions
    "Nx.sum(Nx.iota({2, 3, 4}), axes: [1])",
    "Nx.sum(Nx.iota({2, 3, 4}), axes: [1], keep_axes: true)",
    "Nx.sum(Nx.iota({2, 3}))",
    "Nx.sum(Nx.iota({2, 3}), keep_axes: true)",
    "Nx.sum(Nx.iota({2, 3}), axes: [2])",
    "Nx.sum(Nx.iota({2, 3}), axes: [0, 0])",
    "Nx.sum(Nx.iota({2, 3}, names: [:a, :b]), axes: [:a, 1])",
    "Nx.sum(Nx.iota({2, 3}, names: [:a, :b]), axes: [:z])",
    "Nx.mean(Nx.iota({2, 3}, names: [:r, :c]), axes: [:r])",
    "Nx.product(Nx.iota({2, 3}, names: [:a, :b]), axes: [:b], keep_axes: true)",
    "Nx.reduce_max(Nx.iota({2, 3}), axes: [-1])",
    "Nx.logsumexp(Nx.iota({2, 3}), axes: [1])",
    "Nx.variance(Nx.iota({2, 3}), axes: [0], keep_axes: true)",
    "Nx.standard_deviation(Nx.iota({4}))",
    "Nx.all(Nx.iota({2, 3}), axes: [1])",
    "Nx.argmax(Nx.iota({2, 3}), axis: 1)",
    "Nx.argmax(Nx.iota({2, 3}), axis: 1, keep_axis: true)",
    "Nx.argmin(Nx.iota({2, 3}))",
    "Nx.argmax(Nx.iota({2, 3}), axis: 5)",
    "Nx.median(Nx.iota({2, 4}), axis: 1)",
    "Nx.mode(Nx.iota({2, 3}), keep_axis: true)",
    "Nx.reduce(Nx.iota({2, 3}), 0, [axes: [0]], &Nx.add/2)",
    "Nx.reduce(Nx.iota({2, 3}), Nx.iota({2}), &Nx.add/2)",
    "Nx.weighted_mean(Nx.iota({2, 3}), Nx.iota({2, 3}), axes: [1])",
    "Nx.weighted_mean(Nx.iota({2, 3}), Nx.iota({3}))",
    "Nx.LinAlg.norm(Nx.iota({2, 3}, type: :f32))",
    "Nx.LinAlg.norm(Nx.iota({2, 3, 4}, type: :f32))",
    # reshaping
    "Nx.reshape(Nx.iota({2, 3}), {3, 2})",
    "Nx.reshape(Nx.iota({2, 3}), {4, 2})",
    "Nx.reshape(Nx.iota({2, 3}), {:auto, 2})",
    "Nx.reshape(Nx.iota({2, 3}), {:auto, 4})",
    "Nx.reshape(Nx.iota({2, 3}), {6}, names: [:flat])",
    "Nx.reshape(Nx.iota({2, 3}), {6}, names: [:a, :b])",
    "Nx.reshape(Nx.iota({2, 3}), Nx.shape(Nx.iota({3, 2})))",
    "Nx.reshape(Nx.iota({2, 3}), Nx.iota({3, 2}, names: [:u, :w]))",
    "Nx.reshape(Nx.iota({2, 3}, names: [:a, :b]), {2, 3})",
    "Nx.reshape(Nx.iota({2, 3}, names: [:a, :b]), {3, 2}, names: [:x, :y])",
    "Nx.flatten(Nx.iota({2, 3, 4}))",
    "Nx.flatten(Nx.iota({2, 3, 4}), axes: [1, 2])",
    "Nx.flatten(Nx.iota({2, 3, 4}), axes: [0, 2])",
    "Nx.flatten(Nx.iota({2, 3, 4}, names: [:a, :b, :c]), axes: [:b, :c])",
    "Nx.new_axis(Nx.iota({2, 3}), 1, :z)",
    "Nx.new_axis(Nx.iota({2, 3}), -1)",
    "Nx.new_axis(Nx.iota({2, 3}), 4)",
    "Nx.new_axis(Nx.iota({2, 3}, names: [:a, :b]), 0, :n)",
    "Nx.squeeze(Nx.iota({1, 3, 1}))",
    "Nx.squeeze(Nx.iota({1, 3, 1}), axes: [0])",
    "Nx.squeeze(Nx.iota({1, 3, 1}), axes: [1])",
    "Nx.squeeze(Nx.iota({1, 3}, names: [:a, :b]), axes: [:a])",
    "Nx.transpose(Nx.iota({2, 3, 4}))",
    "Nx.transpose(Nx.iota({2, 3}, names: [:a, :b]))",
    "Nx.transpose(Nx.iota({2, 3, 4}, names: [:a, :b, :c]), axes: [:c, :a, :b])",
    "Nx.transpose(Nx.iota({2, 3, 4}), axes: [1, 0])",
    "Nx.broadcast(Nx.iota({3}), {2, 3})",
    "Nx.broadcast(Nx.iota({3}), {3, 2})",
    "Nx.broadcast(Nx.iota({3}), {3, 2}, axes: [0])",
    "Nx.broadcast(Nx.iota({2}, names: [:a]), {3, 2}, axes: [:a])",
    "Nx.broadcast(Nx.iota({2, 3}), {3})",
    "Nx.broadcast(Nx.iota({1, 3}), {2, 3}, names: [:a, :b])",
    "Nx.broadcast(0.0, {2, 2})",
    "Nx.broadcast(Nx.iota({3}), Nx.iota({2, 3}, names: [:r, :c]))",
    "Nx.tile(Nx.iota({2, 3}), [2, 1])",
    "Nx.tile(Nx.iota({3}), [2, 2])",
    "Nx.pad(Nx.iota({2, 3}), 0, [{1, 1, 0}, {0, 0, 1}])",
    "Nx.pad(Nx.iota({2, 3}), 0, [{1, 1, 0}])",
    "Nx.pad(Nx.iota({2, 3}), Nx.iota({2}), [{1, 1, 0}, {0, 0, 0}])",
    "Nx.pad_outer(Nx.iota({2, 3}), 0, [{1, 1}, {0, 2}])",
    "Nx.pad_outer(Nx.iota({2, 3}), :reflect, [{1, 1}, {0, 2}])",
    "Nx.pad_outer(Nx.iota({2, 3}), :reflect, [{1, 1}])",
    # slicing and indexing
    "Nx.slice(Nx.iota({5, 5}), [1, 1], [2, 3])",
    "Nx.slice(Nx.iota({5, 5}), [0, 0], [5, 5], strides: [2, 3])",
    "Nx.slice(Nx.iota({5, 5}), [0, 0], [6, 1])",
    "Nx.slice(Nx.iota({5, 5}), [0], [2])",
    "Nx.slice(Nx.iota({100, 100}), [0, 0], [64, 64])",
    "Nx.slice(Nx.iota({100, 100}), [0, 0], [64, 101])",
    "Nx.slice_along_axis(Nx.iota({4, 6}), 1, 3, axis: 1)",
    "Nx.slice_along_axis(Nx.iota({4, 6}), 0, 7, axis: 1)",
    "Nx.slice_along_axis(Nx.iota({4, 6}), 0, 4, axis: 1, strides: 2)",
    "Nx.slice_along_axis(Nx.iota({4, 6}, names: [:r, :c]), 0, 2, axis: :c)",
    {:unknown, "{left, _right} = Nx.split(Nx.iota({5, 2}), 2)\nleft"},
    "{left, _right} = Nx.split(Nx.iota({5, 2}), 5)\nleft",
    "Nx.put_slice(Nx.iota({4, 4}), [0, 0], Nx.iota({2, 2}))",
    "Nx.put_slice(Nx.iota({4, 4}), [0, 0], Nx.iota({5, 2}))",
    "Nx.put_slice(Nx.iota({4, 4}), [0, 0], Nx.iota({2}))",
    "Nx.put_slice(Nx.iota({4, 4}, names: [:a, nil]), [0, 0], Nx.iota({2, 2}, names: [nil, :b]))",
    "Nx.take(Nx.iota({3, 4}), Nx.tensor([0, 2]), axis: 1)",
    "Nx.take(Nx.iota({3, 4}), Nx.tensor([[0, 2], [1, 1]]))",
    "Nx.take(Nx.iota({3, 4}, names: [:r, :c]), Nx.tensor([0, 1], names: [:i]), axis: 1)",
    "Nx.take(Nx.iota({3, 4}, names: [:r, :c]), Nx.tensor([[0, 1]], names: [:i, :j]), axis: 1)",
    "Nx.take_along_axis(Nx.iota({3, 4}), Nx.tensor([[0], [1], [2]]), axis: 1)",
    "Nx.take_along_axis(Nx.iota({3, 4}), Nx.tensor([[0, 1]]), axis: 1)",
    "Nx.gather(Nx.iota({3, 4}), Nx.tensor([[0, 1], [2, 3]]))",
    "Nx.gather(Nx.iota({3, 4}), Nx.tensor([[0], [2]]))",
    "Nx.gather(Nx.iota({3, 4}), Nx.tensor([[0], [2]]), axes: [1])",
    "Nx.gather(Nx.iota({3}), Nx.tensor([[0, 1]]))",
    "Nx.indexed_add(Nx.iota({3, 4}), Nx.tensor([[0, 1], [2, 3]]), Nx.tensor([1, 2]))",
    "Nx.indexed_put(Nx.iota({3, 4}), Nx.tensor([[0, 1], [2, 3]]), Nx.tensor([1, 2, 3]))",
    "Nx.indexed_put(Nx.iota({3, 4}), Nx.tensor([[0], [2]]), Nx.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))",
    "Nx.indexed_put(Nx.iota({3, 4}), Nx.tensor([[0], [2]]), Nx.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]))",
    "Nx.indexed_add(Nx.iota({3, 4}), Nx.tensor([0, 1]), 5)",
    "Nx.indexed_add(Nx.iota({3, 4}), Nx.tensor([0, 1, 2]), 5)",
    # joining
    "Nx.concatenate([Nx.iota({2, 3}), Nx.iota({2, 4})], axis: 1)",
    "Nx.concatenate([Nx.iota({2, 3}), Nx.iota({3, 4})], axis: 1)",
    "Nx.concatenate([Nx.iota({2, 3}), Nx.iota({1, 3}), Nx.iota({4, 3})])",
    "Nx.concatenate([Nx.iota({2, 3}, names: [:a, :b]), Nx.iota({2, 3}, names: [:a, :c])])",
    "Nx.concatenate([Nx.iota({2, 3}, names: [:a, nil]), Nx.iota({2, 3}, names: [nil, :b])])",
    "Nx.stack([Nx.iota({2, 3}), Nx.iota({2, 3})], axis: 1, name: :s)",
    "Nx.stack([Nx.iota({2, 3}), Nx.iota({3, 2})])",
    "Nx.stack([Nx.iota({2}), Nx.iota({2}), Nx.iota({2})], axis: -1)",
    "Nx.stack([Nx.iota({2}, names: [:a]), Nx.iota({2})])",
    # contracting
    "Nx.dot(Nx.iota({2, 3}), Nx.iota({3, 4}))",
    "Nx.dot(Nx.iota({2, 3}), Nx.iota({4, 3}))",
    "Nx.dot(Nx.iota({3}), Nx.iota({3}))",
    "Nx.dot(Nx.iota({2, 3, 4}), Nx.iota({5, 4, 6}))",
    "Nx.dot(Nx.iota({2, 3}), Nx.iota({3}))",
    "Nx.dot(2, Nx.iota({2, 3}))",
    "Nx.dot(Nx.iota({2, 3, 4}), [2], [0], Nx.iota({2, 4, 5}), [1], [0])",
    "Nx.dot(Nx.iota({2, 3, 4}), [2], [0], Nx.iota({3, 4, 5}), [1], [0])",
    "Nx.dot(Nx.iota({2, 3}), [0], Nx.iota({2, 5}), [0])",
    "Nx.dot(Nx.iota({2, 3}), [1], Nx.iota({2, 5}), [0])",
    "Nx.dot(Nx.iota({2, 3}, names: [:r, :c]), Nx.iota({3, 2}, names: [:c, :r]))",
    "Nx.dot(Nx.iota({2, 3}, names: [:r, :c]), Nx.iota({3, 4}, names: [:c, :k]))",
    "Nx.dot(Nx.iota({2, 3, 4}, names: [:b, :i, :j]), [2], [0], Nx.iota({2, 4, 5}, names: [:b, :j, :k]), [1], [0])",
    "matrix = Nx.iota({4, 3})\nNx.dot(matrix, Nx.transpose(matrix))",
    "matrix = Nx.iota({4, 3})\nNx.dot(matrix, matrix)",
    "Nx.outer(Nx.iota({2, 3}), Nx.iota({4}))",
    "Nx.conv(Nx.iota({1, 2, 5}, type: :f32), Nx.iota({3, 2, 2}, type: :f32))",
    "Nx.conv(Nx.iota({1, 2, 5}, type: :f32), Nx.iota({3, 1, 2}, type: :f32))",
    "Nx.conv(Nx.iota({1, 2, 5, 5}, type: :f32), Nx.iota({4, 2, 3, 3}, type: :f32), strides: [2, 2], padding: :same)",
    "Nx.conv(Nx.iota({1, 2, 5}, type: :f32), Nx.iota({3, 2, 7}, type: :f32))",
    "Nx.conv(Nx.iota({1, 2, 5}, names: [:n, :c, :w], type: :f32), Nx.iota({3, 2, 2}, type: :f32))",
    # windows
    "Nx.window_sum(Nx.iota({4, 6}), {2, 3})",
    "Nx.window_sum(Nx.iota({4, 6}), {2, 3}, strides: [2, 3])",
    "Nx.window_max(Nx.iota({4, 6}), {2, 3}, padding: :same)",
    "Nx.window_max(Nx.iota({4, 6}, names: [:h, :w]), {2, 2})",
    "Nx.window_mean(Nx.iota({4, 6}), {2})",
    "Nx.window_min(Nx.iota({4, 6}), {5, 1})",
    "Nx.window_product(Nx.iota({4, 6}), {1, 6})",
    "Nx.window_sum(Nx.iota({4, 6}), {2, 2}, window_dilations: [2, 1], padding: [{0, 0}, {1, 1}])",
    "Nx.window_scatter_max(Nx.iota({4, 6}), Nx.iota({2, 2}), 0, {2, 3}, strides: [2, 3])",
    "Nx.window_scatter_max(Nx.iota({4, 6}), Nx.iota({3, 3}), 0, {2, 3}, strides: [2, 3])",
    "Nx.window_reduce(Nx.iota({4, 6}), 0, {2, 2}, [strides: [2, 2]], fn element, accumulator -> Nx.add(element, accumulator) end)",
    # along an axis
    "{values, _indices} = Nx.top_k(Nx.iota({2, 5}), k: 3)\nvalues",
    "{values, _indices} = Nx.top_k(Nx.iota({2, 5}), k: 6)\nvalues",
    "Nx.fft(Nx.iota({2, 3}), length: 4)",
    "Nx.fft(Nx.iota({2, 5}), length: :power_of_two)",
    "Nx.fft(Nx.tensor(1))",
    "Nx.ifft(Nx.iota({2, 3}), axis: 0, length: 5)",
    "Nx.rfft(Nx.iota({2, 6}))",
    "Nx.irfft(Nx.iota({2, 4}))",
    "Nx.fft2(Nx.iota({3, 4}))",
    "Nx.fft2(Nx.iota({2, 3, 4}), lengths: [4, 8])",
    "Nx.diff(Nx.iota({2, 5}))",
    "Nx.diff(Nx.iota({2, 5}), order: 2, axis: 0)",
    # matrices
    "Nx.LinAlg.determinant(Nx.iota({2, 3, 3}, type: :f32))",
    "Nx.LinAlg.determinant(Nx.iota({2, 3}, type: :f32))",
    "Nx.LinAlg.determinant(Nx.iota({3, 2}, type: :f32))",
    "Nx.LinAlg.invert(Nx.eye(3, type: :f32))",
    "Nx.LinAlg.solve(Nx.eye(3, type: :f32), Nx.iota({3}, type: :f32))",
    "Nx.LinAlg.solve(Nx.eye(3, type: :f32), Nx.iota({4}, type: :f32))",
    "Nx.LinAlg.triangular_solve(Nx.eye(3, type: :f32), Nx.iota({3, 2}, type: :f32))",
    "Nx.LinAlg.triangular_solve(Nx.eye(3, type: :f32), Nx.iota({2, 3}, type: :f32))",
    "Nx.LinAlg.triangular_solve(Nx.eye(3, type: :f32), Nx.iota({2, 3}, type: :f32), left_side: false)",
    "Nx.LinAlg.adjoint(Nx.iota({2, 3}, names: [:r, :c]))",
    "Nx.LinAlg.pinv(Nx.iota({2, 3}, type: :f32))",
    "Nx.LinAlg.matrix_power(Nx.eye(3), 2)",
    "Nx.LinAlg.matrix_power(Nx.iota({2, 3}), 2)",
    "Nx.LinAlg.cholesky(Nx.iota({2, 3}, type: :f32))",
    "Nx.LinAlg.lu(Nx.iota({2, 3}, type: :f32))",
    "Nx.LinAlg.eigh(Nx.iota({2, 3}, type: :f32))",
    "Nx.LinAlg.qr(Nx.iota({3}, type: :f32))",
    "Nx.LinAlg.matrix_rank(Nx.iota({2, 3}, type: :f32))",
    "Nx.LinAlg.matrix_rank(Nx.iota({2}, type: :f32))",
    "Nx.LinAlg.least_squares(Nx.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), Nx.iota({3}, type: :f32))",
    "Nx.LinAlg.least_squares(Nx.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), Nx.iota({2}, type: :f32))",
    "Nx.covariance(Nx.iota({4, 3}, type: :f32))",
    "Nx.covariance(Nx.iota({2, 4, 3}, type: :f32))",
    "Nx.covariance(Nx.iota({4}, type: :f32))",
    "Nx.covariance(Nx.iota({4, 3}, type: :f32), Nx.iota({3}, type: :f32))",
    "Nx.take_diagonal(Nx.iota({3, 4}))",
    "Nx.take_diagonal(Nx.iota({3, 4}), offset: 2)",
    "Nx.take_diagonal(Nx.iota({3, 4}), offset: 4)",
    "Nx.make_diagonal(Nx.iota({3}), offset: -1)",
    "Nx.make_diagonal(Nx.iota({2, 3}))",
    "Nx.put_diagonal(Nx.iota({3, 3}), Nx.tensor([1, 2, 3]))",
    "Nx.put_diagonal(Nx.iota({3, 3}), Nx.tensor([1, 2]))",
    # creating
    "Nx.tensor([[1, 2, 3], [4, 5, 6]])",
    "Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])",
    "Nx.tensor(1.5)",
    "Nx.tensor([[1, 2]], names: [:x])",
    "Nx.tensor([[65, 66], [67, 68]])",
    "Nx.f32([[1.0, 2.0]])",
    "Nx.u8([[1, 2, 3]])",
    "Nx.iota({2, 3}, names: [:a, :b])",
    "Nx.iota({2, 3}, axis: 2)",
    "Nx.iota({2}, vectorized_axes: [x: 3])",
    "Nx.eye(3)",
    "Nx.eye({2, 3, 3})",
    "Nx.eye({3})",
    "Nx.template({2, 3}, :f32)",
    "Nx.template({2, 3}, :f32, names: [:a, :b])",
    "Nx.tri(3, 4)",
    "Nx.linspace(0, 10, n: 5)",
    "Nx.linspace(Nx.tensor([0, 1]), Nx.tensor([1, 2, 3]), n: 5)",
    "Nx.Random.key(42)",
    "Nx.Random.split(Nx.Random.key(42), parts: 3)",
    "Nx.Random.fold_in(Nx.Random.key(1), 7)",
    "Nx.Random.uniform_split(Nx.Random.key(1), 0.0, 1.0, shape: {2, 5}, names: [:a, :b])",
    "Nx.Random.normal_split(Nx.Random.key(1), 0.0, 1.0, shape: {3})",
    "Nx.Random.gumbel_split(Nx.Random.key(1), shape: {2, 2})",
    "Nx.Random.randint_split(Nx.Random.key(1), 0, 10, shape: {4})",
    # names and vectorization
    "Nx.rename(Nx.iota({2, 3}), [:x, nil])",
    "Nx.rename(Nx.iota({2, 3}), [:x])",
    "Nx.vectorize(Nx.iota({2, 3}), :x)",
    "Nx.vectorize(Nx.iota({2, 3}), [:x, :y, :z])",
    "Nx.vectorize(Nx.iota({2, 3}), x: 3)",
    "Nx.vectorize(Nx.vectorize(Nx.iota({2, 3, 4}), :x), :y)",
    "Nx.devectorize(Nx.vectorize(Nx.iota({2, 3, 4}), [:x, :y]))",
    "Nx.devectorize(Nx.vectorize(Nx.iota({2, 3}, names: [:a, :b]), :x), keep_names: false)",
    "Nx.devectorize(Nx.add(Nx.vectorize(Nx.iota({2, 3}), :x), Nx.vectorize(Nx.iota({4, 1}), :y)))",
    "Nx.revectorize(Nx.iota({2, 3, 4}) |> Nx.vectorize(x: 2, y: 3), [a: :auto])",
    "Nx.revectorize(Nx.iota({2, 3, 4}) |> Nx.vectorize(x: 2, y: 3), [a: 2], target_shape: {:auto, 2}, target_names: [:p, :q])",
    "Nx.revectorize(Nx.iota({2, 3, 4}) |> Nx.vectorize(x: 2, y: 3), [a: 5])",
    "Nx.add(Nx.vectorize(Nx.iota({2, 3}), :x), Nx.vectorize(Nx.iota({4, 3}), :y))",
    "Nx.add(Nx.vectorize(Nx.iota({2, 3}), :x), Nx.vectorize(Nx.iota({4, 3}), :x))",
    "Nx.add(Nx.vectorize(Nx.iota({2, 3}), :x), Nx.iota({4, 3}))",
    "Nx.multiply(Nx.vectorize(Nx.iota({2, 1}), :x), Nx.vectorize(Nx.iota({1, 3}), :x))",
    "Nx.sum(Nx.vectorize(Nx.iota({2, 3}), :x))",
    "Nx.sum(Nx.vectorize(Nx.iota({2, 3, 4}), :x), axes: [0])",
    "Nx.reduce_max(Nx.vectorize(Nx.iota({2, 3}), :x), keep_axes: true)",
    "Nx.argmax(Nx.vectorize(Nx.iota({2, 3}), :x))",
    "Nx.argmax(Nx.vectorize(Nx.iota({2, 3, 4}), :x), keep_axis: true)",
    "Nx.reshape(Nx.vectorize(Nx.iota({2, 3, 2}), :x), {6})",
    "Nx.reshape(Nx.vectorize(Nx.iota({2, 6}), :x), {4, 2})",
    "Nx.new_axis(Nx.vectorize(Nx.iota({2, 3}), :x), 0)",
    "Nx.transpose(Nx.vectorize(Nx.iota({2, 3, 4}), :x))",
    "Nx.broadcast(Nx.vectorize(Nx.iota({2, 3}), :x), {4, 3})",
    "Nx.pad(Nx.vectorize(Nx.iota({2, 3}), :x), 0, [{1, 1, 0}])",
    "Nx.select(Nx.vectorize(Nx.iota({2, 3}), :x), 1, 0)",
    "Nx.dot(Nx.vectorize(Nx.iota({2, 3}), :x), Nx.iota({3, 4}))",
    "Nx.take(Nx.vectorize(Nx.iota({2, 3}), :x), Nx.tensor([0, 1]))",
    "Nx.take_along_axis(Nx.vectorize(Nx.iota({2, 3}), :x), Nx.tensor([0, 2]))",
    "Nx.gather(Nx.vectorize(Nx.iota({2, 3}), :x), Nx.tensor([[0], [1]]))",
    "Nx.slice(Nx.vectorize(Nx.iota({2, 5}), :x), [1], [2])",
    "Nx.put_slice(Nx.vectorize(Nx.iota({2, 5}), :x), [0], Nx.tensor([1, 2]))",
    "Nx.concatenate([Nx.vectorize(Nx.iota({2, 3}), :x), Nx.vectorize(Nx.iota({2, 4}), :x)])",
    "Nx.stack([Nx.vectorize(Nx.iota({2, 3}), :x), Nx.vectorize(Nx.iota({2, 3}), :x)])",
    "Nx.window_sum(Nx.vectorize(Nx.iota({2, 5}), :x), {2})",
    "Nx.shape(Nx.vectorize(Nx.iota({2, 3}), :x)) |> Nx.iota()",
    # values across statements and calls
    "matrix = Nx.iota({2, 3})\ntransposed = Nx.transpose(matrix)\nNx.add(matrix, transposed)",
    "matrix = Nx.iota({2, 3})\nNx.add(matrix, Nx.transpose(Nx.transpose(matrix)))",
    "Nx.iota({2, 3}) |> Nx.add(Nx.iota({4})) |> Nx.sum()",
    "local_helper(Nx.iota({2, 3}))",
    "local_helper(Nx.iota({2, 4}))",
    "Nx.add(make_mask(), Nx.iota({4, 3}))",
    "Nx.add(make_mask(), Nx.iota({4, 4}))",
    "defn_kernel(Nx.iota({2, 3}), Nx.iota({2}))",
    "defn_kernel(Nx.iota({2, 3}), Nx.iota({3}))",
    "defn_reshape(Nx.iota({3}))",
    "defn_reshape(Nx.iota({4}))",
    "defn_numbers(Nx.iota({2}))",
    "defn_outer(Nx.iota({2, 3}))",
    "defn_outer(Nx.iota({2, 5}))",
    "defn_compare(Nx.iota({2, 3}), Nx.iota({3}))",
    "defn_compare(Nx.iota({2, 3}), Nx.iota({2}))",
    {:some_path,
     "matrix = Nx.iota({2, 3})\nrow = if :erlang.phash2(1) > 0, do: Nx.iota({3}), else: Nx.iota({4})\nNx.add(matrix, row)"},
    "Nx.add(Nx.iota({2, 3}), pick(0))",
    "Nx.add(Nx.iota({2, 3}), pick(1))",
    {:no_finding, "grow(Nx.iota({2}), 3)"},
    {:unknown, "axes = Enum.to_list(0..0)\nNx.sum(Nx.iota({2, 3}), axes: axes)"},
    {:unknown, "shape = List.to_tuple(Enum.to_list(2..3))\nNx.iota(shape)"}
  ]

  # Code Nx accepts where the code does not line axes up, each beside a
  # neighbor that does. A case is a function of `config` (a map of sizes),
  # `t` (a tensor of a shape not known) and `shaper` and `other` (values of
  # `Shaper`'s implementations). It expects the misalignment of a kind, a
  # mismatch of a kind (Nx raising on some path), or nothing (`:quiet`).
  @lint_helpers """
    alias ArgusNxTensorAnalyses.TensorShapesTest.Shaper
    defp over_heads(config), do: Nx.iota({config.heads}, names: [:heads])
    defp heads_by(config, tensor), do: Nx.reshape(tensor, {config.heads, :auto})
    defp length_of(x), do: Nx.sqrt(Nx.sum(Nx.multiply(x, x)))
    defp divided(x, divisor), do: Nx.divide(x, divisor)
    defp magnitude(x), do: Nx.sqrt(Nx.sum(Nx.multiply(x, x)))
    defp scaled_by(x, divisor), do: Nx.divide(x, divisor)
    def divides_by_caller_absolute(t), do: scaled_by(t, Nx.abs(t))
    defp shrunk_by(x, divisor), do: Nx.divide(x, divisor)
    def divides_by_caller_positive(t), do: shrunk_by(t, Nx.add(Nx.abs(t), 1))
    def differentiates_magnitude(t), do: Nx.Defn.grad(t, &magnitude/1)
    def divides_by_absolute(config), do: divided(1, Nx.abs(Nx.iota({config.heads})))
    defp pick_axis(config, which) do
      case which do
        :rows -> Nx.iota({config.rows})
        :cols -> Nx.iota({config.cols})
      end
    end
  """

  @lint_cases_base [
    # size variables that disagree where the sizes meet
    {{:misaligned, "size_variables"},
     "Nx.add(Nx.iota({config.heads}), Nx.iota({config.kv_heads}))"},
    {:quiet, "Nx.add(Nx.iota({config.heads}), Nx.iota({config.heads}))"},
    {:quiet, "Nx.add(Nx.iota({config.heads}), Nx.iota({1}))"},
    {{:misaligned, "size_variables"}, "Nx.add(Nx.iota({config.heads}), Nx.iota({1, 4}))"},
    {:quiet, "Nx.add(Nx.iota({config.heads, 3}), Nx.iota({3}))"},
    {{:misaligned, "size_variables"},
     "Nx.add(Nx.iota({config.heads * config.dim}), Nx.iota({config.hidden}))"},
    {:quiet,
     "Nx.add(Nx.iota({config.heads * config.dim}), Nx.iota({config.dim * config.heads}))"},
    {{:misaligned, "size_variables"},
     "Nx.reshape(Nx.iota({config.hidden}), {config.heads, config.dim})"},
    {:quiet, "Nx.reshape(Nx.iota({config.heads, config.dim}), {config.dim, config.heads})"},
    {:quiet,
     "Nx.iota({config.heads, config.dim}) |> Nx.reshape({config.heads, :auto}) |> Nx.add(Nx.iota({config.heads, config.dim}))"},
    {{:misaligned, "size_variables"},
     "Nx.iota({config.heads, config.dim}) |> Nx.reshape({config.heads, :auto}) |> Nx.add(Nx.iota({config.heads, config.head_dim}))"},
    {{:misaligned, "size_variables"},
     "Nx.dot(Nx.iota({2, config.dim}), Nx.iota({config.head_dim, 3}))"},
    {:quiet, "Nx.dot(Nx.iota({2, config.dim}), Nx.iota({config.dim, 3}))"},
    {{:misaligned, "size_variables"},
     "Nx.concatenate([Nx.iota({config.rows, 2}), Nx.iota({config.cols, 3})], axis: 1)"},
    {:quiet, "Nx.concatenate([Nx.iota({config.rows, 2}), Nx.iota({config.rows, 3})], axis: 1)"},
    {{:misaligned, "size_variables"},
     "Nx.broadcast(Nx.iota({config.dim}), {2, config.head_dim})"},
    {:quiet, "Nx.broadcast(Nx.iota({config.dim}), {2, config.dim})"},
    {{:misaligned, "size_variables"},
     "Nx.add(Nx.vectorize(Nx.iota({config.heads, 2}), :x), Nx.vectorize(Nx.iota({config.kv_heads, 2}), :x))"},
    {:quiet,
     "Nx.add(Nx.vectorize(Nx.iota({config.heads, 2}), :x), Nx.vectorize(Nx.iota({config.heads, 2}), :x))"},
    # sizes only the tensor that arrives determines are no variables of the code's
    {:quiet, "Nx.add(Nx.iota({Nx.axis_size(t, 0)}), Nx.iota({config.heads}))"},
    {:quiet, "Nx.add(t, Nx.iota({config.heads}))"},
    {{:misaligned, "size_variables"},
     "Nx.add(Nx.iota({Nx.axis_size(Nx.iota({config.heads}), 0)}), Nx.iota({config.kv_heads}))"},
    # across calls, tuples, maps and the branch a literal picks
    {{:misaligned, "size_variables"},
     "Nx.add(over_heads(config), Nx.iota({config.kv_heads}, names: [:heads]))"},
    {:quiet, "Nx.add(over_heads(config), Nx.iota({config.heads}, names: [:heads]))"},
    {{:misaligned, "size_variables"},
     "{rows, cols} = {config.rows, config.cols}\nNx.add(Nx.iota({rows}), Nx.iota({cols}))"},
    {:quiet,
     "sizes = %{rows: config.rows}\nNx.add(Nx.iota({sizes.rows}), Nx.iota({config.rows}))"},
    {{:misaligned, "size_variables"},
     "config |> heads_by(Nx.iota({config.hidden})) |> Nx.add(Nx.iota({config.kv_heads, 1}))"},
    {:quiet, "config |> pick_axis(:rows) |> Nx.add(Nx.iota({config.rows}))"},
    {{:misaligned, "size_variables"},
     "config |> pick_axis(:cols) |> Nx.add(Nx.iota({config.rows}))"},
    # named axes meeting axes with no name
    {{:misaligned, "unnamed_axis"},
     "Nx.add(Nx.iota({2, 3}, names: [:rows, :cols]), Nx.iota({2, 3}))"},
    {:quiet,
     "Nx.add(Nx.iota({2, 3}, names: [:rows, :cols]), Nx.iota({2, 3}, names: [:rows, :cols]))"},
    {:quiet,
     "Nx.add(Nx.iota({2, 3}, names: [:rows, :cols]), Nx.iota({1, 3}, names: [nil, :cols]))"},
    {:quiet, "Nx.multiply(Nx.iota({2, 3}, names: [:rows, :cols]), 2.0)"},
    {{:misaligned, "unnamed_axis"},
     "Nx.multiply(Nx.iota({2, 3}, names: [:rows, :cols]), Nx.iota({3}))"},
    {:quiet, "Nx.multiply(Nx.iota({2, 3}, names: [:rows, :cols]), Nx.iota({3}, names: [:cols]))"},
    {{:misaligned, "unnamed_axis"},
     "Nx.dot(Nx.iota({2, 3}, names: [:rows, :cols]), [1], Nx.iota({3, 4}), [0])"},
    {:quiet,
     "Nx.dot(Nx.iota({2, 3}, names: [:rows, :cols]), [:cols], Nx.iota({3, 4}, names: [:cols, :out]), [:cols])"},
    {{:misaligned, "contracted_names"},
     "Nx.dot(Nx.iota({2, 3}, names: [:rows, :cols]), [:cols], Nx.iota({3, 4}, names: [:inner, :out]), [:inner])"},
    {:quiet, "Nx.dot(Nx.iota({2, 3}), [1], Nx.iota({3, 4}), [0])"},
    # vectorizing an axis under another name than its own
    {{:misaligned, "vectorize_name"},
     "Nx.vectorize(Nx.iota({2, 3}, names: [:rows, :cols]), :heads)"},
    {:quiet, "Nx.vectorize(Nx.iota({2, 3}, names: [:heads, :cols]), :heads)"},
    {:quiet, "Nx.vectorize(Nx.iota({2, 3}), :heads)"},
    # two dispatches on one value run one implementation; on two, any two
    {:quiet, "Shaper.take(shaper, Shaper.make(shaper, config), config)"},
    {{:mismatch, "names"}, "Shaper.take(other, Shaper.make(shaper, config), config)"},
    # divisors the math keeps from being negative but not from being zero
    {{:nonfinite, "divide_by_zero", "square"}, "Nx.divide(t, Nx.sum(Nx.multiply(t, t)))"},
    {:finite, "Nx.divide(t, Nx.add(Nx.sum(Nx.multiply(t, t)), 1.0e-6))"},
    {{:nonfinite, "divide_by_zero", "square"}, "Nx.divide(t, Nx.sum(Nx.pow(t, 2)))"},
    {{:nonfinite, "divide_by_zero", "square"}, "Nx.divide(t, length_of(t))"},
    {{:nonfinite, "divide_by_zero", "square"}, "Nx.rsqrt(Nx.mean(Nx.multiply(t, t)))"},
    {:finite, "Nx.rsqrt(Nx.add(Nx.mean(Nx.multiply(t, t)), 1.0e-5))"},
    {{:nonfinite, "divide_by_zero", "comparison"}, "Nx.divide(t, Nx.sum(Nx.greater(t, 0)))"},
    {{:nonfinite, "divide_by_zero", "index"}, "Nx.divide(1, Nx.iota({4}))"},
    {{:nonfinite, "divide_by_zero", "index"}, "Nx.quotient(t, Nx.iota({1}))"},
    {{:nonfinite, "divide_by_zero", "spread"}, "Nx.divide(t, Nx.standard_deviation(t))"},
    {{:nonfinite, "divide_by_zero", "clamp"}, "Nx.divide(t, Nx.max(t, 0))"},
    {:finite, "Nx.divide(t, Nx.max(Nx.abs(t), 1.0e-6))"},
    {{:nonfinite, "divide_by_zero", "absolute"}, "Nx.divide(t, Nx.abs(t))"},
    {{:nonfinite, "divide_by_zero", "norm"}, "Nx.divide(t, Nx.LinAlg.norm(t))"},
    {{:nonfinite, "divide_by_zero", "zero"}, "Nx.divide(t, 0)"},
    {{:nonfinite, "divide_by_zero", "absolute"}, "Nx.pow(Nx.abs(t), -0.5)"},
    {:finite, "Nx.pow(Nx.add(Nx.abs(t), 1), -0.5)"},
    # logarithms of values the math keeps from being negative but not zero
    {{:nonfinite, "log_of_zero", "square"}, "Nx.log(Nx.sum(Nx.multiply(t, t)))"},
    {{:nonfinite, "log_of_zero", "comparison"}, "Nx.log(Nx.sum(Nx.greater(t, 0)))"},
    {:finite, "Nx.log(Nx.add(Nx.sum(Nx.multiply(t, t)), 1.0e-6))"},
    # roots and logarithms of differences of values that cannot be negative
    {{:hazard, "root_of_negative", "cancellation"},
     "Nx.sqrt(Nx.subtract(Nx.mean(Nx.multiply(t, t)), Nx.pow(Nx.mean(t), 2)))"},
    {:finite,
     "Nx.sqrt(Nx.max(Nx.subtract(Nx.mean(Nx.multiply(t, t)), Nx.pow(Nx.mean(t), 2)), 0))"},
    {{:hazard, "log_of_negative", "cancellation"}, "Nx.log(Nx.subtract(1, Nx.multiply(t, t)))"},
    {{:hazard, "root_of_negative", "cancellation"},
     "Nx.pow(Nx.subtract(Nx.mean(Nx.multiply(t, t)), Nx.pow(Nx.mean(t), 2)), 0.5)"},
    # a logarithm of a softmax written out, shifted or not
    {{:hazard, "log_of_zero", "underflow"}, "Nx.log(Nx.divide(Nx.exp(t), Nx.sum(Nx.exp(t))))"},
    {{:hazard, "log_of_zero", "underflow"},
     "shifted = Nx.subtract(t, Nx.reduce_max(t))\nNx.log(Nx.divide(Nx.exp(shifted), Nx.sum(Nx.exp(shifted))))"},
    # a softmax or log-sum-exp over values not shifted by their maximum
    {{:hazard, "exp_overflow", "unshifted"}, "Nx.divide(Nx.exp(t), Nx.sum(Nx.exp(t)))"},
    {{:hazard, "exp_overflow", "unshifted"}, "Nx.log(Nx.sum(Nx.exp(t)))"},
    {:finite,
     "shifted = Nx.subtract(t, Nx.reduce_max(t))\nNx.divide(Nx.exp(shifted), Nx.sum(Nx.exp(shifted)))"},
    {:finite,
     "largest = Nx.reshape(Nx.reduce_max(t), {1})\nshifted = Nx.subtract(t, Nx.max(largest, 0))\nNx.divide(Nx.exp(shifted), Nx.sum(Nx.exp(shifted)))"},
    # operands from inputs that nothing checks
    {{:unchecked, "unchecked_divisor", "cancel"}, "Nx.divide(t, Nx.add(t, 1))"},
    {{:unchecked, "unchecked_divisor", "input"}, "Nx.divide(t, config.heads)"},
    {{:unchecked, "unchecked_divisor", "input"}, "Nx.divide(t, t)"},
    {{:unchecked, "unchecked_logarithm", "input"}, "Nx.log(t)"},
    {{:unchecked, "unchecked_root", "negative"}, "Nx.sqrt(t)"},
    {{:unchecked, "unchecked_root", "negative"}, "Nx.sqrt(Nx.subtract(t, 1))"},
    # and the same checked, by a test on the path or by a select
    {:finite, "n = config.heads\nif n == 0, do: t, else: Nx.divide(t, n)"},
    {:finite, "n = config.heads\nif n != 0, do: Nx.divide(t, n), else: t"},
    {:finite, "n = config.heads\nif n > 0, do: Nx.divide(t, n), else: t"},
    {:finite, "n = config.heads\nif n < 1, do: t, else: Nx.log(n)"},
    {:finite, "Nx.divide(t, Nx.select(Nx.equal(t, 0), 1, t))"},
    {:finite, "Nx.divide(t, Nx.select(Nx.not_equal(t, 0), t, 1))"},
    # the same, as a `defn` compiles `==` and `>`
    {:finite, "Nx.divide(t, Nx.select(Nx.Defn.Kernel.__equal__(t, 0), 1, t))"},
    {{:nonfinite, "divide_by_zero", "comparison"},
     "Nx.divide(t, Nx.sum(Nx.Defn.Kernel.__more_than__(t, 0)))"},
    {:finite, "Nx.log(Nx.select(Nx.greater(t, 0), t, 1))"},
    {:finite, "Nx.sqrt(Nx.abs(t))"},
    # a test that does not keep the operand from zero checks nothing
    {{:unchecked, "unchecked_divisor", "input"},
     "n = config.heads\nif n >= 0, do: Nx.divide(t, n), else: t"},
    {{:unchecked, "unchecked_divisor", "input"},
     "Nx.divide(t, Nx.select(Nx.greater(t, 0), 1, t))"},
    # nor does one of a value the operand is only made from
    {{:unchecked, "unchecked_logarithm", "input"},
     "n = config.heads\nif n < 1, do: t, else: Nx.log(Nx.multiply(t, n))"},
    # functions defined on part of the line, of values their math takes to
    # the edge or past it
    {{:hazard, "infinite_at_edge", "saturation"}, "Nx.atanh(Nx.tanh(t))"},
    {{:hazard, "infinite_at_edge", "clip"}, "Nx.atanh(Nx.clip(t, -1, 1))"},
    {:finite, "Nx.atanh(Nx.clip(t, -0.999, 0.999))"},
    {{:hazard, "infinite_at_edge", "saturation"}, "Nx.erf_inv(Nx.erf(t))"},
    {{:hazard, "infinite_at_edge", "saturation"}, "Nx.log1p(Nx.negate(Nx.sigmoid(t)))"},
    {{:hazard, "infinite_at_edge", "saturation"}, "Nx.log1p(Nx.tanh(t))"},
    {:finite, "Nx.asin(Nx.clip(t, -1, 1))"},
    {:finite, "Nx.acos(Nx.tanh(t))"},
    {{:hazard, "outside_domain", "written"}, "Nx.asin(Nx.multiply(2, Nx.tanh(t)))"},
    {{:nonfinite, "outside_domain", "saturation"}, "Nx.acosh(Nx.tanh(t))"},
    {:finite, "Nx.acosh(Nx.cosh(t))"},
    {:finite, "Nx.log1p(Nx.multiply(t, t))"},
    {:finite, "Nx.log1p(Nx.exp(t))"},
    # an arc sine or cosine of a ratio within ±1 only before rounding
    {{:hazard, "outside_domain", "rounding"},
     "Nx.acos(Nx.divide(Nx.dot(t, t), Nx.max(Nx.multiply(Nx.LinAlg.norm(t), Nx.LinAlg.norm(t)), 1.0e-6)))"},
    {{:hazard, "outside_domain", "rounding"},
     "unit = Nx.divide(t, Nx.add(Nx.sqrt(Nx.sum(Nx.multiply(t, t))), 1.0e-6))\nNx.asin(Nx.dot(unit, unit))"},
    {:finite,
     "Nx.acos(Nx.clip(Nx.divide(Nx.dot(t, t), Nx.max(Nx.multiply(Nx.LinAlg.norm(t), Nx.LinAlg.norm(t)), 1.0e-6)), -1, 1))"},
    # and of values nothing keeps in the domain, unless a test does
    {{:unchecked, "unchecked_domain", "input"}, "Nx.atanh(t)"},
    {{:unchecked, "unchecked_domain", "input"}, "Nx.asin(t)"},
    {{:unchecked, "unchecked_domain", "input"}, "Nx.log1p(t)"},
    {{:unchecked, "unchecked_domain", "input"}, "Nx.acosh(t)"},
    {{:unchecked, "unchecked_domain", "unbounded"}, "Nx.asin(Nx.divide(t, 2))"},
    {:finite, "n = config.heads\nif n > -1, do: Nx.log1p(n), else: t"},
    {:finite, "n = config.heads\nif n >= 1, do: Nx.acosh(n), else: t"},
    {:finite, "n = config.heads\nif n < 1, do: if(n > -1, do: Nx.atanh(n), else: t), else: t"},
    # roots and norms that can be zero where a grad differentiates them
    {{:nonfinite, "infinite_gradient", "square"},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.sqrt(Nx.multiply(x, x))) end)"},
    {{:nonfinite, "infinite_gradient", "norm"}, "Nx.Defn.grad(t, fn x -> Nx.LinAlg.norm(x) end)"},
    {{:nonfinite, "infinite_gradient", "norm"},
     "{_value, gradient} = Nx.Defn.value_and_grad(t, fn x -> Nx.LinAlg.norm(x) end)\ngradient"},
    {:finite, "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.sqrt(Nx.add(Nx.multiply(x, x), 1.0e-12))) end)"},
    {:finite, "Nx.sum(Nx.sqrt(Nx.multiply(t, t)))"},
    {:finite,
     "Nx.Defn.grad(t, fn x -> Nx.add(Nx.sum(x), Nx.sqrt(Nx.Defn.Kernel.stop_grad(Nx.sum(Nx.multiply(x, x))))) end)"},
    # calls that take only integers handed floats
    {{:type_error, "non_integer_operand", "float"}, "Nx.bitwise_and(Nx.divide(t, 2), 1)"},
    {{:type_error, "non_integer_operand", "float"}, "Nx.take(t, Nx.divide(t, 2))"},
    {{:type_error, "non_integer_operand", "float"}, "Nx.take(t, Nx.floor(Nx.divide(t, 2)))"},
    {:quiet, "Nx.take(t, Nx.as_type(Nx.floor(Nx.divide(t, 2)), :s32))"},
    {{:type_error, "non_integer_operand", "float"}, "Nx.quotient(Nx.multiply(t, 1.5), 2)"},
    {:quiet, "Nx.quotient(t, 2)"},
    {{:type_error, "non_integer_operand", "float"}, "Nx.right_shift(Nx.sum(Nx.sigmoid(t)), 1)"},
    {:quiet, "Nx.bitwise_and(Nx.iota({2}), 1)"},
    {{:type_error, "non_integer_operand", "float"},
     "Nx.bitwise_and(Nx.iota({2}, type: :f32), 1)"},
    {{:type_error, "non_integer_operand", "float"},
     "Nx.gather(t, Nx.new_axis(Nx.multiply(t, 0.5), -1))"},
    # tensors made in a type the backend lacks
    {{:unsupported, "f64"}, "Nx.as_type(t, :f64)"},
    {{:unsupported, "f64"}, "Nx.iota({2}, type: {:f, 64})"},
    {{:unsupported, "f64"}, "Nx.Constants.pi({:f, 64})"},
    {:quiet, "Nx.iota({2}, type: :f32)"}
  ]

  # ── Options: priv/tensor_shapes/options.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_options []
  @fixture_modules_options ""

  # ── Traced: priv/tensor_shapes/traced.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_traced []
  @fixture_modules_traced ""

  # ── DefnFlow: priv/tensor_shapes/defn_flow.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_defn_flow []
  @fixture_modules_defn_flow ""

  # ── Containers: priv/tensor_shapes/containers.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_containers []
  @fixture_modules_containers ""

  # ── Gradients: priv/tensor_shapes/gradients.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_gradients []
  @fixture_modules_gradients ""

  # ── Math: priv/tensor_shapes/math.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_math []
  @fixture_modules_math ""

  # ── Indices: priv/tensor_shapes/indices.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_indices []
  @fixture_modules_indices ""

  # ── Literals: priv/tensor_shapes/literals.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_literals []
  @fixture_modules_literals ""

  # ── Access: priv/tensor_shapes/access.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_access []
  @fixture_modules_access ""

  # ── Tuples: priv/tensor_shapes/tuples.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_tuples []
  @fixture_modules_tuples ""

  # ── ShapeGaps: priv/tensor_shapes/shape_gaps.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_shape_gaps []
  @fixture_modules_shape_gaps ""

  # ── ReshapeOrder: priv/tensor_shapes/reshape_order.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_reshape_order []
  @fixture_modules_reshape_order ""

  # ── Dtypes: priv/tensor_shapes/dtypes.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_dtypes []
  @fixture_modules_dtypes ""

  # ── Serving: priv/tensor_shapes/serving.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_serving []
  @fixture_modules_serving ""

  # ── Consumption: priv/tensor_shapes/consumption.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_consumption []
  @fixture_modules_consumption ""

  # ── LinAlg: priv/tensor_shapes/linalg.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_linalg []
  @fixture_modules_linalg ""

  @lint_cases @lint_cases_base ++
                @lint_cases_options ++
                @lint_cases_traced ++
                @lint_cases_defn_flow ++
                @lint_cases_containers ++
                @lint_cases_gradients ++
                @lint_cases_math ++
                @lint_cases_indices ++
                @lint_cases_literals ++
                @lint_cases_access ++
                @lint_cases_tuples ++
                @lint_cases_shape_gaps ++
                @lint_cases_reshape_order ++
                @lint_cases_dtypes ++
                @lint_cases_serving ++
                @lint_cases_consumption ++
                @lint_cases_linalg

  @fixture_modules [
    @fixture_modules_options,
    @fixture_modules_traced,
    @fixture_modules_defn_flow,
    @fixture_modules_containers,
    @fixture_modules_gradients,
    @fixture_modules_math,
    @fixture_modules_indices,
    @fixture_modules_literals,
    @fixture_modules_access,
    @fixture_modules_tuples,
    @fixture_modules_shape_gaps,
    @fixture_modules_reshape_order,
    @fixture_modules_dtypes,
    @fixture_modules_serving,
    @fixture_modules_consumption,
    @fixture_modules_linalg
  ]

  @shaper """
  defprotocol ArgusNxTensorAnalyses.TensorShapesTest.Shaper do
    def make(shaper, config)
    def take(shaper, tensor, config)
  end

  defmodule ArgusNxTensorAnalyses.TensorShapesTest.Wide do
    defstruct []

    defimpl ArgusNxTensorAnalyses.TensorShapesTest.Shaper do
      def make(_shaper, config), do: Nx.iota({config.a}, names: [:wide])
      def take(_shaper, tensor, config), do: Nx.add(tensor, Nx.iota({config.a}, names: [:wide]))
    end
  end

  defmodule ArgusNxTensorAnalyses.TensorShapesTest.Tall do
    defstruct []

    defimpl ArgusNxTensorAnalyses.TensorShapesTest.Shaper do
      def make(_shaper, config), do: Nx.iota({config.b}, names: [:tall])
      def take(_shaper, tensor, config), do: Nx.add(tensor, Nx.iota({config.b}, names: [:tall]))
    end
  end
  """

  setup_all do
    directory =
      Path.join(System.tmp_dir!(), "tensor_shapes_test_#{System.unique_integer([:positive])}")

    File.mkdir_p!(directory)
    on_exit(fn -> File.rm_rf!(directory) end)

    source = Path.join(directory, "fixtures.ex")
    File.write!(source, fixture_source())

    capture_io(:stderr, fn ->
      {:ok, _modules, _diagnostics} =
        Kernel.ParallelCompiler.compile_to_path([source], directory, return_diagnostics: true)
    end)

    beam = Path.join(directory, "Elixir.#{inspect(@fixtures)}.beam")
    probe = Path.join(directory, "probe.dl")
    File.write!(probe, probe_program())

    {:ok, rows} =
      TensorShapes.solve(Path.wildcard(Path.join(directory, "*.beam")), probe,
        unsupported_types: [:f64]
      )

    %{beam: beam, source: source, rows: rows}
  end

  for {spec, index} <- Enum.with_index(@cases, 1) do
    {expectation, body} =
      case spec do
        {expectation, body} -> {expectation, body}
        body -> {:agrees, body}
      end

    test "#{index}: #{String.replace(body, "\n", "; ")}", %{rows: rows} do
      index = unquote(index)

      assert_agrees(
        unquote(expectation),
        run_case(index),
        derived_shapes(rows, index),
        findings(rows, index)
      )
    end
  end

  for {{expectation, body}, index} <- Enum.with_index(@lint_cases, 1) do
    test "lint #{index}: #{String.replace(body, "\n", "; ")}", %{rows: rows} do
      index = unquote(index)
      found = lint_findings(rows, index)

      case unquote(Macro.escape(expectation)) do
        :quiet ->
          assert found == [],
                 "the code lines its axes up, and the analysis finds #{inspect(found)}"

          assert run_lint(index) == :accepted

        {:misaligned, kind} ->
          assert [_ | _] = for({"tensor_axis_misalignment", ^kind, _detail} <- found, do: kind),
                 "expected a #{kind} misalignment, found #{inspect(found)}"

          refute Enum.any?(found, &match?({"tensor_shape_mismatch", _, _}, &1)), inspect(found)
          assert run_lint(index) == :accepted

        {:mismatch, kind} ->
          assert [_ | _] = for({"tensor_shape_mismatch", ^kind, _detail} <- found, do: kind),
                 "expected a #{kind} mismatch, found #{inspect(found)}"

        {:nonfinite, kind, cause} ->
          assert {"tensor_nonfinite_result", kind, cause} in found,
                 "expected a #{kind} from a #{cause}, found #{inspect(found)}"

          assert lint_outcome(index) == :nonfinite

        {:hazard, kind, cause} ->
          assert {"tensor_nonfinite_result", kind, cause} in found,
                 "expected a #{kind} from #{cause}, found #{inspect(found)}"

          assert run_lint(index) == :accepted

        {:unchecked, kind, cause} ->
          assert {"tensor_nonfinite_result", kind, cause} in found,
                 "expected an #{kind} from #{cause}, found #{inspect(found)}"

          refute Enum.any?(found, fn {_relation, found_kind, _cause} ->
                   not String.starts_with?(found_kind, "unchecked_")
                 end),
                 "expected no definite finding, found #{inspect(found)}"

          assert run_lint(index) == :accepted

        {:type_error, kind, subject} ->
          assert {"tensor_type_error", kind, subject} in found,
                 "expected a #{kind} of a #{subject}, found #{inspect(found)}"

          assert_raise ArgumentError, fn -> run_lint(index) end

        {:unsupported, type} ->
          assert {"tensor_type_error", "unsupported_type", type} in found,
                 "expected an unsupported #{type}, found #{inspect(found)}"

          assert run_lint(index) == :accepted

        :finite ->
          assert found == [],
                 "the math keeps the result finite, and the analysis finds #{inspect(found)}"

          assert lint_outcome(index) == :finite

        {:finds, {relation, kind, subject}, outcome} ->
          assert Enum.any?(found, fn {found_relation, found_kind, found_subject} ->
                   found_relation == relation and found_kind == kind and
                     (subject == :any or found_subject == subject)
                 end),
                 "expected a #{relation} #{kind} (#{inspect(subject)}), found #{inspect(found)}"

          assert_outcome(outcome, index)

        {:finds_none, outcome} ->
          assert found == [], "expected no finding, found #{inspect(found)}"
          assert_outcome(outcome, index)
      end
    end
  end

  # `divided/2` divides by an input in a context of its own, and by an
  # absolute value in the context `divides_by_absolute/1` calls it in,
  # which hands it a size variable.
  test "an unchecked operand gives way to a definite finding at its call", %{rows: rows} do
    function = "#{inspect(@lint_fixtures)}:divided/2"

    kinds =
      for [_id, ^function, _operation, kind | _rest] <-
            Map.get(rows, "tensor_nonfinite_result", []),
          uniq: true,
          do: kind

    assert kinds == ["divide_by_zero"]
  end

  # `scaled_by/2` divides by what its callers compute, an absolute value,
  # and `shrunk_by/2` by one they keep positive.
  test "a helper's parameter is what the program's calls hand it", %{rows: rows} do
    kinds = fn name ->
      function = "#{inspect(@lint_fixtures)}:#{name}"

      for [_id, ^function, _operation, kind, cause | _rest] <-
            Map.get(rows, "tensor_nonfinite_result", []),
          uniq: true,
          do: {kind, cause}
    end

    assert kinds.("scaled_by/2") == [{"divide_by_zero", "absolute"}]
    assert kinds.("shrunk_by/2") == []
  end

  # `magnitude/1` is differentiated where `differentiates_magnitude/1`
  # hands a capture of it to a grad.
  test "a function a grad is handed is differentiated", %{rows: rows} do
    function = "#{inspect(@lint_fixtures)}:magnitude/1"

    kinds =
      for [_id, ^function, _operation, kind | _rest] <-
            Map.get(rows, "tensor_nonfinite_result", []),
          uniq: true,
          do: kind

    assert kinds == ["infinite_gradient"]
  end

  # ── Options: tests of their own ──
  # (end of Options tests)

  # ── Traced: tests of their own ──
  # (end of Traced tests)

  # ── DefnFlow: tests of their own ──
  # (end of DefnFlow tests)

  # ── Containers: tests of their own ──
  # (end of Containers tests)

  # ── Gradients: tests of their own ──
  # (end of Gradients tests)

  # ── Math: tests of their own ──
  # (end of Math tests)

  # ── Indices: tests of their own ──
  # (end of Indices tests)

  # ── Literals: tests of their own ──
  # (end of Literals tests)

  # ── Access: tests of their own ──
  # (end of Access tests)

  # ── Tuples: tests of their own ──
  # (end of Tuples tests)

  # ── ShapeGaps: tests of their own ──
  # (end of ShapeGaps tests)

  # ── ReshapeOrder: tests of their own ──
  # (end of ReshapeOrder tests)

  # ── Dtypes: tests of their own ──
  # (end of Dtypes tests)

  # ── Serving: tests of their own ──
  # (end of Serving tests)

  # ── Consumption: tests of their own ──
  # (end of Consumption tests)

  # ── LinAlg: tests of their own ──
  # (end of LinAlg tests)

  test "run/2 places a finding at its call", %{beam: beam, source: source} do
    {:ok, placed} = TensorShapes.run([beam])

    reshape =
      Enum.find(
        placed,
        &(&1.finding.detail =~ "shape {2, 3} is not compatible with new shape {4, 2}")
      )

    assert reshape.file == source

    assert source |> File.read!() |> String.split("\n") |> Enum.at(reshape.line - 1) ==
             "Nx.reshape(Nx.iota({2, 3}), {4, 2})"
  end

  test "run/2 places the calls that bring a helper its shapes", %{beam: beam, source: source} do
    {:ok, placed} = TensorShapes.run([beam])
    lines = source |> File.read!() |> String.split("\n")

    callers =
      for %{finding: finding, related: frames} <- placed,
          finding.detail =~ "cannot broadcast tensor of dimensions {2, 3} to {4}",
          frame <- frames,
          do: Enum.at(lines, frame.line - 1)

    assert "local_helper(Nx.iota({2, 3}))" in callers
  end

  test "run/2 labels a call with the shapes it gets and the call that makes each", %{
    beam: beam
  } do
    {:ok, placed} = TensorShapes.run([beam])

    reshape =
      Enum.find(
        placed,
        &(&1.finding.detail =~ "shape {2, 3} is not compatible with new shape {4, 2}")
      )

    assert reshape.finding.title == "Nx.reshape/2 gets a shape it cannot reshape to"
    assert reshape.finding.at_label == "gets {2, 3}"

    assert [%{label: "makes {2, 3}, the first argument of Nx.reshape/2"}] =
             reshape.finding.related

    assert [%{line: line}] = reshape.related
    assert line == reshape.line
  end

  defp assert_agrees(:agrees, {:raises, message}, _derived, found) do
    assert found != [], "Nx raises #{inspect(message)}, and the analysis finds nothing"
  end

  defp assert_agrees(:agrees, {:returns, shape}, derived, found) do
    assert found == [], "Nx accepts the shapes, and the analysis finds #{inspect(found)}"

    if shape do
      assert derived == [shape], "Nx gives #{shape}, and the analysis derives #{inspect(derived)}"
    end
  end

  defp assert_agrees(:some_path, {:returns, _shape}, _derived, found) do
    assert found != []
    assert Enum.all?(found, &(&1.certainty == "on_some_path")), inspect(found)
  end

  defp assert_agrees(:unknown, {:returns, _shape}, derived, found) do
    assert {derived, found} == {[], []}
  end

  defp assert_agrees(:no_finding, {:returns, _shape}, _derived, found) do
    assert found == []
  end

  defp run_case(index) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      case apply(@fixtures, :"case_#{index}", []) do
        %Nx.Tensor{} = tensor -> {:returns, spell(tensor)}
        _other -> {:returns, nil}
      end
    end)
  rescue
    error -> {:raises, Exception.message(error)}
  end

  # A tensor's shape as the probe program spells it: `{2, 3}[:a, nil]`,
  # then `|:x=2` for each vectorized axis.
  defp spell(%Nx.Tensor{shape: shape, names: names, vectorized_axes: vectorized}) do
    sizes = shape |> Tuple.to_list() |> Enum.map_join(", ", &Integer.to_string/1)
    vectorized = Enum.map_join(vectorized, "", fn {name, size} -> "|#{inspect(name)}=#{size}" end)

    "{#{sizes}}[#{Enum.map_join(names, ", ", &inspect/1)}]" <> vectorized
  end

  defp derived_shapes(rows, index) do
    function = function_id(index)
    for [^function, shape] <- Map.get(rows, "returned_shape", []), do: shape
  end

  # The findings in the case's function, and those in the functions it
  # calls, reached with the shapes it gives them.
  defp findings(rows, index) do
    function = function_id(index)

    reached =
      for [id, kind, _order, _call, ^function, "call" | _frame] <-
            Map.get(rows, "tensor_shape_mismatch_via", []),
          do: {id, kind}

    for [id, func, operation, kind, detail, certainty, _operands] <-
          Map.get(rows, "tensor_shape_mismatch", []),
        func == function or {id, kind} in reached,
        uniq: true,
        do: %{operation: operation, kind: kind, detail: detail, certainty: certainty}
  end

  defp function_id(index), do: "#{inspect(@fixtures)}:case_#{index}/0"

  # The findings of the shape relations in the lint case's function and in
  # the functions it reaches, as `{relation, kind, detail}`, and of results
  # that can be infinite or NaN in it, as `{relation, kind, cause}`.
  defp lint_findings(rows, index) do
    function = "#{inspect(@lint_fixtures)}:lint_#{index}/4"

    # its closures' too, which a grad differentiates
    closure = "#{inspect(@lint_fixtures)}:-lint_#{index}/4-fun-"

    nonfinite =
      for [_id, found_in, _operation, kind, cause | _origin] <-
            Map.get(rows, "tensor_nonfinite_result", []),
          found_in == function or String.starts_with?(found_in, closure),
          uniq: true,
          do: {"tensor_nonfinite_result", kind, cause}

    types =
      for [_id, ^function, _operation, kind, subject | _rest] <-
            Map.get(rows, "tensor_type_error", []),
          uniq: true,
          do: {"tensor_type_error", kind, subject}

    calls =
      for [_id, found_in, _operation, kind, detail | _rest] <-
            Map.get(rows, "tensor_call_error", []),
          found_in == function or String.starts_with?(found_in, closure),
          uniq: true,
          do: {"tensor_call_error", kind, detail}

    shape_findings(rows, function) ++ nonfinite ++ types ++ calls
  end

  defp shape_findings(rows, function) do
    for relation <- ["tensor_shape_mismatch", "tensor_axis_misalignment"],
        reached =
          for(
            [id, kind, _order, _call, ^function, "call" | _frame] <-
              Map.get(rows, relation <> "_via", []),
            do: {id, kind}
          ),
        [id, func, _operation, kind, detail, _certainty, _operands] <- Map.get(rows, relation, []),
        func == function or {id, kind} in reached,
        uniq: true,
        do: {relation, kind, detail}
  end

  # Nx accepts the lint case when every size variable is 1, the shapers
  # one implementation's.
  defp run_lint(index) do
    config = Map.new(~w(heads kv_heads dim head_dim hidden rows cols a b)a, &{&1, 1})

    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      shaper = struct(ArgusNxTensorAnalyses.TensorShapesTest.Wide)
      apply(@lint_fixtures, :"lint_#{index}", [config, Nx.iota({1}), shaper, shaper])
      :accepted
    end)
  end

  # What the lint case gives where `t` is zero and every size variable is
  # 1: `:nonfinite` where its result holds an infinity or a NaN, or it
  # raises dividing integers by zero, and `:finite` otherwise.
  defp lint_outcome(index) do
    config = Map.new(~w(heads kv_heads dim head_dim hidden rows cols a b)a, &{&1, 1})

    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      shaper = struct(ArgusNxTensorAnalyses.TensorShapesTest.Wide)
      result = apply(@lint_fixtures, :"lint_#{index}", [config, Nx.iota({1}), shaper, shaper])
      nonfinite = Nx.logical_or(Nx.is_nan(result), Nx.is_infinity(result))

      if nonfinite |> Nx.any() |> Nx.to_number() == 1, do: :nonfinite, else: :finite
    end)
  rescue
    ArithmeticError -> :nonfinite
  end

  # What running lint case `index` does, as `{:finds, ...}` names it:
  # `:raises`, `:nonfinite` where its result holds an infinity or a NaN,
  # `:finite` otherwise; `:accepted` is anything but `:raises`, `:any`
  # anything at all.
  defp assert_outcome(:any, _index), do: :ok

  defp assert_outcome(:accepted, index),
    do: refute(lint_result(index) == :raises, "expected Nx to accept lint #{index}")

  defp assert_outcome(outcome, index), do: assert(lint_result(index) == outcome)

  defp lint_result(index) do
    config = Map.new(~w(heads kv_heads dim head_dim hidden rows cols a b)a, &{&1, 1})

    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      shaper = struct(ArgusNxTensorAnalyses.TensorShapesTest.Wide)
      result = apply(@lint_fixtures, :"lint_#{index}", [config, Nx.iota({1}), shaper, shaper])
      if nonfinite?(result), do: :nonfinite, else: :finite
    end)
  rescue
    _error -> :raises
  end

  defp nonfinite?(%Nx.Tensor{} = tensor) do
    tensor
    |> Nx.is_nan()
    |> Nx.logical_or(Nx.is_infinity(tensor))
    |> Nx.any()
    |> Nx.to_number() == 1
  end

  defp nonfinite?(tuple) when is_tuple(tuple), do: tuple |> Tuple.to_list() |> nonfinite?()
  defp nonfinite?(list) when is_list(list), do: Enum.any?(list, &nonfinite?/1)

  defp nonfinite?(map) when is_map(map) and not is_struct(map),
    do: map |> Map.values() |> nonfinite?()

  defp nonfinite?(_other), do: false

  defp fixture_source do
    cases =
      @cases
      |> Enum.with_index(1)
      |> Enum.map_join("\n", fn {spec, index} ->
        body =
          case spec do
            {_expectation, body} -> body
            body -> body
          end

        "  def case_#{index} do\n#{body}\n  end\n"
      end)

    lints =
      @lint_cases
      |> Enum.with_index(1)
      |> Enum.map_join("\n", fn {{_expectation, body}, index} ->
        "  def lint_#{index}(config, t, shaper, other) do\n#{body}\n  end\n"
      end)

    """
    defmodule #{inspect(@fixtures)} do
      import Nx.Defn

    #{@helpers}
    #{cases}
    end

    defmodule #{inspect(@lint_fixtures)} do
    #{@lint_helpers}
    #{lints}
    end

    #{@shaper}

    #{Enum.join(@fixture_modules, "\n")}
    """
  end

  # The program with the shape each function returns, as `spell/1`
  # writes it.
  defp probe_program do
    """
    .include "#{TensorShapes.rules_file()}"

    .decl returned(function: symbol, tensor: Tensor)
    returned(function, tensor) :- return_value($Canonical(function, nil), function, $Tensor(tensor)).

    spelling_demand(axes) :- returned(_, [_, axes]).

    .decl vectorized_demand(axes: VectorizedAxes)
    vectorized_demand(vectorized) :- returned(_, [vectorized, _]).
    vectorized_demand(rest) :- vectorized_demand([_, _, rest]).

    .decl vectorized_spelled(axes: VectorizedAxes, spelling: symbol)
    vectorized_spelled(nil, "") :- vectorized_demand(nil).
    vectorized_spelled([name, [size, nil], rest], cat("|", name, "=", to_string(size), tail)) :-
      vectorized_demand([name, [size, nil], rest]),
      size > 0,
      vectorized_spelled(rest, tail).

    .decl returned_shape(function: symbol, shape: symbol)
    returned_shape(function, cat(sizes, names, vectorized_text)) :-
      returned(function, [vectorized, axes]),
      sizes_spelled(axes, sizes),
      names_spelled(axes, names),
      vectorized_spelled(vectorized, vectorized_text).
    .output returned_shape
    """
  end
end
