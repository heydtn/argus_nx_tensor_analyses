defmodule ArgusNxTensorAnalyses.TensorShapesTest do
  # Nx is the oracle: each case below is compiled into a module, run for
  # the shape Nx gives it or the error Nx raises, and the analysis of the
  # compiled module has to agree.
  use ArgusNxTensorAnalyses.TensorAnalysisCase

  import ExUnit.CaptureIO

  alias ArgusNxTensorAnalyses.TensorShapes

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
    "{left, _right} = Nx.split(Nx.iota({5, 2}), 2)\nleft",
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
    "Nx.tensor([[9, 9], [10, 13]])",
    "Nx.tensor([92, 116])",
    "Nx.tensor([[34, 35, 91], [1, 2, 3]])",
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
    defp call_with(fun, argument), do: fun.(argument)
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
    {{:misaligned, "reshape_order"},
     "Nx.reshape(Nx.iota({config.heads, config.dim}), {config.dim, config.heads})"},
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
    # a closure a helper calls takes the helper's argument before what it captured
    {:quiet,
     "weight = Nx.iota({3, 4})\ncall_with(fn x -> Nx.dot(x, weight) end, Nx.iota({config.heads, 3}))"},
    {{:mismatch, "dot"},
     "weight = Nx.iota({3, 4})\ncall_with(fn x -> Nx.dot(x, weight) end, Nx.iota({config.heads, 5}))"},
    # an iota's zero is one element's: its sum is not zero, its running sum is
    {:finite, "Nx.divide(t, Nx.sum(Nx.iota({4})))"},
    {{:nonfinite, "divide_by_zero", "index"}, "Nx.divide(t, Nx.cumulative_sum(Nx.iota({4})))"},
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
    # within log1p's domain, but a softplus written out, whose exponential overflows
    {{:hazard, "exp_overflow", "softplus"}, "Nx.log1p(Nx.exp(t))"},
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
    {:quiet, "Nx.iota({2}, type: :f32)"},
    # a jit's options are its compiler's, and name no tensor's type
    {:quiet, "Nx.add(Nx.Defn.jit(fn x -> x end, type: :f64).(t), 1)"}
  ]

  # ── Options: priv/tensor_shapes/options.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_options [
    # option keys the function does not take
    {{:finds, {"tensor_call_error", "unknown_option", "axis"}, :raises}, "Nx.sum(t, axis: 0)"},
    {:quiet, "Nx.sum(t, axes: [0])"},
    {{:finds, {"tensor_call_error", "unknown_option", "axes"}, :raises},
     "Nx.argmax(t, axes: [0])"},
    {:quiet, "Nx.argmax(t, axis: 0)"},
    {{:finds, {"tensor_call_error", "unknown_option", "stride"}, :raises},
     "Nx.window_sum(t, {1}, stride: [1])"},
    {:quiet, "Nx.window_sum(t, {1}, strides: [1])"},
    {{:finds, {"tensor_call_error", "unknown_option", "num"}, :raises},
     "Nx.linspace(0, 1, num: 3)"},
    {:quiet, "Nx.linspace(0, 1, n: 3)"},
    {{:finds, {"tensor_call_error", "unknown_option", "full_matrices"}, :raises},
     "Nx.LinAlg.svd(Nx.iota({2, 2}, type: :f32), full_matrices: false)"},
    {:quiet, "Nx.LinAlg.svd(Nx.iota({2, 2}, type: :f32), full_matrices?: false)"},
    {{:finds, {"tensor_call_error", "unknown_option", "dim"}, :raises},
     "Nx.sum(t, dim: config.a - 1, keep_axes: true)"},
    {{:finds, {"tensor_call_error", "unknown_option", "type"}, :raises},
     "Nx.Constants.pi(:f32, type: :f32)"},
    {:quiet, "Nx.Constants.pi(:f32, backend: Nx.BinaryBackend)"},
    # a call whose options Nx rejects makes no tensor of the type they name
    {{:finds, {"tensor_call_error", "unknown_option", "type"}, :raises},
     "Nx.Constants.pi(:f32, type: :f64)"},
    {:quiet, "Nx.Constants.pi(:f32)"},
    {{:finds, {"tensor_call_error", "unknown_option", "dim"}, :raises},
     "Nx.iota({2}, type: :f64, dim: 0)"},
    {{:finds, {"tensor_call_error", "unknown_option", "axis"}, :raises},
     "Nx.Random.uniform(Nx.Random.key(1), shape: {2}, axis: 0)"},
    {{:finds_none, :accepted}, "Nx.Random.uniform(Nx.Random.key(1), shape: {2})"},
    {{:finds, {"tensor_call_error", "unknown_option", "axis"}, :raises},
     "reduce = fn x, opts -> Nx.sum(x, opts) end\nreduce.(t, axis: 0)"},
    {:quiet, "reduce = fn x, opts -> Nx.sum(x, opts) end\nreduce.(t, axes: [0])"},
    {:quiet, "Nx.reshape(t, {1}, axis: 0)"},
    # a sampler's options are read, its type among them
    {{:unsupported, "f64"}, "Nx.Random.uniform(Nx.Random.key(1), type: :f64)"},
    # options that are not a keyword list
    {{:finds, {"tensor_call_error", "options_not_keyword", "[1, 0]"}, :raises},
     "Nx.transpose(Nx.iota({2, 3}), [1, 0])"},
    {:quiet, "Nx.transpose(Nx.iota({2, 3}), axes: [1, 0])"},
    {{:finds, {"tensor_call_error", "options_not_keyword", "[0.5, 0.5]"}, :raises},
     "Nx.Random.choice(Nx.Random.key(1), Nx.iota({2}), [0.5, 0.5])"},
    {{:finds_none, :accepted},
     "Nx.Random.choice(Nx.Random.key(1), Nx.iota({2}), Nx.tensor([0.5, 0.5]))"},
    # axis options in a form Nx does not take
    {{:finds, {"tensor_call_error", "option_form", "axes: 1"}, :raises},
     "Nx.sum(Nx.iota({2, 3}), axes: 1)"},
    {:quiet, "Nx.sum(Nx.iota({2, 3}), axes: [1])"},
    {{:finds, {"tensor_call_error", "option_form", "axis: [1]"}, :raises},
     "Nx.argmax(Nx.iota({2, 3}), axis: [1])"},
    {:quiet, "Nx.argmax(Nx.iota({2, 3}), axis: 1)"},
    {{:finds, {"tensor_call_error", "option_form", "axes: 1"}, :raises},
     "x = Nx.iota({2, 3})\nNx.sum(x, axes: Nx.rank(x) - 1)"},
    {:quiet, "x = Nx.iota({2, 3})\nNx.sum(x, axes: [Nx.rank(x) - 1])"},
    {{:finds, {"tensor_call_error", "option_form", "axes: 1"}, :raises},
     "sum_along = fn x, axis -> Nx.sum(x, axes: axis) end\nsum_along.(Nx.iota({2, 3}), 1)"},
    {:quiet,
     "sum_along = fn x, axis -> Nx.sum(x, axes: [axis]) end\nsum_along.(Nx.iota({2, 3}), 1)"},
    {{:finds, {"tensor_call_error", "option_form", "axis: a list"}, :raises},
     "Nx.cumulative_sum(Nx.iota({2, 3}), axis: [config.a])"},
    # option values Nx does not take
    {{:finds, {"tensor_call_error", "option_value", "direction: :descending"}, :raises},
     "Nx.sort(t, direction: :descending)"},
    {:quiet, "Nx.sort(t, direction: :desc)"},
    {{:finds, {"tensor_call_error", "option_value", "tie_break: :first"}, :raises},
     "Nx.argmax(t, tie_break: :first)"},
    {{:finds, {"tensor_call_error", "option_value", "padding: :causal"}, :raises},
     "Nx.window_sum(t, {1}, padding: :causal)"},
    {:quiet, "Nx.window_sum(t, {1}, padding: :same)"},
    {{:finds, {"tensor_call_error", "option_value", "padding: :full"}, :raises},
     "Nx.conv(Nx.iota({1, 1, 4}, type: :f32), Nx.iota({1, 1, 2}, type: :f32), padding: :full)"},
    {{:finds, {"tensor_call_error", "option_value", "mode: :full"}, :raises},
     "Nx.LinAlg.qr(Nx.iota({2, 2}, type: :f32), mode: :full)"},
    {:quiet, "Nx.LinAlg.qr(Nx.iota({2, 2}, type: :f32), mode: :complete)"},
    {{:finds, {"tensor_call_error", "option_value", "transform_a: :adjoint"}, :raises},
     "Nx.LinAlg.triangular_solve(Nx.eye(2), Nx.iota({2}, type: :f32), transform_a: :adjoint)"},
    {:quiet,
     "Nx.LinAlg.triangular_solve(Nx.eye(2), Nx.iota({2}, type: :f32), transform_a: :conjugate)"},
    {{:finds, {"tensor_call_error", "option_value", "max_iter: nil"}, :raises},
     "Nx.LinAlg.svd(Nx.iota({2, 2}, type: :f32), max_iter: nil)"},
    {{:finds, {"tensor_call_error", "option_value", "length: :power_of_two"}, :raises},
     "Nx.irfft(t, length: :power_of_two)"},
    {:quiet, "Nx.rfft(t, length: :power_of_two)"},
    {{:finds, {"tensor_call_error", "option_value", ":constant"}, :raises},
     "Nx.pad_outer(Nx.iota({2}), :constant, [{1, 1}])"},
    {:quiet, "Nx.pad_outer(Nx.iota({2}), :reflect, [{1, 1}])"},
    {{:finds, {"tensor_call_error", "option_value", "direction: :descending"}, :raises},
     "sort_by = fn x, direction -> Nx.sort(x, direction: direction) end\nsort_by.(t, :descending)"},
    {:quiet,
     "sort_by = fn x, direction -> Nx.sort(x, direction: direction) end\nsort_by.(t, :desc)"}
  ]
  @fixture_modules_options """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.Options do
    def summed_by_axis, do: Nx.sum(Nx.iota({2, 3}), axis: 1)
    def summed_by_axes, do: Nx.sum(Nx.iota({2, 3}), axes: [1])
    def transposed_by_list, do: Nx.transpose(Nx.iota({2, 3, 4}), [0, 2, 1])
    def reshaped_after_axis, do: Nx.iota({2, 3}) |> Nx.sum(axis: 1) |> Nx.reshape({2})
    def sums(tensor, opts), do: Nx.sum(tensor, opts)
    def sums_by_axis(tensor), do: sums(tensor, axis: 0)
    def sums_by_axes(tensor), do: sums(tensor, axes: [0])
    def sums_through(tensor, opts), do: sums(tensor, opts)
    def sums_through_by_dim(tensor), do: sums_through(tensor, dim: 0)
    def sums_through_by_axes(tensor), do: sums_through(tensor, axes: [0])

    def sums_axes(tensor, axes) do
      axes = if is_list(axes), do: axes, else: [axes]
      Nx.sum(tensor, axes: axes)
    end

    def sums_first_axis(tensor), do: sums_axes(tensor, 0)
    def sorts(tensor, direction), do: Nx.sort(tensor, direction: direction)
    def sorts_descending(tensor), do: sorts(tensor, :descending)
    def sorts_down(tensor), do: sorts(tensor, :desc)
    def pi_typed_by_option, do: Nx.Constants.pi(:f32, type: :f64)
    def iota_typed_with_typo, do: Nx.iota({2}, type: :f64, dim: 0)
    def iota_typed, do: Nx.iota({2}, type: :f64)
  end
  """

  # ── Traced: priv/tensor_shapes/traced.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_traced [
    # reading a tensor's data in a function Nx traces
    {{:finds, {"tensor_call_error", "data_read_in_trace", "jit"}, :raises},
     "Nx.Defn.jit(fn x -> Nx.add(x, Nx.to_number(Nx.sum(x))) end).(t)"},
    {{:finds, {"tensor_call_error", "data_read_in_trace", "jit"}, :raises},
     "Nx.Defn.jit(fn x -> Nx.tensor(Nx.to_list(x)) end).(t)"},
    {{:finds, {"tensor_call_error", "data_read_in_trace", "grad"}, :raises},
     "Nx.Defn.grad(Nx.as_type(t, :f32), fn x -> Nx.multiply(x, Nx.to_number(Nx.sum(x))) end)"},
    {{:finds, {"tensor_call_error", "data_read_in_trace", "jit"}, :raises},
     "Nx.Defn.jit(fn x -> Nx.add(x, IO.iodata_length(Nx.serialize(x))) end).(t)"},
    {{:finds, {"tensor_call_error", "data_read_in_trace", "jit"}, :raises},
     "Nx.Defn.jit(fn x -> Nx.add(x, Nx.to_number(Nx.sum(Nx.from_binary(<<1, 2>>, :s8)))) end).(t)"},
    {{:finds, {"tensor_call_error", "data_read_in_trace", "jit"}, :raises},
     "Nx.Defn.jit(fn x -> Nx.add(x, Nx.to_number(Nx.sum(Nx.tri(2, 2)))) end).(t)"},
    {{:finds_none, :accepted}, "Nx.to_number(Nx.sum(Nx.Defn.jit(fn x -> Nx.add(x, 1) end).(t)))"},
    {{:finds_none, :accepted}, "IO.iodata_length(Nx.serialize(Nx.iota({2})))"},
    {{:finds_none, :accepted},
     "scale = Nx.tensor(3)\nNx.Defn.jit(fn x -> Nx.multiply(x, Nx.to_number(scale)) end).(t)"},
    # EXLA's traces, which run `Nx.Defn`'s with EXLA as the compiler (not
    # run: this package does not depend on EXLA)
    {{:finds, {"tensor_call_error", "data_read_in_trace", "jit"}, :any},
     "EXLA.jit(fn x -> Nx.add(x, Nx.to_number(Nx.sum(x))) end).(t)"},
    {{:finds_none, :any}, "Nx.to_number(Nx.sum(EXLA.jit(fn x -> Nx.add(x, 1) end).(t)))"},
    {{:finds, {"tensor_call_error", "captured_tensor", "jit"}, :any},
     "data = Nx.iota({1})\nEXLA.jit(fn x -> Nx.add(x, data) end).(t)"},
    # conversions of tensors whose shape Nx rejects
    {{:finds, {"tensor_shape_mismatch", "scalar", :any}, :raises},
     "Nx.to_number(Nx.argmax(Nx.iota({1, 5}), axis: -1))"},
    {:quiet, "Nx.to_number(Nx.argmax(Nx.iota({1, 5})))"},
    {{:finds, {"tensor_shape_mismatch", "scalar", :any}, :raises},
     "Nx.to_number(Nx.vectorize(Nx.iota({1}), :x))"},
    {{:finds, {"tensor_shape_mismatch", "needs_axes", :any}, :raises},
     "Nx.to_list(Nx.sum(Nx.iota({3})))"},
    {:quiet, "Nx.to_list(Nx.iota({3}))"},
    {{:finds, {"tensor_shape_mismatch", "needs_axes", :any}, :raises},
     "Nx.to_heatmap(Nx.sum(Nx.iota({3})))"},
    {:quiet, "Nx.to_heatmap(Nx.iota({2, 3}))"},
    {{:finds, {"tensor_shape_mismatch", "needs_axes", :any}, :raises},
     "Nx.to_batched(Nx.sum(Nx.iota({3})), 1)"},
    {{:finds, {"tensor_shape_mismatch", "batch", :any}, :raises},
     "Nx.to_batched(Nx.iota({2, 3}), 3)"},
    {:quiet, "Nx.to_batched(Nx.iota({4, 3}), 2) |> Enum.to_list()"},
    {{:finds, {"tensor_call_error", "batch_size", "0"}, :raises},
     "Nx.to_batched(Nx.iota({2}), 0)"},
    # Elixir's operators and truth tests applied to tensors
    {{:finds, {"tensor_call_error", "tensor_arithmetic", "*"}, :raises},
     "x = Nx.iota({2})\nx * x"},
    {:quiet, "x = Nx.iota({2})\nNx.multiply(x, x)"},
    {{:finds, {"tensor_call_error", "tensor_arithmetic", "fconv"}, :raises}, "Nx.sum(t) / 2"},
    {{:finds, {"tensor_call_error", "tensor_arithmetic", "*"}, :raises},
     "weights = %{w: Nx.iota({2})}\nweights.w * 2"},
    {{:finds, {"tensor_call_error", "tensor_arithmetic", "*"}, :raises},
     "Nx.Defn.grad(Nx.as_type(t, :f32), fn x -> Nx.sum(x * x) end)"},
    {{:finds, {"tensor_call_error", "tensor_boolean", "and"}, :raises},
     "Nx.greater(t, 0) and true"},
    {{:finds, {"tensor_call_error", "tensor_boolean", "not"}, :raises}, "not Nx.greater(t, 0)"},
    {{:finds, {"tensor_call_error", "tensor_truth", ""}, :accepted},
     "if Nx.all(Nx.less(t, 0)), do: t, else: Nx.negate(t)"},
    {:quiet, "if Nx.to_number(Nx.all(Nx.less(t, 0))) == 1, do: t, else: Nx.negate(t)"},
    {:quiet,
     "mask = if Nx.to_number(Nx.sum(t)) > 0, do: Nx.iota({1}), else: nil\nif mask, do: Nx.multiply(t, mask), else: t"},
    {{:finds, {"tensor_call_error", "tensor_comparison", "0"}, :accepted},
     "if Nx.sum(t) > 0, do: t, else: Nx.negate(t)"},
    {{:finds, {"tensor_call_error", "tensor_comparison", "0"}, :accepted}, "Nx.sum(t) == 0"},
    {:quiet, "Nx.to_number(Nx.sum(t)) > 0"},
    # an answer about a tensor's struct, which is no tensor, and a template,
    # which is one
    {{:finds_none, :accepted}, "Nx.bit_size(t) * 8"},
    {{:finds_none, :accepted}, "Nx.donatable?(t) and Nx.bit_size(t) > 8"},
    {{:finds_none, :accepted},
     "Nx.Defn.jit(fn x -> Nx.add(x, Nx.to_number(Nx.bit_size(x))) end).(t)"},
    {{:finds, {"tensor_call_error", "tensor_arithmetic", "*"}, :raises}, "Nx.to_template(t) * 8"},
    # a sampler's `{sample, key}`, which is no tensor
    {{:finds_none, :accepted},
     "Nx.Random.multivariate_normal(Nx.Random.key(1), Nx.tensor([0.0, 0.0]), Nx.eye(2)) > 0"},
    {{:finds_none, :accepted}, "Nx.Random.normal(Nx.Random.key(1)) > 0"},
    # a jitted function run while Nx traces another, and compiled templates
    {{:finds, {"tensor_call_error", "jit_in_trace", "jit"}, :raises},
     "inner = Nx.Defn.jit(fn y -> Nx.add(y, 1) end)\nNx.Defn.jit(fn x -> inner.(x) end).(t)"},
    {{:finds_none, :accepted},
     "inner = Nx.Defn.jit(fn y -> Nx.add(y, 1) end, on_conflict: :reuse)\nNx.Defn.jit(fn x -> inner.(x) end).(t)"},
    {{:finds, {"tensor_call_error", "jit_in_trace", "jit"}, :raises},
     "Nx.Defn.jit(fn x -> Nx.Defn.jit_apply(&Nx.exp/1, [x]) end).(t)"},
    {{:finds_none, :accepted},
     "Nx.Defn.jit(fn x -> Nx.Defn.jit_apply(&Nx.exp/1, [x], on_conflict: :reuse) end).(t)"},
    {{:finds, {"tensor_call_error", "jit_in_trace", "jit"}, :raises},
     "compiler = Map.get(config, :compiler, Nx.Defn.Evaluator)\ninner = Nx.Defn.jit(fn y -> Nx.add(y, 1) end, compiler: compiler)\nNx.Defn.jit(fn x -> inner.(x) end).(t)"},
    {{:finds_none, :accepted},
     "compiler = Map.get(config, :compiler, Nx.Defn.Evaluator)\ninner = Nx.Defn.jit(fn y -> Nx.add(y, 1) end, compiler: compiler, on_conflict: :reuse)\nNx.Defn.jit(fn x -> inner.(x) end).(t)"},
    {{:finds, {"tensor_call_error", "jit_in_trace", "jit"}, :raises},
     "compiler = Map.get(config, :compiler, Nx.Defn.Evaluator)\nNx.Defn.jit(fn x -> Nx.Defn.jit_apply(&Nx.exp/1, [x], compiler: compiler) end).(t)"},
    {{:finds, {"tensor_call_error", "compiled_template", :any}, :raises},
     "compiled = Nx.Defn.compile(&Nx.exp/1, [Nx.template({2}, :s32)])\ncompiled.(Nx.iota({3}))"},
    {{:finds, {"tensor_call_error", "compiled_template", :any}, :raises},
     "compiled = Nx.Defn.compile(&Nx.add/2, [Nx.template({2}, :s32), Nx.template({2}, :s32)])\ncompiled.(Nx.iota({2}), Nx.iota({2, 2}))"},
    {{:finds_none, :accepted},
     "compiled = Nx.Defn.compile(&Nx.exp/1, [Nx.template({2}, :s32)])\ncompiled.(Nx.iota({2}))"},
    # a closure handed to a trace that captures a tensor
    {{:finds, {"tensor_call_error", "captured_tensor", "grad"}, :accepted},
     "data = Nx.iota({1}, type: :f32)\nNx.Defn.grad(Nx.as_type(t, :f32), fn x -> Nx.sum(Nx.multiply(x, data)) end)"},
    {{:finds, {"tensor_call_error", "captured_tensor", "jit"}, :accepted},
     "data = Nx.iota({1})\nNx.Defn.jit(fn x -> Nx.add(x, data) end).(t)"},
    {{:finds, {"tensor_call_error", "captured_tensor", "jit"}, :accepted},
     "data = Nx.iota({4})\nNx.Defn.jit(fn x -> Nx.add(x, elem(Nx.top_k(data, k: 2), 0)) end).(t)"},
    {{:finds_none, :accepted},
     "data = Nx.iota({4})\nNx.Defn.jit(fn x -> Nx.add(x, Nx.size(data)) end).(t)"},
    {{:finds_none, :accepted},
     "data = Nx.iota({1})\nNx.Defn.jit(fn x, y -> Nx.add(x, y) end).(t, data)"},
    # templates computed with
    {{:finds, {"tensor_call_error", "template_computed", "0"}, :raises},
     "Nx.add(Nx.template({1}, :s32), t)"},
    {:quiet, "Nx.iota(Nx.shape(Nx.template({2}, :s32)))"},
    {{:finds, {"tensor_call_error", "template_computed", "0"}, :raises},
     "Nx.serialize(Nx.template({2}, :f32))"},
    {{:finds_none, :accepted}, "Nx.shape(Nx.donatable(Nx.template({2}, :f32)))"},
    {{:finds, {"tensor_call_error", "template_computed", "0"}, :raises},
     "Nx.Defn.jit(&Nx.exp/1).(Nx.template({2}, :f32))"},
    {{:finds, {"tensor_call_error", "template_computed", "0"}, :raises},
     "Nx.Defn.jit_apply(&Nx.exp/1, [Nx.template({2}, :f32)])"},
    {{:finds_none, :accepted},
     "Nx.Defn.compile(&Nx.exp/1, [Nx.template({2}, :f32)]).(Nx.iota({2}, type: :f32))"}
  ]
  @fixture_modules_traced """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.TracedDefn do
    import Nx.Defn

    defn logs_loss(x), do: x |> Nx.sum() |> logged()

    deftransformp logged(loss) do
      _ = log_value(loss)
      loss
    end

    defp log_value(tensor), do: Nx.to_number(tensor)

    deftransform shape_of(tensor), do: Nx.shape(tensor)

    defn doubled(x), do: x * 2
    defn calls_doubled(x), do: doubled(x) + Nx.size(shape_of(x))
  end
  """

  # ── DefnFlow: priv/tensor_shapes/defn_flow.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_defn_flow [
    # a Kernel operator gets an atom, which Nx makes no tensor of
    {{:finds, {"tensor_call_error", "atom_operand", :any}, :raises},
     "Nx.Defn.Kernel.__equal__(t, :train)"},
    {{:finds_none, :accepted}, "Nx.Defn.Kernel.__equal__(t, 1)"},
    # `elem/2` gives the tuple's element, shape and all
    {{:mismatch, "broadcast"}, "Nx.add(Nx.Defn.Kernel.elem({t, Nx.iota({3})}, 1), Nx.iota({2}))"},
    {:quiet, "Nx.add(Nx.Defn.Kernel.elem({t, Nx.iota({3})}, 1), Nx.iota({3}))"}
  ]
  @fixture_modules_defn_flow ~S"""
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.DefnFlow do
    import Nx.Defn

    defn positive_part(x), do: if(x > 0, do: x, else: 0)
    defn positive_all(x), do: if(Nx.all(x > 0), do: x, else: 0)
    def run_positive_part, do: positive_part(Nx.iota({3}))
    def run_positive_all, do: positive_all(Nx.iota({3}))

    defn training_scale(x, opts \\ []), do: if(opts[:training], do: x * 2, else: x)

    defn dropout_scale(x, opts \\ []) do
      opts = keyword!(opts, dropout: false)
      if opts[:dropout], do: x * 2, else: x
    end

    defn branch_sizes(x), do: if(Nx.all(x > 0), do: Nx.iota({2}), else: Nx.iota({3}))
    defn branch_sizes_agree(x), do: if(Nx.all(x > 0), do: Nx.iota({3}), else: Nx.iota({3}) * 2)
    defn branch_scalar(x), do: if(Nx.all(x > 0), do: Nx.iota({3}), else: 0)
    defn branch_grows(x), do: if(Nx.all(x > 0), do: Nx.iota({3}), else: Nx.iota({3, 1}))
    defn branch_forms(x), do: if(Nx.all(x > 0), do: {x, x}, else: x)
    defn branch_pairs(x), do: if(Nx.all(x > 0), do: {x, x}, else: {x, -x})
    def run_branch_forms, do: branch_forms(Nx.iota({3}))
    def run_branch_pairs, do: branch_pairs(Nx.iota({3}))

    defn branch_elements(x) do
      {left, right} = if Nx.all(x > 0), do: {x, Nx.iota({3})}, else: {x, Nx.iota({2})}
      left + right
    end

    defn sign_of(x) do
      cond do
        Nx.all(x > 0) -> 1
        Nx.all(x <= 0) -> -1
      end
    end

    defn sign_or_zero(x) do
      cond do
        Nx.all(x > 0) -> 1
        true -> 0
      end
    end

    defn checked_root(x), do: if(Nx.all(x >= 0), do: Nx.sqrt(x), else: raise("negative"))

    defn runtime_checked_root(x),
      do: if(Nx.all(x >= 0), do: Nx.sqrt(x), else: runtime_raise("negative"))

    defn strict_root(x, opts \\ [strict: false]),
      do: if(opts[:strict], do: raise("strict"), else: Nx.sqrt(x))

    defn scalar_accumulator(x) do
      {_i, total} =
        while {i = 0, total = 0.0}, i < 3 do
          {i + 1, total + Nx.iota({3})}
        end

      x + total
    end

    defn vector_accumulator(x) do
      {_i, total} =
        while {i = 0, total = Nx.broadcast(0.0, {3})}, i < 3 do
          {i + 1, total + Nx.iota({3})}
        end

      x + total
    end

    defn integer_total(x) do
      {_i, total, _x} =
        while {i = 0, total = 0, x}, i < 3 do
          {i + 1, total + Nx.sum(x) / 2, x}
        end

      total
    end

    defn integer_product(x) do
      {_i, total, _x} =
        while {i = 0, total = 1, x}, i < 3 do
          {i + 1, total * Nx.mean(x), x}
        end

      total
    end

    defn float_total(x) do
      {_i, total, _x} =
        while {i = 0, total = 0.0, x}, i < 3 do
          {i + 1, total + Nx.sum(x) / 2, x}
        end

      total
    end

    defn vector_condition(x) do
      {total, _i} =
        while {total = Nx.iota({3}), i = 0}, total < 3 do
          {total + 1, i}
        end

      x + total
    end

    defn scalar_condition(x) do
      {total, _i} =
        while {total = Nx.iota({3}), i = 0}, Nx.all(total < 3) do
          {total + 1, i}
        end

      x + total
    end

    defn captured_step(x, step) do
      {_i, total} =
        while {i = 0, total = x}, i < 3 do
          {i + 1, total + step}
        end

      total
    end

    defn carried_step(x, step) do
      {_i, total, _step} =
        while {i = 0, total = x, step}, i < 3 do
          {i + 1, total + step, step}
        end

      total
    end

    defn captured_length(x, y) do
      {_i, total} =
        while {i = 0, total = x}, i < Nx.axis_size(y, 0) do
          {i + 1, total + 1}
        end

      total
    end

    defn captured_reduce(x, y), do: Nx.reduce(x, 0, fn a, b -> a + b + y end)
    defn separate_reduce(x, y), do: Nx.reduce(x, 0, fn a, b -> a + b end) + Nx.sum(y)

    defn mode_scale(x, opts \\ [mode: :train]),
      do: if(opts[:mode] == :train, do: x * 2, else: x)

    defn mode_case(x, opts \\ [mode: :train]) do
      case opts[:mode] do
        :train -> x * 2
        _ -> x
      end
    end

    defn biased(x, opts \\ []), do: x + opts[:bias]

    defn shifted(x, opts \\ []) do
      opts = keyword!(opts, shift: 0.0)
      x + opts[:shift]
    end

    defn iota_of(n), do: Nx.iota({n})
    defn iota_of_option(x, opts \\ [n: 3]), do: x + Nx.iota({opts[:n]})
    defn iota_pair(n, m), do: Nx.add(Nx.iota({n}), Nx.iota({m}))
    defn broadcast_rows(x, rows), do: Nx.broadcast(x, {rows, 3})
    defn sum_along(x, axis), do: Nx.sum(x, axes: [axis])
    defn slice_first(x, length), do: Nx.slice(x, [0], [length])
    defn top_of(x, k), do: Nx.top_k(x, k: k)
    defn iota_in_tuple({x, n}), do: x + Nx.iota({n})
    defnp iota_helper(x, n), do: x + Nx.iota({n})
    defn helper_three(x), do: iota_helper(x, 3)
    defn range_to(x, n), do: while(total = x, i <- 0..n, do: total + i)
    defn range_along(x), do: while(total = x, i <- 0..(Nx.axis_size(x, 0) - 1), do: total + i)

    defn printed_dropped(x) do
      print_value(x)
      x + 1
    end

    defn printed_kept(x) do
      x = print_value(x)
      x + 1
    end

    defn second_wrong(x), do: elem({x, Nx.iota({3})}, 1) + Nx.iota({2})
    defn second_right(x), do: elem({x, Nx.iota({3})}, 1) + Nx.iota({3})

    defn watched_wrong(x),
      do: x + (Nx.iota({3}) |> print_value(&Function.identity/1, []) |> Nx.add(Nx.iota({2})))
  end
  """

  # ── Containers: priv/tensor_shapes/containers.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_containers [
    # containers handed to defn, jit, jit_apply or Nx.Batch holding values Nx cannot trace
    {{:finds, {"tensor_call_error", "container_leaf", "nil at argument 2.b"}, :raises},
     "Nx.Defn.jit(fn x, _ -> x end).(t, %{w: t, b: nil})"},
    {{:finds_none, :accepted}, "Nx.Defn.jit(fn x, _ -> x end).(t, %{w: t, b: t})"},
    {{:finds, {"tensor_call_error", "container_leaf", "true at argument 1.causal"}, :raises},
     "ArgusNxTensorAnalyses.TensorShapesTest.ContainerDefns.pass(%{weight: t, causal: true})"},
    {{:finds_none, :accepted},
     "ArgusNxTensorAnalyses.TensorShapesTest.ContainerDefns.pass(%{weight: t, causal: Nx.tensor(1)})"},
    {{:finds_none, :accepted},
     "Nx.Defn.jit(fn x -> ArgusNxTensorAnalyses.TensorShapesTest.ContainerDefns.pass(%{weight: x, causal: true}).weight end).(t)"},
    {{:finds, {"tensor_call_error", "container_leaf", "nil at argument 1.inner.b"}, :raises},
     "ArgusNxTensorAnalyses.TensorShapesTest.ContainerDefns.pass(%{inner: %{w: t, b: nil}})"},
    {{:finds, {"tensor_call_error", "container_leaf", "a list at argument 1.layers"}, :raises},
     "Nx.Defn.jit_apply(fn layers -> layers end, [%{layers: [t, t]}])"},
    {{:finds_none, :accepted}, "Nx.Defn.jit_apply(fn layers -> layers end, [%{layers: {t, t}}])"},
    {{:finds, {"tensor_call_error", "container_leaf", "a string at argument 2{1}"}, :raises},
     "Nx.Defn.jit(fn x, _ -> x end).(t, {t, \"x\"})"},
    {{:finds, {"tensor_call_error", "container_leaf", "true at entry 1{1}"}, :raises},
     "Nx.Batch.stack([{t, true}])"},
    {{:finds_none, :accepted}, "Nx.Batch.stack([{t, t}])"},
    {{:finds, {"tensor_call_error", "container_leaf", "possibly nil at argument 2.b"}, :raises},
     "bias = if Nx.to_number(Nx.sum(t)) > 0, do: t, else: nil\nNx.Defn.jit(fn x, _ -> x end).(t, %{w: t, b: bias})"},
    # a list or keyword list at the top is taken as it is
    {{:finds_none, :accepted}, "Nx.Defn.jit(fn x, _ -> x end).(t, [t, :relu])"},
    {{:finds_none, :accepted},
     "ArgusNxTensorAnalyses.TensorShapesTest.ContainerDefns.scaled(t, factor: 3)"},
    # a jitted closure takes the call's arguments ahead of what it captured
    # (each struct built as a map, which a case compiled in the fixtures'
    # file can build, and not run: Nx takes one only with its container
    # protocol consolidated again)
    {{:finds, {"tensor_call_error", "dropped_field_read", :any}, :any},
     "layer = %{__struct__: ArgusNxTensorAnalyses.TensorShapesTest.DroppingLayer, weight: t, causal: true}\nNx.Defn.jit(fn layer -> if layer.causal, do: Nx.add(layer.weight, config.a), else: layer.weight end).(layer)"},
    {{:finds_none, :any},
     "layer = %{__struct__: ArgusNxTensorAnalyses.TensorShapesTest.KeepingLayer, weight: t, causal: true}\nNx.Defn.jit(fn layer -> if layer.causal, do: Nx.add(layer.weight, config.a), else: layer.weight end).(layer)"},
    {{:finds_none, :any},
     "flags = %{causal: config.a > 0}\nlayer = %{__struct__: ArgusNxTensorAnalyses.TensorShapesTest.DroppingLayer, weight: t, causal: true}\nNx.Defn.jit(fn layer -> if flags.causal, do: layer.weight, else: 0 end).(layer)"},
    # a gradient's function capturing what it differentiates
    {{:finds, {"tensor_call_error", "captured_gradient", "{1}"}, :accepted},
     "weight = Nx.as_type(t, :f32)\nbias = Nx.add(weight, 1.0)\nNx.Defn.grad({weight, bias}, fn {w, _b} -> Nx.sum(Nx.multiply(w, bias)) end)"},
    {{:finds, {"tensor_call_error", "captured_gradient", ""}, :accepted},
     "weight = Nx.as_type(t, :f32)\nNx.Defn.grad(weight, fn w -> Nx.sum(Nx.multiply(w, weight)) end)"},
    {{:finds_none, :accepted},
     "weight = Nx.as_type(t, :f32)\nbias = Nx.add(weight, 1.0)\nNx.Defn.grad({weight, bias}, fn {w, b} -> Nx.sum(Nx.multiply(w, b)) end)"},
    # capturing a tensor not differentiated keeps its gradient, but the
    # capture itself raises on EMLX and EXLA (the traced checks' warning)
    {{:finds, {"tensor_call_error", "captured_tensor", "grad"}, :accepted},
     "weight = Nx.as_type(t, :f32)\nbias = Nx.add(weight, 1.0)\nNx.Defn.grad(weight, fn w -> Nx.sum(Nx.multiply(w, bias)) end)"}
  ]
  @fixture_modules_containers """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.DroppingLayer do
    @derive {Nx.Container, containers: [:weight]}
    defstruct [:weight, causal: false, heads: nil]
  end

  defmodule ArgusNxTensorAnalyses.TensorShapesTest.KeepingLayer do
    @derive {Nx.Container, containers: [:weight], keep: [:causal, :heads]}
    defstruct [:weight, causal: false, heads: nil]
  end

  defmodule ArgusNxTensorAnalyses.TensorShapesTest.BiasedLayer do
    @derive {Nx.Container, containers: [:weight, :bias]}
    defstruct [:weight, bias: nil]
  end

  defmodule ArgusNxTensorAnalyses.TensorShapesTest.PlainSettings do
    defstruct [:rate]
  end

  defmodule ArgusNxTensorAnalyses.TensorShapesTest.MisorderedPair do
    defstruct [:first, :second]

    defimpl Nx.Container do
      def traverse(%{first: first, second: second} = pair, accumulator, fun) do
        {first, accumulator} = fun.(first, accumulator)
        {second, accumulator} = fun.(second, accumulator)
        {%{pair | first: first, second: second}, accumulator}
      end

      def reduce(%{first: first, second: second}, accumulator, fun) do
        accumulator = fun.(second, accumulator)
        fun.(first, accumulator)
      end

      def serialize(_pair), do: raise(ArgumentError, "not serialized")
    end
  end

  defmodule ArgusNxTensorAnalyses.TensorShapesTest.OrderedPair do
    defstruct [:first, :second]

    defimpl Nx.Container do
      def traverse(%{first: first, second: second} = pair, accumulator, fun) do
        {first, accumulator} = fun.(first, accumulator)
        {second, accumulator} = fun.(second, accumulator)
        {%{pair | first: first, second: second}, accumulator}
      end

      def reduce(%{first: first, second: second}, accumulator, fun) do
        accumulator = fun.(first, accumulator)
        fun.(second, accumulator)
      end

      def serialize(_pair), do: raise(ArgumentError, "not serialized")
    end
  end

  defmodule ArgusNxTensorAnalyses.TensorShapesTest.ContainerDefns do
    import Nx.Defn

    alias ArgusNxTensorAnalyses.TensorShapesTest.DroppingLayer

    defn branch(layer, x), do: if(layer.causal, do: x * 100, else: x)
    defn by_heads(layer, x), do: Nx.reshape(x, {layer.heads, :auto})
    defn heads_of(layer, x), do: Nx.reshape(x, {layer.heads, :auto})
    defn pass(value), do: value

    defn scaled(x, opts \\\\ []) do
      opts = keyword!(opts, factor: 2)
      x * opts[:factor]
    end

    defn built_inside(x), do: branch_inside(%DroppingLayer{weight: x, causal: true}, x)
    defn branch_inside(layer, x), do: if(layer.causal, do: x * 100, else: x)

    defn traced_caller(x), do: through_transform(x)
    deftransformp through_transform(x), do: pass(%{mode: :train, value: x}).value

    defn looped(pair) do
      {_count, pair} =
        while {count = 0, pair}, count < 1 do
          {count + 1, pair}
        end

      pair
    end
  end

  defmodule ArgusNxTensorAnalyses.TensorShapesTest.ContainerUse do
    alias ArgusNxTensorAnalyses.TensorShapesTest.{
      BiasedLayer,
      ContainerDefns,
      DroppingLayer,
      KeepingLayer,
      MisorderedPair,
      OrderedPair,
      PlainSettings
    }

    def dropping_branch(t), do: ContainerDefns.branch(%DroppingLayer{weight: t, causal: true}, t)
    def keeping_branch(t), do: ContainerDefns.branch(%KeepingLayer{weight: t, causal: true}, t)

    def dropping_heads(t),
      do: ContainerDefns.by_heads(%DroppingLayer{weight: t, heads: 2}, Nx.iota({4}))

    def keeping_heads(t),
      do: ContainerDefns.by_heads(%KeepingLayer{weight: t, heads: 2}, Nx.iota({4}))

    def dropping_jitted(t),
      do: Nx.Defn.jit(&ContainerDefns.heads_of/2).(%DroppingLayer{weight: t, heads: 2}, Nx.iota({4}))

    def keeping_jitted(t),
      do: Nx.Defn.jit(&ContainerDefns.heads_of/2).(%KeepingLayer{weight: t, heads: 2}, Nx.iota({4}))

    def dropping_returned(t), do: ContainerDefns.pass(%DroppingLayer{weight: t, causal: true}).causal
    def keeping_returned(t), do: ContainerDefns.pass(%KeepingLayer{weight: t, causal: true}).causal

    def dropping_jit_returned(t),
      do: Nx.Defn.jit(&ContainerDefns.pass/1).(%DroppingLayer{weight: t, heads: 2}).heads

    def keeping_jit_returned(t),
      do: Nx.Defn.jit(&ContainerDefns.pass/1).(%KeepingLayer{weight: t, heads: 2}).heads

    def built_inside(t), do: ContainerDefns.built_inside(t)
    def traced_caller(t), do: ContainerDefns.traced_caller(t)
    def unset_bias(t), do: ContainerDefns.pass(%BiasedLayer{weight: t})
    def set_bias(t), do: ContainerDefns.pass(%BiasedLayer{weight: t, bias: t})
    def plain_settings(t), do: ContainerDefns.pass({t, %PlainSettings{rate: t}})

    def misordered_loop(t),
      do: ContainerDefns.looped(%MisorderedPair{first: t, second: Nx.tensor(7.0)})

    def ordered_loop(t), do: ContainerDefns.looped(%OrderedPair{first: t, second: Nx.tensor(7.0)})
  end
  """

  # ── Gradients: priv/tensor_shapes/gradients.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_gradients [
    # a standard deviation where every value is the same, an epsilon after it or not
    {{:nonfinite, "infinite_gradient", "spread"},
     "Nx.Defn.grad(t, fn x -> Nx.standard_deviation(x) end)"},
    {{:nonfinite, "infinite_gradient", "spread"},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.divide(Nx.subtract(x, Nx.mean(x)), Nx.add(Nx.standard_deviation(x), 1.0e-5))) end)"},
    {:finite, "Nx.Defn.grad(t, fn x -> Nx.sqrt(Nx.add(Nx.variance(x), 1.0e-5)) end)"},
    # an angle at the origin
    {{:finds, {"tensor_nonfinite_result", "gradient_at_origin", :any}, :nonfinite},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.atan2(Nx.multiply(x, x), Nx.abs(x))) end)"},
    {{:finds, {"tensor_nonfinite_result", "gradient_at_origin", :any}, :nonfinite},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.phase(Nx.complex(Nx.multiply(x, x), Nx.abs(x)))) end)"},
    {{:nonfinite, "gradient_at_origin", "input"},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.atan2(x, x)) end)"},
    {:finite,
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.atan2(Nx.multiply(x, x), Nx.add(Nx.abs(x), 1.0))) end)"},
    # an arc cosine clipped to ±1, an arc hyperbolic cosine of 1 plus a square
    {{:nonfinite, "gradient_at_edge", "clip"},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.acos(Nx.clip(Nx.add(x, 1), -1, 1))) end)"},
    {:finite,
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.acos(Nx.clip(Nx.add(x, 1), -0.999, 0.999))) end)"},
    {{:finds, {"tensor_nonfinite_result", "gradient_at_edge", "saturation"}, :finite},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.asin(Nx.tanh(x))) end)"},
    {{:nonfinite, "gradient_at_edge", "square"},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.acosh(Nx.add(1, Nx.multiply(x, x)))) end)"},
    {:finite,
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.acosh(Nx.clip(Nx.add(1, Nx.multiply(x, x)), 1.0001, 1.0e9))) end)"},
    # a logarithm, root or division a select masks out, and the double select
    {{:nonfinite, "masked_gradient", "logarithm"},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.select(Nx.greater(x, 0), Nx.log(x), 0)) end)"},
    {{:nonfinite, "masked_gradient", "logarithm"},
     "Nx.Defn.grad(t, fn p -> Nx.sum(Nx.select(Nx.greater(p, 0), Nx.multiply(p, Nx.log(p)), 0)) end)"},
    {{:nonfinite, "masked_gradient", "root"},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.select(Nx.greater(x, 0), Nx.sqrt(x), 0)) end)"},
    {{:nonfinite, "masked_gradient", "division"},
     "Nx.Defn.grad(t, fn y -> Nx.sum(Nx.select(Nx.equal(y, 0), 0, Nx.divide(1, y))) end)"},
    {:finite,
     "Nx.Defn.grad(t, fn y -> Nx.sum(Nx.select(Nx.equal(y, 0), 0, Nx.divide(1, Nx.select(Nx.equal(y, 0), 1, y)))) end)"},
    {:finite,
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.select(Nx.greater(x, 0), Nx.log(Nx.select(Nx.greater(x, 0), x, 1.0)), 0)) end)"},
    # calls Nx has no derivative for, unless stop_grad keeps the gradient out
    {{:finds, {"tensor_call_error", "no_gradient", ""}, :raises},
     "Nx.Defn.grad(t, fn x -> Nx.reduce(x, 0, fn a, b -> Nx.add(a, b) end) end)"},
    {{:finds, {"tensor_call_error", "no_gradient", ""}, :raises},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.window_product(x, {1})) end)"},
    {{:finds_none, :finite},
     "Nx.Defn.grad(t, fn x -> Nx.add(Nx.sum(x), Nx.reduce(Nx.Defn.Kernel.stop_grad(x), 0, fn a, b -> Nx.add(a, b) end)) end)"},
    {{:finds_none, :finite},
     "Nx.Defn.grad(t, fn x -> Nx.add(Nx.sum(x), Nx.sum(Nx.quotient(Nx.iota({3}), 2))) end)"},
    # an exponent differentiated where the base is negative
    {{:nonfinite, "exponent_gradient", "written"},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.pow(-2.0, x)) end)"},
    {{:nonfinite, "exponent_gradient", "negative"},
     "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.pow(Nx.subtract(x, 1), x)) end)"},
    {:finite, "Nx.Defn.grad(t, fn x -> Nx.sum(Nx.pow(2.0, x)) end)"},
    # a Fourier transform of a differentiated real value: the gradient is complex
    {{:finds, {"tensor_call_error", "complex_gradient", ""}, :finite},
     "gradient = Nx.Defn.grad(Nx.add(Nx.iota({4}, type: :f32), 1.0), fn x -> x |> Nx.rfft() |> Nx.abs() |> Nx.sum() end)\n{:c, 64} = Nx.type(gradient)\ngradient"},
    {{:finds_none, :finite},
     "Nx.Defn.grad(Nx.as_type(t, :f32), fn x -> Nx.sum(Nx.multiply(x, Nx.sum(Nx.abs(Nx.fft(Nx.iota({2}, type: :f32)))))) end)"},
    # an additive mask far below zero into a sigmoid, and a select that masks instead
    {{:nonfinite, "sigmoid_gradient_overflow", "overflow"},
     "Nx.Defn.grad(Nx.as_type(t, :f32), fn x -> Nx.sum(Nx.sigmoid(Nx.add(x, Nx.multiply(Nx.subtract(1, Nx.iota({1})), -1.0e9)))) end)"},
    {{:finds, {"tensor_nonfinite_result", "sigmoid_gradient_overflow", "half_precision"},
      :finite},
     "Nx.Defn.grad(Nx.as_type(t, :f32), fn x -> Nx.sum(Nx.sigmoid(Nx.add(x, -20.0))) end)"},
    # a mask written past f32's range, which the rules read as its largest
    {{:nonfinite, "sigmoid_gradient_overflow", "overflow"},
     "Nx.Defn.grad(Nx.as_type(t, :f32), fn x -> Nx.sum(Nx.sigmoid(Nx.add(x, Nx.multiply(Nx.subtract(1, Nx.iota({1})), -1.0e300)))) end)"},
    {:finite,
     "Nx.Defn.grad(Nx.as_type(t, :f32), fn x -> Nx.sum(Nx.sigmoid(Nx.select(Nx.equal(Nx.iota({1}), 0), -1.0e9, x))) end)"},
    # a negative exponent an option holds
    {{:nonfinite, "divide_by_zero", "absolute"},
     "opts = [p: -1]\nNx.pow(Nx.abs(Nx.as_type(t, :f32)), opts[:p])"},
    # custom_grad's gradient function: no list, too few gradients, an input not listed
    {{:finds, {"tensor_call_error", "custom_grad_not_list", ""}, :raises},
     "Nx.Defn.grad(t, fn x -> Nx.Defn.Kernel.custom_grad(Nx.multiply(x, x), [x], fn g -> g end) end)"},
    {{:finds_none, :finite},
     "Nx.Defn.grad(t, fn x -> Nx.Defn.Kernel.custom_grad(Nx.multiply(x, x), [x], fn g -> [g] end) end)"},
    {{:finds, {"tensor_call_error", "custom_grad_short", "1 of 2"}, :finite},
     "Nx.Defn.grad(t, fn x ->\n  y = Nx.multiply(x, 2)\n  Nx.Defn.Kernel.custom_grad(Nx.multiply(x, y), [x, y], fn g -> [g] end)\nend)"},
    {{:finds_none, :finite},
     "Nx.Defn.grad(t, fn x ->\n  y = Nx.multiply(x, 2)\n  Nx.Defn.Kernel.custom_grad(Nx.multiply(x, y), [x, y], fn g -> [g, g] end)\nend)"},
    {{:finds, {"tensor_call_error", "custom_grad_unlisted", ""}, :finite},
     "Nx.Defn.grad({t, t}, fn {a, b} -> Nx.Defn.Kernel.custom_grad(Nx.multiply(a, b), [a], fn g -> [Nx.multiply(g, b)] end) end)"},
    {{:finds_none, :finite},
     "Nx.Defn.grad({t, t}, fn {a, b} -> Nx.Defn.Kernel.custom_grad(Nx.multiply(a, b), [a, b], fn g -> [Nx.multiply(g, b), Nx.multiply(g, a)] end) end)"},
    # a gradient of a function returning a tuple or a map
    {{:finds, {"tensor_call_error", "gradient_of_container", "tuple"}, :raises},
     "Nx.Defn.grad(t, fn x -> {Nx.sum(x), x} end)"},
    {{:finds, {"tensor_call_error", "gradient_of_container", "map"}, :raises},
     "Nx.Defn.grad(t, fn x -> %{loss: Nx.sum(x)} end)"},
    {{:finds_none, :finite},
     "Nx.Defn.value_and_grad(t, fn x -> {Nx.sum(x), x} end, fn {loss, _} -> loss end)"},
    # decompositions of matrices built degenerate
    {{:finds, {"tensor_call_error", "singular_gradient", "rank_one"}, :raises},
     "Nx.Defn.grad(Nx.iota({2}, type: :f32), fn v -> Nx.sum(Nx.LinAlg.cholesky(Nx.outer(v, v))) end)"},
    {{:nonfinite, "degenerate_gradient", "rank_one"},
     "Nx.Defn.grad(Nx.add(Nx.iota({2}, type: :f32), 1.0), fn v -> Nx.sum(Nx.LinAlg.pinv(Nx.outer(v, v))) end)"},
    {{:finds, {"tensor_call_error", "singular_gradient", "low_rank"}, :raises},
     "Nx.Defn.grad(Nx.add(Nx.iota({2}, type: :f32), 1.0), fn v ->\n  u = Nx.reshape(v, {2, 1})\n  Nx.sum(Nx.LinAlg.cholesky(Nx.dot(u, Nx.transpose(u))))\nend)"},
    {{:nonfinite, "degenerate_gradient", "scaled_identity"},
     "Nx.Defn.grad(Nx.add(Nx.as_type(t, :f32), 2.0), fn v ->\n  {_, singular, _} = Nx.LinAlg.svd(Nx.multiply(Nx.reshape(v, {}), Nx.eye(2)))\n  Nx.sum(singular)\nend)"},
    {{:finds_none, :finite},
     "Nx.Defn.grad(Nx.iota({2}, type: :f32), fn v -> Nx.sum(Nx.LinAlg.cholesky(Nx.add(Nx.outer(v, v), Nx.eye(2)))) end)"}
  ]
  @fixture_modules_gradients """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.GradientsFixtures do
    import Nx.Defn

    defn safe_norm(x) do
      norm = Nx.LinAlg.norm(x)
      custom_grad(norm, [x], fn g -> [g * x / Nx.max(norm, 1.0e-6)] end)
    end

    defn plain_norm(x), do: Nx.LinAlg.norm(x)
    defn root_of(x, opts \\\\ []), do: x ** opts[:p]
    defn cube_root(x), do: x ** (1 / 3)
    defn cube(x), do: x ** (3 / 1)
    defn huge_power(x), do: x ** (1 / 1.0e-30)

    defn masked_log(x) do
      grad(x, fn x -> Nx.sum(Nx.select(x > 0, Nx.log(x), 0)) end)
    end

    def differentiates_safe_norm(t), do: Nx.Defn.grad(t, &safe_norm/1)
    def differentiates_plain_norm(t), do: Nx.Defn.grad(t, &plain_norm/1)

    def differentiates_root_option(t),
      do: Nx.Defn.grad(t, fn x -> Nx.sum(root_of(Nx.multiply(x, x), p: 0.5)) end)

    def differentiates_cube_root(t),
      do: Nx.Defn.grad(t, fn x -> Nx.sum(cube_root(Nx.multiply(x, x))) end)

    def differentiates_cube(t), do: Nx.Defn.grad(t, fn x -> Nx.sum(cube(Nx.multiply(x, x))) end)

    def differentiates_huge_power(t),
      do: Nx.Defn.grad(t, fn x -> Nx.sum(huge_power(Nx.multiply(x, x))) end)
  end
  """

  # ── Math: priv/tensor_shapes/math.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_math [
    # a sample from a range whose minimum can be negative can be negative,
    # and so can a normal sample
    {{:unchecked, "unchecked_logarithm", "negative"},
     "{sample, _key} = Nx.Random.uniform(Nx.Random.key(1), -1.0, 1.0)\nNx.log(sample)"},
    {{:unchecked, "unchecked_logarithm", "negative"},
     "low = Nx.subtract(Nx.as_type(t, :f32), 1.0)\n{sample, _key} = Nx.Random.uniform(Nx.Random.key(1), low, 1.0)\nNx.log(sample)"},
    {{:unchecked, "unchecked_root", "negative"},
     "{sample, _key} = Nx.Random.normal(Nx.Random.key(1))\nNx.sqrt(sample)"},
    {{:nonfinite, "divide_by_zero", "square"},
     "{sample, _key} = Nx.Random.shuffle(Nx.Random.key(1), Nx.multiply(t, t))\nNx.divide(1, sample)"},
    # a softplus, log-sigmoid or logistic written out over an exponent that
    # can be positive, and their stable forms
    {{:hazard, "exp_overflow", "softplus"}, "Nx.log(Nx.add(1, Nx.exp(t)))"},
    {{:hazard, "exp_overflow", "softplus"}, "Nx.log1p(Nx.exp(Nx.multiply(t, 2.0)))"},
    {:finite, "Nx.add(Nx.max(t, 0), Nx.log1p(Nx.exp(Nx.negate(Nx.abs(t)))))"},
    {:finite, "shifted = Nx.subtract(t, Nx.reduce_max(t))\nNx.log1p(Nx.exp(shifted))"},
    {{:hazard, "log_of_zero", "sigmoid"}, "Nx.log(Nx.sigmoid(t))"},
    {:finite, "Nx.log(Nx.sigmoid(Nx.abs(t)))"},
    {:finite, "Nx.subtract(Nx.min(t, 0), Nx.log1p(Nx.exp(Nx.negate(Nx.abs(t)))))"},
    {{:hazard, "exp_overflow", "logistic"}, "Nx.divide(Nx.exp(t), Nx.add(1, Nx.exp(t)))"},
    {{:hazard, "exp_overflow", "logistic"}, "e = Nx.exp(t)\nNx.divide(e, Nx.add(e, 1))"},
    {:finite, "Nx.sigmoid(t)"},
    # a weighted mean divides by the sum of its weights
    {{:nonfinite, "divide_by_zero", "comparison"},
     "Nx.weighted_mean(Nx.tensor([1.0, 2.0]), Nx.greater(Nx.tensor([0, 0]), 1))"},
    {:finite,
     "Nx.weighted_mean(Nx.tensor([1.0, 2.0]), Nx.add(Nx.greater(Nx.tensor([0, 0]), 1), 1.0e-6))"},
    # the signs of windows, medians and differences
    {{:nonfinite, "divide_by_zero", "comparison"},
     "Nx.divide(t, Nx.window_sum(Nx.greater(t, 0), {1}))"},
    {{:nonfinite, "divide_by_zero", "square"},
     "Nx.rsqrt(Nx.window_mean(Nx.multiply(t, t), {1}))"},
    {:finite, "Nx.rsqrt(Nx.add(Nx.window_mean(Nx.multiply(t, t), {1}), 1.0e-5))"},
    {{:nonfinite, "divide_by_zero", "absolute"}, "Nx.divide(t, Nx.window_max(Nx.abs(t), {1}))"},
    {{:nonfinite, "divide_by_zero", "absolute"}, "Nx.divide(t, Nx.median(Nx.abs(t)))"},
    {{:nonfinite, "log_of_zero", "comparison"},
     "Nx.log(Nx.window_product(Nx.greater(t, 0), {1}))"},
    {{:unchecked, "unchecked_divisor", "cancel"},
     "Nx.divide(t, Nx.diff(Nx.concatenate([t, t])))"},
    # the classes of medians, sorts, spaced points and means
    {{:type_error, "non_integer_operand", "float"},
     "Nx.take(Nx.iota({5}), Nx.median(Nx.tensor([0, 1, 2, 3])))"},
    {:quiet, "Nx.take(Nx.iota({5}), Nx.median(Nx.tensor([0, 1, 2])))"},
    {{:type_error, "non_integer_operand", "float"},
     "Nx.take(Nx.iota({5}), Nx.median(Nx.iota({2, 3}), axis: 0))"},
    {:quiet, "Nx.take(Nx.iota({5}), Nx.median(Nx.iota({2, 3}), axis: 1))"},
    {{:type_error, "non_integer_operand", "float"},
     "Nx.take_along_axis(Nx.tensor([3, 1, 2]), Nx.argsort(Nx.tensor([3, 1, 2]), type: :f32))"},
    {:quiet, "Nx.take_along_axis(Nx.tensor([3, 1, 2]), Nx.argsort(Nx.tensor([3, 1, 2])))"},
    {{:type_error, "non_integer_operand", "float"},
     "Nx.take(Nx.iota({3}), Nx.linspace(0, 2, n: 3))"},
    {:quiet, "Nx.take(Nx.iota({3}), Nx.linspace(0, 2, n: 3, type: :s32))"},
    {{:type_error, "non_integer_operand", "float"},
     "Nx.bitwise_and(Nx.window_mean(Nx.iota({4}), {2}), 1)"},
    {:quiet, "Nx.bitwise_and(Nx.window_sum(Nx.iota({4}), {2}), 1)"},
    {{:type_error, "non_integer_operand", "float"},
     "Nx.take(Nx.iota({2}), Nx.covariance(Nx.iota({3, 2})))"},
    # random samples from zero, which are exactly zero now and then
    {{:hazard, "log_of_zero", "sample"},
     "Nx.log(Nx.Random.uniform_split(Nx.Random.key(1), 0.0, 1.0))"},
    {:finite, "Nx.log(Nx.Random.uniform_split(Nx.Random.key(1), 1.0e-7, 1.0))"},
    {{:hazard, "log_of_zero", "sample"},
     "{sample, _key} = Nx.Random.uniform(Nx.Random.key(1))\nNx.log(sample)"},
    {:finite,
     "{sample, _key} = Nx.Random.uniform(Nx.Random.key(1), 1.0e-7, 1.0)\nNx.log(sample)"},
    {{:hazard, "divide_by_zero", "sample"},
     "Nx.divide(1, Nx.Random.randint_split(Nx.Random.key(1), 0, 10))"},
    {{:unchecked, "unchecked_logarithm", "sample"},
     "Nx.log(Nx.Random.uniform_split(Nx.Random.key(1), Nx.negate(Nx.abs(t)), 1.0))"},
    # logarithms to a base: of zero, and to a base that can be 1
    {{:nonfinite, "log_of_zero", "absolute"}, "Nx.log(Nx.abs(t), 2)"},
    {{:nonfinite, "log_base_one", "clip"}, "Nx.log(Nx.exp(t), Nx.clip(t, 1, 5))"},
    {{:finds, {"tensor_nonfinite_result", "log_base_one", "size"}, :raises},
     "{n} = Nx.shape(Nx.iota({config.heads}))\nNx.log(Nx.exp(t), n)"},
    {:finite, "Nx.log(Nx.exp(t), 2)"},
    # products of fractions over many elements, which underflow to zero
    {{:nonfinite, "log_of_zero", "product_underflow"},
     "Nx.log(Nx.product(Nx.broadcast(0.1, {50})))"},
    {{:hazard, "log_of_zero", "product_underflow"},
     "Nx.log(Nx.product(Nx.sigmoid(Nx.iota({config.heads}))))"},
    {{:nonfinite, "divide_by_zero", "product_underflow"},
     "Nx.divide(1, Nx.cumulative_product(Nx.broadcast(0.1, {50})))"},
    {:finite, "Nx.sum(Nx.log(Nx.broadcast(0.1, {50})))"},
    {:finite, "Nx.log(Nx.product(Nx.broadcast(0.1, {5})))"},
    # comparisons with NaN, which have one value whatever the other operand
    {{:finds, {"tensor_call_error", "nan_comparison", "equal"}, :finite},
     "Nx.equal(t, Nx.Constants.nan())"},
    {{:finds, {"tensor_call_error", "nan_comparison", "not_equal"}, :finite},
     "Nx.not_equal(t, :nan)"},
    {:quiet, "Nx.is_nan(t)"},
    # integer powers whose exponent can be negative
    {{:finds, {"tensor_call_error", "integer_negative_power", :any}, :raises},
     "Nx.pow(Nx.add(Nx.iota({3}), 1), -1)"},
    {{:finds, {"tensor_call_error", "integer_negative_power", :any}, :raises},
     "Nx.pow(2, Nx.subtract(Nx.iota({3}), 1))"},
    {:finite, "Nx.pow(2.0, Nx.subtract(Nx.iota({3}), 1))"},
    # constants made in a type that has no such value
    {{:finds, {"tensor_call_error", "constant_type", "s32"}, :raises}, "Nx.Constants.nan(:s32)"},
    {{:finds, {"tensor_call_error", "constant_type", "s32"}, :raises},
     "Nx.Constants.epsilon({:s, 32})"},
    {{:finds, {"tensor_call_error", "constant_type", "c64"}, :raises},
     "Nx.Constants.max_finite(:c64)"},
    {{:finds, {"tensor_call_error", "constant_type", "f32"}, :raises}, "Nx.Constants.i(:f32)"},
    {:quiet, "Nx.Constants.nan(:f32)"},
    {:quiet, "Nx.Constants.max_finite(:s32)"},
    # the floor, ceiling or integer part of a logarithm to base 2 or 10
    {{:finds, {"tensor_call_error", "rounded_logarithm", "floor"}, :finite},
     "Nx.floor(Nx.log2(Nx.tensor([8192, 32768, 67108864])))"},
    {{:finds, {"tensor_call_error", "rounded_logarithm", "ceil"}, :finite},
     "Nx.ceil(Nx.divide(Nx.log(Nx.tensor([8.0, 16.0])), Nx.log(2)))"},
    {{:finds, {"tensor_call_error", "rounded_logarithm", "truncation"}, :finite},
     "Nx.as_type(Nx.log2(Nx.tensor([8192, 32768])), :s32)"},
    {:quiet, "Nx.subtract(31, Nx.count_leading_zeros(Nx.tensor([8192, 32768, 67108864])))"},
    # a log-sum-exp scaled by a factor that can be zero, or negative
    {{:nonfinite, "log_of_zero", "comparison"},
     "Nx.logsumexp(Nx.multiply(t, 1.0), exp_scaling_factor: Nx.greater(t, 0))"},
    {{:unchecked, "unchecked_logarithm", "negative"},
     "Nx.logsumexp(Nx.multiply(t, 1.0), exp_scaling_factor: Nx.subtract(t, 1))"},
    {:finite,
     "Nx.logsumexp(Nx.multiply(t, 1.0), exp_scaling_factor: Nx.add(Nx.greater(t, 0), 1.0e-6))"}
  ]
  @fixture_modules_math """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.MathDefn do
    import Nx.Defn

    defn softplus(x), do: Nx.log(1 + Nx.exp(x))
    defn logistic(x), do: Nx.exp(x) / (1 + Nx.exp(x))
    defn stable_softplus(x), do: Nx.max(x, 0) + Nx.log1p(Nx.exp(-Nx.abs(x)))

    defn log_of_sample(key) do
      {sample, _key} = Nx.Random.uniform(key, shape: {8})
      Nx.log(sample)
    end

    defn log_of_centered_sample(key) do
      {sample, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {8})
      Nx.log(sample)
    end

    defn log_of_shifted_sample(key) do
      {sample, _key} = Nx.Random.uniform(key, 1.0e-7, 1.0, shape: {8})
      Nx.log(sample)
    end
  end
  """

  # ── Indices: priv/tensor_shapes/indices.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_indices [
    # indices the code's math can take below zero
    {{:finds, {"tensor_call_error", "negative_index", "subtract"}, :raises},
     "Nx.take(Nx.iota({3}), Nx.subtract(t, 1))"},
    {:quiet, "Nx.take(Nx.iota({3}), Nx.max(Nx.subtract(t, 1), 0))"},
    {{:finds, {"tensor_call_error", "negative_index", "written"}, :raises},
     "Nx.take_along_axis(Nx.iota({3}), Nx.select(Nx.equal(t, 0), -100, t), axis: 0)"},
    {:quiet,
     "labels = Nx.select(Nx.equal(t, 0), -100, t)\nNx.take_along_axis(Nx.iota({3}), Nx.select(Nx.less(labels, 0), 0, labels), axis: 0)"},
    {{:finds, {"tensor_call_error", "negative_index", "written"}, :raises},
     "Nx.take(Nx.iota({3}), Nx.tensor([0, -1]))"},
    {{:finds, {"tensor_call_error", "negative_index", "remainder"}, :raises},
     "Nx.take(Nx.iota({3}), Nx.remainder(Nx.subtract(t, 1), 3))"},
    {:quiet, "Nx.take(Nx.iota({3}), Nx.remainder(Nx.add(t, 2), 3))"},
    {{:finds, {"tensor_call_error", "negative_index", "remainder"}, :raises},
     "Nx.take(Nx.iota({3}), ArgusNxTensorAnalyses.TensorShapesTest.IndicesHelpers.previous_class(t))"},
    {:quiet,
     "Nx.take(Nx.iota({3}), ArgusNxTensorAnalyses.TensorShapesTest.IndicesHelpers.next_class(t))"},
    {{:finds, {"tensor_call_error", "negative_index", "subtract"}, :raises},
     "Nx.gather(Nx.iota({3}), Nx.new_axis(Nx.subtract(t, 1), -1))"},
    {{:finds, {"tensor_call_error", "negative_index", "subtract"}, :raises},
     "Nx.indexed_add(Nx.iota({3}), Nx.new_axis(Nx.subtract(t, 1), -1), Nx.tensor([5]))"},
    {{:finds, {"tensor_call_error", "negative_index", "negate"}, :raises},
     "Nx.indexed_put(Nx.iota({3}), Nx.new_axis(Nx.negate(Nx.add(t, 1)), -1), Nx.tensor([5]))"},
    # slice starts Nx moves into the tensor
    {{:finds, {"tensor_call_error", "negative_slice_start", "written"}, :accepted},
     "Nx.slice_along_axis(Nx.iota({6}), -1, 1)"},
    {:quiet, "Nx.slice_along_axis(Nx.iota({6}), 5, 1)"},
    {{:finds, {"tensor_call_error", "negative_slice_start", "written"}, :accepted},
     "Nx.slice(Nx.iota({2, 6}), [0, -1], [2, 1])"},
    {{:finds, {"tensor_call_error", "negative_slice_start", "written"}, :accepted},
     "Nx.put_slice(Nx.iota({6}), [-1], Nx.tensor([9]))"},
    {{:finds, {"tensor_call_error", "negative_slice_start", "subtract"}, :accepted},
     "Nx.slice_along_axis(Nx.iota({6}), Nx.squeeze(Nx.subtract(t, 2)), 2)"},
    {:quiet, "Nx.slice_along_axis(Nx.iota({6}), Nx.squeeze(Nx.max(Nx.subtract(t, 2), 0)), 2)"},
    {{:finds, {"tensor_call_error", "negative_slice_start", "subtract"}, :accepted},
     "Nx.slice(Nx.iota({2, 6}), [0, Nx.Defn.Kernel.-(Nx.sum(t), 1)], [2, 1])"},
    {:quiet,
     "Nx.slice(Nx.iota({2, 6}), [0, Nx.Defn.Kernel.max(Nx.Defn.Kernel.-(Nx.sum(t), 1), 0)], [2, 1])"},
    {:quiet, "Nx.slice_along_axis(t, Nx.axis_size(t, 0) - 1, 1)"},
    {{:finds, {"tensor_call_error", "slice_past_end", :any}, :accepted},
     "Nx.slice(Nx.iota({6}), [5], [3])"},
    {:quiet, "Nx.slice(Nx.iota({6}), [3], [3])"},
    {{:finds, {"tensor_call_error", "slice_past_end", :any}, :accepted},
     "Nx.put_slice(Nx.iota({6}), [5], Nx.iota({2}))"},
    {{:finds, {"tensor_call_error", "slice_past_end", :any}, :accepted},
     "Nx.slice_along_axis(Nx.iota({2, config.heads}), 1, config.heads, axis: 1)"},
    {:quiet, "Nx.slice_along_axis(Nx.iota({2, config.heads}), 0, config.heads, axis: 1)"},
    {{:finds, {"tensor_call_error", "slice_past_end", :any}, :accepted},
     "Nx.slice_along_axis(t, 1, Nx.axis_size(t, 0))"},
    {{:type_error, "non_integer_start", "float"}, "Nx.slice(Nx.iota({6}), [1.0], [2])"},
    {{:type_error, "non_integer_start", "float"},
     "Nx.slice_along_axis(Nx.iota({6}), Nx.divide(Nx.sum(t), 2), 2)"},
    {{:type_error, "non_integer_start", "float"},
     "Nx.slice(Nx.iota({2, 6}), [0, Nx.divide(Nx.sum(t), 2)], [2, 2])"},
    {:quiet, "Nx.slice(Nx.iota({2, 6}), [0, Nx.sum(t)], [2, 2])"},
    {{:finds, {"tensor_type_error", "non_integer_start", "float"}, :accepted},
     "Nx.slice_along_axis(Nx.iota({6}), if(config.a > 1, do: 1.5, else: 1), 2)"},
    {{:finds, {"tensor_type_error", "non_integer_start", "float"}, :accepted},
     "Nx.slice(Nx.iota({2, 6}), [0, if(config.a > 1, do: 1.5, else: 1)], [2, 2])"},
    {{:type_error, "non_integer_start", "float"},
     "Nx.slice(Nx.iota({2, 6}), [1.5, if(config.a > 1, do: 1.5, else: 1)], [2, 2])"},
    # a ddof at or past the count a spread divides by
    {{:finds, {"tensor_nonfinite_result", "ddof_not_below_count", "equal"}, :nonfinite},
     "Nx.variance(Nx.iota({1, 2}, type: :f32), axes: [0], ddof: 1)"},
    {:quiet, "Nx.variance(Nx.iota({2, 2}, type: :f32), axes: [0], ddof: 1)"},
    {{:finds, {"tensor_nonfinite_result", "ddof_not_below_count", "equal"}, :nonfinite},
     "Nx.standard_deviation(Nx.iota({3}, type: :f32), axes: [], ddof: 1)"},
    {{:finds, {"tensor_nonfinite_result", "ddof_not_below_count", "greater"}, :finite},
     "Nx.covariance(Nx.iota({2, 2}, type: :f32), ddof: 3)"},
    {:quiet, "Nx.covariance(Nx.iota({3, 2}, type: :f32), ddof: 1)"},
    {{:finds, {"tensor_call_error", "negative_ddof", "-1"}, :accepted},
     "Nx.variance(Nx.iota({2}, type: :f32), ddof: -1)"},
    # random ranges
    {{:finds, {"tensor_call_error", "random_range_empty", "5 to 5"}, :raises},
     "Nx.Random.randint(Nx.Random.key(1), 5, 5)"},
    {:quiet, "Nx.Random.randint(Nx.Random.key(1), 5, 6)"},
    {{:finds, {"tensor_call_error", "random_range_reversed", "10 to 1"}, :accepted},
     "Nx.Random.randint(Nx.Random.key(1), 10, 1)"},
    {{:finds, {"tensor_call_error", "random_range_reversed", "1.0 to 0.0"}, :accepted},
     "Nx.Random.uniform(Nx.Random.key(1), 1.0, 0.0)"},
    {:quiet, "Nx.Random.uniform(Nx.Random.key(1), 0.0, 1.0)"},
    {:quiet, "Nx.Random.uniform(Nx.Random.key(1), 1.0e-40, 1.0)"},
    {{:finds, {"tensor_call_error", "random_range_outside_type", "0 to 300 as u8"}, :accepted},
     "Nx.Random.randint(Nx.Random.key(1), 0, 300, type: :u8)"},
    {{:finds, {"tensor_call_error", "random_range_outside_type", "0 to 256 as u8"}, :raises},
     "Nx.Random.randint(Nx.Random.key(1), 0, 256, type: :u8)"},
    {{:finds, {"tensor_call_error", "random_range_outside_type", "-5 to 5 as u8"}, :accepted},
     "Nx.Random.randint(Nx.Random.key(1), -5, 5, type: :u8)"},
    {:quiet, "Nx.Random.randint(Nx.Random.key(1), 0, 255, type: :u8)"},
    {{:finds, {"tensor_call_error", "random_type_not_integer", "float"}, :raises},
     "Nx.Random.randint(Nx.Random.key(1), 0.0, 5.0)"},
    {{:finds, {"tensor_call_error", "random_type_not_integer", "float"}, :raises},
     "Nx.Random.randint(Nx.Random.key(1), 0, 5, type: :f32)"},
    {{:finds, {"tensor_call_error", "random_bound_truncated", "float"}, :accepted},
     "Nx.Random.randint(Nx.Random.key(1), 0, 2.5, type: :s32)"},
    # complex operands to calls that reject them
    {{:type_error, "complex_operand", "complex"}, "Nx.argmax(Nx.fft(Nx.iota({4})))"},
    {:quiet, "Nx.argmax(Nx.abs(Nx.fft(Nx.iota({4}))))"},
    {{:type_error, "complex_operand", "complex"}, "Nx.reduce_max(Nx.fft(Nx.iota({4})))"},
    {{:type_error, "complex_operand", "complex"}, "Nx.sort(Nx.rfft(Nx.iota({4})))"},
    {{:type_error, "complex_operand", "complex"}, "Nx.greater(Nx.complex(t, t), 0)"},
    {{:type_error, "complex_operand", "complex"}, "Nx.max(Nx.conjugate(t), 1)"},
    {{:type_error, "complex_operand", "complex"}, "Nx.rfft(Nx.fft(Nx.iota({4})))"},
    {{:type_error, "complex_operand", "complex"},
     "Nx.floor(Nx.multiply(Nx.fft(Nx.iota({4})), 2))"},
    {{:type_error, "complex_operand", "complex"}, "Nx.logsumexp(Nx.iota({4}, type: :c64))"},
    {{:type_error, "complex_operand", "complex"},
     "Nx.remainder(Nx.multiply(t, Nx.Constants.i()), 2)"},
    {{:type_error, "complex_operand", "complex"}, "Nx.Defn.Kernel.max(Nx.ifft(Nx.iota({4})), 1)"},
    {{:type_error, "complex_operand", "complex"},
     "Nx.LinAlg.svd(Nx.reshape(Nx.fft(Nx.iota({4})), {2, 2}))"},
    {{:type_error, "complex_operand", "complex"},
     "Nx.LinAlg.norm(Nx.reshape(Nx.fft(Nx.iota({4})), {2, 2}), ord: :nuclear)"},
    {:quiet, "Nx.LinAlg.norm(Nx.fft(Nx.iota({4})))"},
    {{:finds, {"tensor_type_error", "complex_spread", "complex"}, :accepted},
     "Nx.variance(Nx.fft(Nx.iota({4})))"},
    {:quiet, "Nx.variance(Nx.abs(Nx.fft(Nx.iota({4}))))"}
  ]
  @fixture_modules_indices """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.IndicesHelpers do
    import Nx.Defn

    defn previous_class(labels), do: rem(labels - 1, 3)
    defn next_class(labels), do: rem(labels + 1, 3)

    defn empty_range(key), do: Nx.Random.randint(key, 5, 5)
    defn narrow_range(key), do: Nx.Random.randint(key, 0, 300, type: :u8)
    defn fitting_range(key), do: Nx.Random.randint(key, 0, 255, type: :u8)
  end
  """

  # ── Literals: priv/tensor_shapes/literals.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_literals [
    # integers past s32 that Nx types s32 before any wider type reaches them
    {{:finds, {"tensor_type_error", "integer_past_s32", "1700000000000 as s32"}, :accepted},
     "Nx.add(Nx.tensor(0, type: :s64), 1_700_000_000_000)"},
    {{:finds_none, :accepted},
     "Nx.add(Nx.tensor(0, type: :s64), Nx.tensor(1_700_000_000_000, type: :s64))"},
    {{:finds, {"tensor_type_error", "integer_past_s32", "3000000000 as s32"}, :accepted},
     "Nx.tensor(3_000_000_000)"},
    {{:finds_none, :accepted}, "Nx.tensor(3_000_000_000, type: :s64)"},
    {{:finds, {"tensor_type_error", "integer_past_s32", "3000000000 as s32"}, :accepted},
     "Nx.tensor([1, 3_000_000_000], names: [:x])"},
    {{:finds_none, :accepted}, "Nx.tensor([1.0, 3_000_000_000])"},
    {{:finds, {"tensor_type_error", "integer_past_s32", "5000000000 as s32"}, :accepted},
     "Nx.greater(Nx.tensor(3_000_000_000, type: :s64), 5_000_000_000)"},
    {{:finds, {"tensor_type_error", "integer_past_s32", "4294967295 as s32"}, :accepted},
     "Nx.bitwise_and(Nx.tensor(0x1FFFFFFFF, type: :u64), 0xFFFFFFFF)"},
    {{:finds_none, :accepted},
     "Nx.bitwise_and(Nx.tensor(0x1FFFFFFFF, type: :u64), Nx.u64(0xFFFFFFFF))"},
    {{:finds, {"tensor_type_error", "integer_past_s32", "5000000000 as s32"}, :accepted},
     "Nx.max(Nx.tensor(0, type: :s64), 5_000_000_000)"},
    {{:finds, {"tensor_type_error", "integer_past_s32", "-2147483649 as s32"}, :accepted},
     "Nx.clip(Nx.tensor(0, type: :s64), -2_147_483_649, 0)"},
    {{:finds, {"tensor_type_error", "integer_past_s32", "3000000000 as s32"}, :accepted},
     "Nx.Random.randint(Nx.Random.key(1), 0, 3_000_000_000, type: :s64)"},
    {{:finds, {"tensor_type_error", "integer_past_s32", "2147483648 as s32"}, :accepted},
     "Nx.add(t, 2_147_483_648)"},
    {{:finds_none, :accepted}, "Nx.add(t, 2_147_483_647)"},
    {{:finds_none, :accepted}, "Nx.add(t, -2_147_483_648)"},
    {{:finds_none, :accepted}, "Nx.as_type(3_000_000_000, :s64)"},
    # an integer of ten digits within s32 has its value
    {{:mismatch, "axis"}, "Nx.sum(Nx.iota({2}), axes: [2_000_000_000])"},
    # short atom types where Nx.Type takes only tuples
    {{:finds, {"tensor_type_error", "atom_type_rejected", ":f32"}, :raises},
     "Nx.Type.float?(:f32)"},
    {{:finds_none, :finite}, "Nx.Type.float?({:f, 32})"},
    {{:finds_none, :finite}, "Nx.Type.float?(Nx.Type.normalize!(:f32))"},
    {{:finds, {"tensor_type_error", "atom_type_rejected", ":bf16"}, :raises},
     "Nx.Type.merge(:bf16, {:f, 32})"},
    {{:finds, {"tensor_type_error", "atom_type_rejected", ":f32"}, :raises},
     "Nx.Type.to_floating(:f32)"},
    {{:finds, {"tensor_type_error", "atom_type_misread", ":f64"}, :accepted},
     "Nx.Type.to_complex(:f64)"},
    {{:finds_none, :accepted}, "Nx.Type.to_complex(:f32)"},
    {{:finds, {"tensor_type_error", "atom_type_misread", ":c128"}, :accepted},
     "Nx.Type.to_real(:c128)"},
    {{:finds, {"tensor_type_error", "atom_type_misread", ":u8"}, :accepted},
     "Nx.Type.to_aggregate(:u8)"},
    {{:finds_none, :accepted}, "Nx.Type.to_aggregate(:s64)"},
    {{:finds, {"tensor_type_error", "atom_type_misread", ":f32"}, :accepted},
     "Nx.Type.infinite_float?(:f32)"},
    {{:finds, {"tensor_type_error", "atom_type_rejected", ":f32"}, :raises},
     "if Nx.Type.float?(Map.get(config, :type, :f32)), do: Nx.as_type(t, :f32), else: t"},
    {{:finds, {"tensor_type_error", "atom_type_rejected", ":f32"}, :raises},
     "Nx.Random.gumbel(Nx.Random.key(1), type: :f32)"},
    {{:finds, {"tensor_type_error", "atom_type_rejected", ":f32"}, :raises},
     "Nx.Random.gumbel(Nx.Random.key(1), shape: {2}, type: Map.get(config, :type, :f32))"},
    {{:finds, {"tensor_type_error", "atom_type_rejected", ":f16"}, :raises},
     "Nx.Random.gumbel_split(Nx.Random.key(1), type: :f16)"},
    {{:finds_none, :finite}, "Nx.Random.gumbel(Nx.Random.key(1), type: {:f, 32})"},
    # types Nx does not have
    {{:finds, {"tensor_type_error", "invalid_type", ":float32"}, :raises},
     "Nx.iota({2}, type: :float32)"},
    {{:finds_none, :accepted}, "Nx.iota({2}, type: :f32)"},
    {{:finds_none, :accepted}, "Nx.iota({2}, type: :f8_e4m3fn)"},
    {{:finds, {"tensor_type_error", "invalid_type", ":int64"}, :raises}, "Nx.as_type(t, :int64)"},
    {{:finds, {"tensor_type_error", "invalid_type", ":bfloat16"}, :raises},
     "Nx.Constants.pi(:bfloat16)"},
    {{:finds_none, :finite}, "Nx.Constants.pi(:bf16)"},
    {{:finds, {"tensor_type_error", "invalid_type", "{:f, 128}"}, :raises},
     "Nx.tensor(1, type: {:f, 128})"},
    {{:finds, {"tensor_type_error", "invalid_type", ":float32"}, :raises},
     "Nx.Random.uniform(Nx.Random.key(1), type: :float32)"},
    {{:finds, {"tensor_type_error", "invalid_type", ":float32"}, :raises},
     "Nx.iota({2}, type: Map.get(config, :type, :float32))"},
    {{:finds_none, :accepted}, "Nx.tensor(1, type: nil)"},
    # literal data a type written at the call cannot hold
    {{:finds, {"tensor_type_error", "literal_wraps", "300 as s8"}, :accepted},
     "Nx.tensor([300, 301], type: :s8)"},
    {{:finds_none, :accepted}, "Nx.tensor([300, 301], type: :s16)"},
    {{:finds, {"tensor_type_error", "literal_wraps", "-1 as u8"}, :accepted},
     "Nx.tensor(-1, type: :u8)"},
    {{:finds, {"tensor_type_error", "literal_wraps", "256 as u8"}, :accepted},
     "Nx.u8([256, -1])"},
    {{:finds, {"tensor_type_error", "literal_wraps", "300 as u8"}, :accepted},
     "Nx.tensor([true, 300], type: :u8)"},
    {{:finds, {"tensor_type_error", "literal_wraps", "18446744073709551616 as u64"}, :accepted},
     "Nx.tensor(18_446_744_073_709_551_616, type: :u64)"},
    {{:finds_none, :accepted}, "Nx.tensor(18_446_744_073_709_551_615, type: :u64)"},
    {{:finds, {"tensor_type_error", "literal_wraps", "2147483648 as s32"}, :accepted},
     "Nx.s32(2_147_483_648)"},
    {{:finds, {"tensor_type_error", "literal_wraps", "300 as u8"}, :accepted},
     "Nx.as_type(300, :u8)"},
    {{:finds, {"tensor_type_error", "literal_wraps", "300 as u8"}, :accepted},
     "Nx.tensor(300, type: Map.get(config, :type, :u8))"},
    {{:finds, {"tensor_type_error", "float_as_integer", "1.5 as s32"}, :raises},
     "Nx.tensor(1.5, type: :s32)"},
    {{:finds, {"tensor_type_error", "float_as_integer", "2.0 as s16"}, :raises},
     "Nx.tensor([1, 2.0], type: :s16)"},
    {{:finds, {"tensor_type_error", "float_as_integer", ":nan as s32"}, :raises},
     "Nx.tensor(:nan, type: :s32)"},
    {{:finds_none, :accepted}, "Nx.tensor([1, 2], type: :s16)"},
    {{:finds, {"tensor_type_error", "literal_overflows", "7.0e4 as f16"}, :nonfinite},
     "Nx.tensor(70_000.0, type: :f16)"},
    {{:finds, {"tensor_type_error", "literal_overflows", "70000.0 as f16"}, :nonfinite},
     "Nx.tensor([1.0, 70_000.0], type: :f16)"},
    {{:finds, {"tensor_type_error", "literal_overflows", "70000 as f16"}, :nonfinite},
     "Nx.f16(70_000)"},
    {{:finds_none, :finite}, "Nx.tensor(65_504.0, type: :f16)"},
    {{:finds, {"tensor_type_error", "literal_overflows", "1.0e39 as bf16"}, :nonfinite},
     "Nx.tensor(1.0e39, type: :bf16)"},
    {{:finds, {"tensor_type_error", "literal_flushes", "1.0e-8 as f16"}, :finite},
     "Nx.tensor(1.0e-8, type: :f16)"},
    {{:finds_none, :finite}, "Nx.tensor(6.0e-8, type: :f16)"},
    {{:finds, {"tensor_type_error", "literal_flushes", "1.0e-46 as f32"}, :finite},
     "Nx.tensor([1.0e-46], type: :f32)"},
    {{:finds_none, :finite}, "Nx.tensor([1.0e-40], type: :bf16)"},
    # f8 keeps an f16's top byte: it overflows where f16 does, and flushes
    # below 2^-16 less half an f16 subnormal
    {{:finds, {"tensor_type_error", "literal_overflows", "65520.0 as f8"}, :nonfinite},
     "Nx.tensor([65520.0], type: :f8)"},
    {{:finds_none, :finite}, "Nx.tensor([65519.0], type: :f8)"},
    {{:finds, {"tensor_type_error", "literal_flushes", "1.52e-5 as f8"}, :finite},
     "Nx.tensor([1.52e-5], type: :f8)"},
    {{:finds_none, :finite}, "Nx.tensor([1.523e-5], type: :f8)"},
    # a map the compiler keeps as one literal, read by key or field
    {{:finds, {"tensor_type_error", "invalid_type", ":float32"}, :raises},
     "Nx.iota({2}, type: Map.get(%{type: :float32}, :type))"},
    {{:finds_none, :accepted}, "Nx.iota({2}, type: Map.get(%{type: :f32}, :type))"},
    {{:finds, {"tensor_type_error", "invalid_type", ":float32"}, :raises},
     "Nx.iota({2}, type: Map.get(%{\"name\" => \"x\", :type => :float32}, :type))"},
    {{:finds, {"tensor_type_error", "invalid_type", ":float32"}, :raises},
     "config = %{type: :float32, size: 1}\nconfig = %{config | size: Nx.size(t)}\nNx.iota({2}, type: config.type)"},
    {{:finds, {"tensor_call_error", "option_form", "axes: 0"}, :raises},
     "call_with(fn options -> Nx.sum(Nx.iota({2, 2}), axes: Map.get(options, :axes)) end, %{axes: 0})"},
    {{:finds_none, :accepted},
     "call_with(fn options -> Nx.sum(Nx.iota({2, 2}), axes: Map.get(options, :axes)) end, %{axes: [0]})"},
    {{:finds, {"tensor_call_error", "option_form", "axes: 0"}, :raises},
     "Nx.sum(Nx.iota({2, 2}), axes: Map.get(%{axes: 0}, :axes))"},
    {{:finds_none, :accepted}, "Nx.sum(Nx.iota({2, 2}), axes: Map.get(%{axes: [0]}, :axes))"},
    {{:finds, {"tensor_call_error", "literal_underflow", "1.0e-12 f16"}, :finite},
     "Nx.add(Nx.f16([0.0]), Map.get(%{eps: 1.0e-12}, :eps))"},
    # a literal map or tuple handed to a jitted function
    {{:finds, {"tensor_call_error", "container_leaf", "nil at argument 1.bias"}, :raises},
     "Nx.Defn.jit(fn m -> m.scale end).(%{scale: 2.0, bias: nil})"},
    {{:finds_none, :accepted}, "Nx.Defn.jit(fn m -> m.scale end).(%{scale: 2.0, bias: 0.0})"},
    {{:finds, {"tensor_call_error", "container_leaf", "nil at argument 1{1}"}, :raises},
     "Nx.Defn.jit(fn {a, _b} -> a end).({2.0, nil})"},
    {{:finds_none, :accepted}, "Nx.Defn.jit(fn {a, _b} -> a end).({2.0, 1.0})"}
  ]

  @literal_fixtures ArgusNxTensorAnalyses.TensorShapesTest.LiteralFixtures
  @literal_type_fixtures ArgusNxTensorAnalyses.TensorShapesTest.LiteralTypeFixtures

  # Calls of the functions of `Nx.Type` that take a type, `:type` standing
  # for each short atom type in turn. Of the functions of a type's
  # constants (`nan_binary/1`, `pi_binary/1`, ...) `min_binary/1` and
  # `nan_binary/1` stand for the rest.
  @literal_type_functions [
    integer?: [:type],
    float?: [:type],
    complex?: [:type],
    infinite_float?: [:type],
    to_string: [:type],
    to_floating: [:type],
    to_complex: [:type],
    to_real: [:type],
    to_aggregate: [:type],
    merge: [:type, {:f, 32}],
    merge: [{:f, 32}, :type],
    merge_number: [:type, 1.0],
    cast_number!: [:type, 1],
    min_binary: [:type],
    nan_binary: [:type],
    normalize!: [:type]
  ]
  @literal_type_names ~w(s2 s4 s8 s16 s32 s64 u2 u4 u8 u16 u32 u64 f8 f16 bf16 f32 f64 f8_e4m3fn c64 c128)a
  @literal_type_calls Enum.with_index(
                        for {function, arguments} <- @literal_type_functions,
                            type <- @literal_type_names do
                          {function, Enum.map(arguments, &if(&1 == :type, do: type, else: &1))}
                        end,
                        1
                      )

  @fixture_modules_literals """
  defmodule #{inspect(@literal_fixtures)} do
    import Nx.Defn

    defp floating?(type), do: Nx.Type.float?(type)
    def floats_through_helper(t), do: if(floating?(:f32), do: Nx.as_type(t, :f32), else: t)
    defp wide_mask, do: 0xFFFFFFFFFF
    def masks_through_helper(t), do: Nx.bitwise_and(t, wide_mask())
    defn offset_past_s32(t), do: t + 3_000_000_000
    defn picks_past_s32(t), do: t[4_294_967_296 - 4_294_967_295]
    defp valid_config?(config), do: Nx.Type.float?(config.type)

    def validates_config(t),
      do: if(valid_config?(%{type: :f32, width: t}), do: Nx.as_type(t, :f32), else: t)

    defp state_type(type), do: Nx.Type.merge(type, {:f, 32})
    def keeps_state(t, config), do: Nx.as_type(t, state_type(config.type))
    def keeps_written_state(t), do: keeps_state(t, %{type: :bf16, width: t})
  end

  defmodule #{inspect(@literal_type_fixtures)} do
  #{Enum.map_join(@literal_type_calls, fn {{function, arguments}, index} -> "  def call_#{index}, do: Nx.Type.#{function}(#{Enum.map_join(arguments, ", ", &inspect/1)})\n" end)}
  end
  """

  # ── Access: priv/tensor_shapes/access.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_access [
    # indexing a scalar
    {{:finds, {"tensor_call_error", "access_scalar", :any}, :raises}, "Nx.sum(Nx.iota({4}))[0]"},
    {:quiet, "Nx.iota({4})[0]"},
    # an index or a range past the end of its axis
    {{:finds, {"tensor_call_error", "access_out_of_bounds", :any}, :raises},
     "Nx.iota({4, 5})[4]"},
    {:quiet, "Nx.iota({4, 5})[3]"},
    {:quiet, "Nx.iota({4, 5})[[.., -5]]"},
    {:quiet, "Nx.iota({4, 5})[1..3//1]"},
    # a range stepping backwards, and one that holds no index
    {{:finds, {"tensor_call_error", "access_negative_step", :any}, :raises},
     "Nx.iota({4, 5})[1..-1//-1]"},
    {:quiet, "Nx.iota({4, 5})[1..-1//1]"},
    {{:finds, {"tensor_call_error", "access_empty_range", :any}, :raises},
     "Nx.iota({4, 5})[3..1//1]"},
    {:quiet, "Nx.iota({4, 5})[-3..-1//1]"},
    # more indices than axes, and names the tensor does not have or names twice
    {{:finds, {"tensor_call_error", "access_too_many_indices", :any}, :raises},
     "Nx.iota({4, 5})[[0, 0, 0]]"},
    {:quiet, "Nx.iota({4, 5})[[0, 0]]"},
    {{:finds, {"tensor_call_error", "access_unknown_name", :any}, :raises},
     "Nx.iota({4, 5}, names: [:a, :b])[c: 1]"},
    {{:finds, {"tensor_call_error", "access_duplicate_name", :any}, :raises},
     "Nx.iota({4, 5}, names: [:a, :b])[[a: 1, a: 0]]"},
    {:quiet, "Nx.iota({4, 5}, names: [:a, :b])[[b: 1, a: 0]]"},
    # floats where an index is expected
    {{:finds, {"tensor_call_error", "access_float_index", :any}, :raises},
     "Nx.iota({4, 5})[1.0]"},
    {:quiet, "Nx.iota({4, 5})[[0, 1]]"},
    {{:finds, {"tensor_call_error", "access_float_index_tensor", :any}, :raises},
     "Nx.iota({4, 5})[Nx.divide(Nx.iota({2}), 2)]"},
    {:quiet, "Nx.iota({4, 5})[Nx.quotient(Nx.iota({2}), 2)]"},
    {{:finds, {"tensor_call_error", "access_tensor_in_list", :any}, :raises},
     "Nx.iota({4, 5})[[Nx.iota({2})]]"},
    {:quiet, "Nx.iota({4, 5})[[Nx.tensor(1)]]"},
    # an index the tensor cannot be updated at
    {{:finds, {"tensor_call_error", "access_update", :any}, :raises},
     "x = Nx.iota({4, 5})\nput_in(x[0], Nx.iota({5}))"},
    {:quiet, "Nx.put_slice(Nx.iota({4, 5}), [0, 0], Nx.iota({1, 5}))"},
    # a scalar tensor index clamped into its axis
    {{:finds, {"tensor_call_error", "access_index_clamped", :any}, :accepted},
     "Nx.iota({4, 5})[Nx.tensor(7)]"},
    {:quiet, "Nx.iota({4, 5})[Nx.tensor(3)]"},
    # a slice's shape meets another
    {{:mismatch, "broadcast"}, "Nx.add(Nx.iota({4, 5})[[.., -1]], Nx.iota({5}))"},
    {:quiet, "Nx.add(Nx.iota({4, 5})[[.., -1]], Nx.iota({4}))"},
    {{:mismatch, "broadcast"}, "Nx.add(Nx.iota({config.heads, 3})[[.., 1..2//1]], Nx.iota({3}))"},
    {:quiet, "Nx.add(Nx.iota({config.heads, 3})[[.., 1..2//1]], Nx.iota({2}))"},
    # a tuple of tensors where a tensor is expected
    {{:finds, {"tensor_call_error", "tuple_as_tensor", :any}, :raises},
     "Nx.sum(Nx.split(Nx.iota({4}), 1))"},
    {{:finds, {"tensor_call_error", "tuple_as_tensor", :any}, :raises},
     "Nx.add(Nx.top_k(Nx.iota({4}), k: 2), 1)"},
    {{:finds, {"tensor_call_error", "tuple_as_tensor", :any}, :raises},
     "Nx.sum(Nx.Defn.value_and_grad(Nx.iota({4}, type: :f32), &Nx.sum/1))"},
    {{:finds, {"tensor_call_error", "tuple_as_tensor", :any}, :raises},
     "Nx.reshape({Nx.iota({3}), Nx.iota({3})}, {6})"},
    {{:finds, {"tensor_call_error", "tuple_as_tensor", :any}, :raises},
     "Nx.multiply(2, Nx.Random.uniform(Nx.Random.key(1), shape: {2}))"},
    {:quiet, "{left, _right} = Nx.split(Nx.iota({4}), 1)\nNx.sum(left)"},
    {:quiet, "Nx.concatenate({Nx.iota({3}), Nx.iota({3})})"},
    {:quiet, "Nx.Defn.Kernel.stop_grad({Nx.iota({3}), Nx.iota({3})})"},
    {{:finds, {"tensor_call_error", "tensors_in_tensor_data", :any}, :raises},
     "Nx.tensor([Nx.sum(Nx.iota({2})), Nx.sum(Nx.iota({3}))])"},
    {:quiet, "Nx.stack([Nx.sum(Nx.iota({2})), Nx.sum(Nx.iota({3}))])"},
    # squeezing every axis of size 1
    {{:finds, {"tensor_call_error", "squeeze_input_size", :any}, :accepted},
     "batch = Nx.axis_size(t, 0)\nNx.squeeze(Nx.iota({batch, 3}))"},
    {:quiet, "batch = Nx.axis_size(t, 0)\nNx.squeeze(Nx.iota({batch, 1, 3}), axes: [1])"},
    {{:mismatch, "broadcast"},
     "Nx.iota({config.heads, 1, 3}) |> Nx.squeeze() |> Nx.add(Nx.iota({4}))"},
    {:quiet, "Nx.iota({config.heads, 1, 3}) |> Nx.squeeze() |> Nx.add(Nx.iota({3}))"}
  ]

  # Bodies with the finding the analysis reports of them, and what Nx does:
  # raises that very message (`:message`), raises another wording of it
  # (`:raises`), or accepts the body (`:accepted`).
  @access_findings [
    {"Nx.iota({4, 5})[4]", "access_out_of_bounds",
     "index 4 is out of bounds for axis 0 in shape {4, 5}", :message},
    {"Nx.iota({4, 5})[[.., -6]]", "access_out_of_bounds",
     "index -6 is out of bounds for axis 1 in shape {4, 5}", :message},
    {"Nx.iota({4, 5})[1..7//1]", "access_out_of_bounds",
     "index 7 is out of bounds for axis 0 in shape {4, 5}", :message},
    {"Nx.iota({4, 5})[1..-1//-1]", "access_negative_step",
     "range step must be positive, got range: 1..-1//-1", :raises},
    {"Nx.iota({4, 5})[[.., 3..0//-2]]", "access_negative_step",
     "range step must be positive, got range: 3..0//-2", :message},
    {"Nx.iota({4, 5})[3..1//1]", "access_empty_range",
     "slicing a tensor requires a non-empty range, got: 3..1//1", :message},
    {"Nx.iota({4, 5})[-1..-3//1]", "access_empty_range",
     "slicing a tensor requires a non-empty range, got: 3..1//1", :message},
    {"Nx.iota({4, 5})[[0, 0, 0]]", "access_too_many_indices",
     "unknown or duplicate axis 2 found when slicing shape {4, 5}", :message},
    {"Nx.iota({4, 5}, names: [:a, :b])[c: 1]", "access_unknown_name",
     "tensor does not have name :c. The tensor names are: [:a, :b]", :message},
    {"Nx.iota({4, 5}, names: [:a, :b])[[a: 1, a: 0]]", "access_duplicate_name",
     "unknown or duplicate axis 0 found when slicing shape {4, 5}", :message},
    {"Nx.iota({4, 5})[1.0]", "access_float_index",
     "tensor[slice] expects slice to be an integer, a scalar tensor, a range, or a list or keyword list of them. Got 1.0",
     :raises},
    {"Nx.iota({4, 5})[[0, 1.0]]", "access_float_index",
     "slicing a tensor on an axis requires an integer, a scalar tensor or a range, got: 1.0",
     :message},
    {"Nx.iota({4, 5})[Nx.divide(Nx.iota({2}), 2)]", "access_float_index_tensor",
     "indices must be an integer tensor, got a float type", :raises},
    {"Nx.iota({4, 5})[Nx.divide(Nx.tensor(2), 2)]", "access_float_index_tensor",
     "index must be integer type, got a float type for axis 0", :raises},
    {"Nx.iota({4, 5})[[Nx.iota({2})]]", "access_tensor_in_list",
     "tensor must be a scalar when accessing a list/keyword of dimensions", :raises},
    {"x = Nx.iota({4, 5})\nput_in(x[0], Nx.iota({5}))", "access_update",
     "Access.get_and_update/3 is not supported. Please use Nx.put_slice/3 instead", :message},
    {"x = Nx.iota({4, 5})\nupdate_in(x[0], &Nx.add(&1, 1))", "access_update",
     "Access.get_and_update/3 is not supported. Please use Nx.put_slice/3 instead", :message},
    {"x = Nx.iota({4, 5})\npop_in(x[0])", "access_update",
     "Access.pop/2 is not yet supported by Nx.Tensor", :message},
    {"Nx.iota({4, 5})[Nx.tensor(7)]", "access_index_clamped",
     "a scalar tensor index of 7 into axis 0 of size 4 reads index 3", :accepted},
    {"Nx.iota({4, 5})[Nx.tensor(-1)]", "access_index_clamped",
     "a scalar tensor index of -1 into axis 0 of size 4 reads index 0", :accepted},
    {"Nx.sum(Nx.split(Nx.iota({4}), 1))", "tuple_as_tensor", "0", :raises},
    {"Nx.multiply(2, Nx.Random.uniform(Nx.Random.key(1), shape: {2}))", "tuple_as_tensor", "1",
     :raises},
    {"Nx.tensor([Nx.sum(Nx.iota({2})), Nx.sum(Nx.iota({3}))])", "tensors_in_tensor_data",
     "invalid value given to Nx.tensor/1", :raises},
    {"Nx.tensor([Nx.sum(Nx.iota({2})), 1], type: :f32)", "tensors_in_tensor_data",
     "invalid value given to Nx.tensor/1", :raises}
  ]

  # Bodies whose slice the analysis must shape as Nx does (`:nx`), or as
  # given where it cannot know a size.
  @access_shapes [
    {"Nx.iota({4, 5}, names: [:a, :b])[1]", :nx},
    {"Nx.iota({4, 5})[-1]", :nx},
    {"Nx.iota({4, 5})[[.., -1]]", :nx},
    {"Nx.iota({2, 3, 4}, names: [:x, :y, :z])[[.., 0]]", :nx},
    {"Nx.iota({4, 5})[1..-1//1]", :nx},
    {"Nx.iota({4, 5})[0..2]", :nx},
    {"Nx.iota({4, 5})[[.., 0..4//2]]", :nx},
    {"Nx.iota({4, 5})[[.., 0..3//2]]", :nx},
    {"Nx.iota({4, 5})[-2..-1//1]", :nx},
    {"Nx.iota({4, 5})[[1, ..]]", :nx},
    {"Nx.iota({2, 3, 4})[[-1, 1..2]]", :nx},
    {"Nx.iota({4, 5})[[]]", :nx},
    {"Nx.iota({4, 5}, names: [:a, :b])[a: 1]", :nx},
    {"Nx.iota({4, 5}, names: [:a, :b])[b: 1..2]", :nx},
    {"Nx.iota({4, 5}, names: [:a, :b])[[b: 1, a: 0]]", :nx},
    {"Nx.iota({4, 5})[Nx.tensor(1)]", :nx},
    {"Nx.iota({4, 5}, names: [:a, :b])[Nx.tensor([0, 2])]", :nx},
    {"Nx.iota({4, 5}, names: [:a, :b])[Nx.tensor([[0, 2]])]", :nx},
    {"Nx.iota({4, 5})[Nx.argmax(Nx.iota({3}))]", :nx},
    {"Nx.vectorize(Nx.iota({2, 4, 5}), :v)[[.., 1]]", :nx},
    {"Nx.iota({4, 5})[0][1]", :nx},
    {"t = Nx.iota({4, 5})\nt[[.., Nx.axis_size(t, 1) - 1]]", :nx},
    {"t = Nx.iota({4, 5})\nt[[Nx.axis_size(t, 0) - 1, ..]]", :nx},
    {"t = Nx.iota({4, 5})\nt[1..(Nx.axis_size(t, 0) - 1)//1]", "{?, 5}[nil, nil]"},
    {"t = Nx.iota({4, 5})\nt[[.., 1..(Nx.axis_size(t, 1) - 1)//1]]", "{4, ?}[nil, nil]"},
    {"last_position(Nx.iota({2, 3, 4}, names: [:batch, :sequence, :hidden]))", :nx},
    {"first_position(Nx.iota({2, 3, 4}))", :nx}
  ]

  @fixture_modules_access """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.AccessFixtures do
    import Nx.Defn

    defn last_position(hidden), do: hidden[[.., -1]]
    defn first_position(hidden), do: hidden[[.., 0, ..]]

    defn head_before(x) do
      k = 0
      x[0..(k - 1)]
    end

    defn head_through(x) do
      k = 2
      x[0..(k - 1)]
    end

    defn span_before(x) do
      k = 2
      x[k..(k - 1)//1]
    end

    defn span_through(x) do
      k = 1
      x[(k - 1)..k//1]
    end

    def heads_before, do: head_before(Nx.iota({4}))
    def heads_through, do: head_through(Nx.iota({4}))
    def spans_before, do: span_before(Nx.iota({4}))
    def spans_through, do: span_through(Nx.iota({4}))

  #{@access_shapes |> Enum.with_index(1) |> Enum.map_join("\n", fn {{body, _expected}, index} -> "  def shape_#{index} do\n#{body}\n  end\n" end)}
  #{@access_findings |> Enum.with_index(1) |> Enum.map_join("\n", fn {{body, _kind, _detail, _outcome}, index} -> "  def finding_#{index} do\n#{body}\n  end\n" end)}
  end
  """

  # ── Tuples: priv/tensor_shapes/tuples.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_tuples [
    # a decomposition's parts, read back where the tuple is matched or its
    # elements taken
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{u, s, _vt} = Nx.LinAlg.svd(Nx.iota({4, 2}, type: :f32))\nNx.multiply(u, s)"},
    {:quiet,
     "{u, s, _vt} = Nx.LinAlg.svd(Nx.iota({4, 2}, type: :f32), full_matrices?: false)\nNx.multiply(u, s)"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "result = Nx.LinAlg.svd(Nx.iota({4, 2}, type: :f32))\nNx.multiply(elem(result, 0), elem(result, 1))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{q, r} = Nx.LinAlg.qr(Nx.iota({4, 2}, type: :f32), mode: :complete)\nNx.add(q, r)"},
    {:quiet, "{q, r} = Nx.LinAlg.qr(Nx.iota({4, 2}, type: :f32))\nNx.dot(q, r)"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{u, s, _vt} = Nx.LinAlg.svd(Nx.iota({3, 4, 2}, type: :f32))\nNx.multiply(u, Nx.new_axis(s, 1))"},
    {:quiet,
     "{u, s, _vt} = Nx.LinAlg.svd(Nx.iota({3, 4, 2}, type: :f32), full_matrices?: false)\nNx.multiply(u, Nx.new_axis(s, 1))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{q, r} = Nx.LinAlg.qr(Nx.vectorize(Nx.iota({2, 4, 2}, type: :f32), :x))\nNx.add(q, r)"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{values, _vectors} = Nx.LinAlg.eigh(Nx.eye(3, type: :f32))\nNx.add(values, Nx.iota({2}))"},
    {:quiet,
     "{values, vectors} = Nx.LinAlg.eigh(Nx.eye(3, type: :f32))\nNx.multiply(vectors, Nx.new_axis(values, 0))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{p, _l, _u} = Nx.LinAlg.lu(Nx.eye(3, type: :f32))\nNx.add(p, Nx.iota({2}))"},
    {:quiet, "{p, l, u} = Nx.LinAlg.lu(Nx.eye(3, type: :f32))\nNx.dot(p, Nx.dot(l, u))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "Nx.add(Nx.LinAlg.cholesky(Nx.eye(3, type: :f32)), Nx.iota({2}))"},
    {:quiet, "Nx.add(Nx.LinAlg.cholesky(Nx.eye(3, type: :f32)), Nx.iota({3}))"},
    # tuples through the program's own functions
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{q, r} = ArgusNxTensorAnalyses.TensorShapesTest.TupleHelpers.decompose(Nx.iota({4, 2}, type: :f32))\nNx.add(q, r)"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "Nx.add(ArgusNxTensorAnalyses.TensorShapesTest.TupleHelpers.left_half(Nx.split(Nx.iota({5, 2}), 2)), Nx.iota({3, 2}))"},
    {{:misaligned, "size_variables"},
     "{q, _r} = ArgusNxTensorAnalyses.TensorShapesTest.TupleHelpers.decompose(Nx.iota({config.rows, 2}, type: :f32))\nNx.add(q, Nx.iota({config.cols, 1}))"},
    # Nx.split's halves
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{left, right} = Nx.split(Nx.iota({5, 2}), 2)\nNx.add(left, right)"},
    {:quiet, "{left, _right} = Nx.split(Nx.iota({5, 2}), -2)\nNx.add(left, Nx.iota({3, 2}))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{left, _right} = Nx.split(Nx.iota({5, 2}), 0.3)\nNx.add(left, Nx.iota({2, 3}))"},
    {:quiet, "{left, _right} = Nx.split(Nx.iota({5, 2}), 0.3)\nNx.add(left, Nx.iota({2, 2}))"},
    {:quiet, "{left, _right} = Nx.split(Nx.iota({5, 2}), 0.5)\nNx.add(left, Nx.iota({3, 2}))"},
    {:quiet,
     "{_left, right} = Nx.split(Nx.iota({5, 4}), 1, axis: 1)\nNx.add(right, Nx.iota({5, 3}))"},
    {{:mismatch, "split"}, "Nx.split(Nx.iota({5, 2}), 1.0)"},
    {{:mismatch, "split"}, "Nx.split(Nx.iota({5, 2}), -0.5)"},
    # a key where Nx.Random takes one
    {{:finds, {"tensor_shape_mismatch", "key", :any}, :raises},
     "Nx.Random.uniform(Nx.Random.split(Nx.Random.key(1)))"},
    {{:finds, {"tensor_shape_mismatch", "key", :any}, :raises},
     "Nx.Random.normal_split(Nx.Random.split(Nx.Random.key(1)), 0.0, 1.0)"},
    {:quiet, "keys = Nx.Random.split(Nx.Random.key(1))\nNx.Random.uniform(keys[1])"},
    # Nx.Random.fold_in: a key for each element of its data
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "Nx.add(Nx.Random.fold_in(Nx.Random.key(1), Nx.iota({3})), Nx.iota({2, 1}))"},
    {:quiet, "Nx.add(Nx.Random.fold_in(Nx.Random.key(1), Nx.iota({3})), Nx.iota({3, 1}))"},
    # samplers' samples and next keys
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{sample, _key} = Nx.Random.uniform(Nx.Random.key(1), shape: {2, 3})\nNx.add(sample, Nx.iota({3, 2}))"},
    {:quiet,
     "{sample, _key} = Nx.Random.uniform(Nx.Random.key(1), shape: {2, 3})\nNx.add(sample, Nx.iota({2, 3}))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{_sample, key} = Nx.Random.normal(Nx.Random.key(1))\nNx.add(key, Nx.iota({3}))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{sample, _key} = Nx.Random.normal(Nx.Random.key(1), Nx.iota({2, 3}, type: :f32), 1.0)\nNx.add(sample, Nx.iota({3, 2}))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "Nx.add(Nx.Random.normal_split(Nx.Random.key(1), Nx.iota({2, 3}, type: :f32), 1.0, shape: {2, 3}), Nx.iota({3, 2}))"},
    {:quiet,
     "Nx.add(Nx.Random.normal_split(Nx.Random.key(1), Nx.iota({2, 3}, type: :f32), 1.0, shape: {2, 3}), Nx.iota({2, 3}))"},
    # a normal sampler's mean and standard deviation
    {{:finds, {"tensor_call_error", "shared_draw", :any}, :accepted},
     "Nx.Random.normal(Nx.Random.key(1), Nx.iota({2, 3}, type: :f32), 1.0)"},
    {:quiet,
     "Nx.Random.normal(Nx.Random.key(1), Nx.iota({2, 3}, type: :f32), 1.0, shape: {2, 3})"},
    {{:finds, {"tensor_shape_mismatch", "sampler_parameters", :any}, :raises},
     "Nx.Random.normal(Nx.Random.key(1), Nx.iota({3}, type: :f32), 1.0, shape: {2})"},
    # bounds that do not fit the sample's shape
    {{:finds, {"tensor_shape_mismatch", "sampler_parameters", :any}, :raises},
     "Nx.Random.uniform(Nx.Random.key(1), Nx.iota({3}, type: :f32), 5.0)"},
    {{:finds, {"tensor_shape_mismatch", "sampler_parameters", :any}, :raises},
     "Nx.Random.uniform(Nx.Random.key(1), Nx.iota({3}, type: :f32), 5.0, shape: {2})"},
    {:quiet, "Nx.Random.uniform(Nx.Random.key(1), Nx.iota({3}, type: :f32), 5.0, shape: {2, 3})"},
    {{:finds, {"tensor_shape_mismatch", "sampler_parameters", :any}, :raises},
     "Nx.Random.randint(Nx.Random.key(1), Nx.iota({3}), 10)"},
    {:quiet, "Nx.Random.randint(Nx.Random.key(1), Nx.iota({3}), 10, shape: {2, 3})"},
    # shuffle and choice
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{shuffled, _key} = Nx.Random.shuffle(Nx.Random.key(1), Nx.iota({3, 4}))\nNx.add(shuffled, Nx.iota({4, 3}))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{picked, _key} = Nx.Random.choice(Nx.Random.key(1), Nx.iota({4, 3}), samples: 5, axis: 0)\nNx.add(picked, Nx.iota({4, 3}))"},
    {:quiet,
     "{picked, _key} = Nx.Random.choice(Nx.Random.key(1), Nx.iota({4, 3}), samples: 5, axis: 0)\nNx.add(picked, Nx.iota({5, 3}))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{picked, _key} = Nx.Random.choice(Nx.Random.key(1), Nx.iota({4, 3}), samples: 5)\nNx.add(picked, Nx.iota({4}))"},
    {{:mismatch, "choice"},
     "Nx.Random.choice(Nx.Random.key(1), Nx.iota({4, 3}), samples: 5, axis: 0, replace: false)"},
    {:quiet,
     "Nx.Random.choice(Nx.Random.key(1), Nx.iota({4, 3}), samples: 4, axis: 0, replace: false)"},
    {{:mismatch, "choice"}, "Nx.Random.choice(Nx.Random.key(1), Nx.iota({4, 3}), samples: 0)"},
    {{:mismatch, "choice"}, "Nx.Random.choice(Nx.Random.key(1), Nx.tensor(1))"},
    {{:mismatch, "choice"},
     "Nx.Random.choice(Nx.Random.key(1), Nx.iota({4, 3}), Nx.tensor([[0.2, 0.3, 0.5]]), samples: 2, axis: 1)"},
    {{:mismatch, "choice"},
     "Nx.Random.choice(Nx.Random.key(1), Nx.iota({4, 3}), Nx.tensor([0.2, 0.3, 0.5]), samples: 2, axis: 0)"},
    {:quiet,
     "Nx.Random.choice(Nx.Random.key(1), Nx.iota({4, 3}), Nx.tensor([0.2, 0.3, 0.5]), samples: 2, axis: 1)"},
    # a multivariate normal's sample: its shape, then the mean's size
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "{sample, _key} = Nx.Random.multivariate_normal(Nx.Random.key(1), Nx.tensor([0.0, 0.0]), Nx.eye(2), shape: {5, 2})\nNx.add(sample, Nx.iota({5, 2}))"},
    {:quiet,
     "{sample, _key} = Nx.Random.multivariate_normal(Nx.Random.key(1), Nx.tensor([0.0, 0.0]), Nx.eye(2), shape: {5})\nNx.add(sample, Nx.iota({5, 2}))"},
    {{:mismatch, "multivariate_normal"},
     "Nx.Random.multivariate_normal(Nx.Random.key(1), Nx.tensor([[0.0, 0.0]]), Nx.eye(2))"},
    {{:mismatch, "multivariate_normal"},
     "Nx.Random.multivariate_normal(Nx.Random.key(1), Nx.tensor([0.0, 0.0]), Nx.eye(3))"},
    {{:mismatch, "multivariate_normal"},
     "Nx.Random.multivariate_normal(Nx.Random.key(1), Nx.tensor([0.0, 0.0]), Nx.iota({2, 3}, type: :f32))"},
    # matrix norms
    {{:finds, {"tensor_shape_mismatch", "axis", :any}, :raises},
     "Nx.squeeze(Nx.LinAlg.norm(Nx.iota({2, 3}, type: :f32), ord: 1, keep_axes: true), axes: [1])"},
    {:quiet,
     "Nx.squeeze(Nx.LinAlg.norm(Nx.iota({2, 3}, type: :f32), keep_axes: true), axes: [1])"},
    {{:finds, {"tensor_shape_mismatch", "axis", :any}, :raises},
     "Nx.squeeze(Nx.LinAlg.norm(Nx.iota({2, 3}, type: :f32), ord: :nuclear, keep_axes: true), axes: [0])"},
    {{:finds, {"tensor_call_error", "norm_axes", :any}, :accepted},
     "Nx.LinAlg.norm(Nx.iota({2, 3}, type: :f32), ord: 1, axes: [0])"},
    {:quiet, "Nx.LinAlg.norm(Nx.iota({2, 3}, type: :f32), axes: [0])"},
    {{:mismatch, "axis"}, "Nx.LinAlg.norm(Nx.iota({2, 3}, type: :f32), ord: 1, axes: [1])"},
    {{:mismatch, "norm"}, "Nx.LinAlg.norm(Nx.iota({3}, type: :f32), ord: :frobenius)"},
    {{:mismatch, "norm"}, "Nx.LinAlg.norm(Nx.iota({2, 3}, type: :f32), ord: 3)"},
    {{:mismatch, "norm"}, "Nx.LinAlg.norm(Nx.iota({2, 3}, type: :f32), ord: :fro)"},
    {:quiet, "Nx.LinAlg.norm(Nx.iota({3}, type: :f32), ord: 3)"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "Nx.add(Nx.LinAlg.norm(Nx.iota({2, 3}, type: :f32), ord: nil, axes: [1]), Nx.iota({3}))"}
  ]
  @fixture_modules_tuples """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.TupleHelpers do
    import Nx.Defn

    def decompose(tensor), do: Nx.LinAlg.qr(tensor)
    def left_half({left, _right}), do: left

    defn key_added(x), do: x + Nx.add(Nx.Random.key(42), Nx.iota({3}))
    defn key_added_fits(x), do: x + Nx.add(Nx.Random.key(42), Nx.iota({2}))

    defn sample_added(key) do
      {sample, _key} = Nx.Random.uniform(key, shape: {2, 3})
      sample + Nx.iota({4})
    end

    defn sample_added_fits(key) do
      {sample, _key} = Nx.Random.uniform(key, shape: {2, 3})
      sample + Nx.iota({3})
    end

    defn split_drawn(key), do: Nx.Random.uniform(Nx.Random.split(key))
    defn uniform_over(key), do: Nx.Random.uniform(key, Nx.iota({3}, type: :f32), 5.0, shape: {2})

    defn parts_added(x) do
      {q, r} = Nx.LinAlg.qr(Nx.iota({4, 2}, type: :f32), mode: :complete)
      x * (q + r)
    end

    defn parts_multiplied(x) do
      {q, r} = Nx.LinAlg.qr(Nx.iota({4, 2}, type: :f32))
      x * Nx.dot(q, r)
    end
  end
  """

  # ── ShapeGaps: priv/tensor_shapes/shape_gaps.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_shape_gaps [
    # a conv laid out by permutations: the result's shape and names
    {{:finds,
      {"tensor_shape_mismatch", "broadcast",
       "cannot broadcast tensor of dimensions {1, 4, 5, 6} to {1, 4, 5, 7}"}, :raises},
     "Nx.conv(Nx.iota({1, 5, 7, 3}), Nx.iota({6, 3, 2, 3}), input_permutation: [0, 3, 1, 2], output_permutation: [0, 3, 1, 2])\n|> Nx.add(Nx.iota({1, 4, 5, 7}))"},
    {:quiet,
     "Nx.conv(Nx.iota({1, 5, 7, 3}), Nx.iota({6, 3, 2, 3}), input_permutation: [0, 3, 1, 2], output_permutation: [0, 3, 1, 2])\n|> Nx.add(Nx.iota({1, 4, 5, 6}))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "Nx.conv(Nx.iota({1, 5, 7, 3}), Nx.iota({2, 3, 3, 6}), input_permutation: [0, 3, 1, 2], kernel_permutation: [3, 2, 0, 1], output_permutation: [0, 3, 1, 2])\n|> Nx.add(Nx.iota({1, 4, 6, 6}))"},
    {:quiet,
     "Nx.conv(Nx.iota({1, 5, 7, 3}), Nx.iota({2, 3, 3, 6}), input_permutation: [0, 3, 1, 2], kernel_permutation: [3, 2, 0, 1], output_permutation: [0, 3, 1, 2])\n|> Nx.add(Nx.iota({1, 4, 5, 6}))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "Nx.conv(Nx.iota({1, 3, 5, 7}), Nx.iota({6, 3, 2, 3}), output_permutation: [0, 2, 3, 1])\n|> Nx.add(Nx.iota({1, 6, 4, 5}))"},
    {:quiet,
     "Nx.conv(Nx.iota({1, 3, 5, 7}), Nx.iota({6, 3, 2, 3}), output_permutation: [0, 2, 3, 1])\n|> Nx.add(Nx.iota({1, 5, 6, 4}))"},
    {{:finds, {"tensor_shape_mismatch", "names", :any}, :raises},
     "Nx.conv(Nx.iota({1, 5, 7, 3}, names: [:n, :h, :w, :c]), Nx.iota({6, 3, 2, 3}), input_permutation: [0, 3, 1, 2], output_permutation: [0, 3, 1, 2])\n|> Nx.add(Nx.iota({1, 4, 5, 6}, names: [:n, :h, :x, :c]))"},
    {:quiet,
     "Nx.conv(Nx.iota({1, 5, 7, 3}, names: [:n, :h, :w, :c]), Nx.iota({6, 3, 2, 3}), input_permutation: [0, 3, 1, 2], output_permutation: [0, 3, 1, 2])\n|> Nx.add(Nx.iota({1, 4, 5, 6}, names: [:n, :h, :w, :c]))"},
    {{:finds, {"tensor_shape_mismatch", "conv", :any}, :raises},
     "Nx.conv(Nx.iota({1, 5, 7, 3}), Nx.iota({6, 2, 2, 3}), input_permutation: [0, 3, 1, 2])"},
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "expected length of permutation (2) to match rank of shape (4)"}, :raises},
     "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), input_permutation: [0, 1])"},
    # a conv's per-axis options and groups
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "rank of strides much match rank of spatial dimensions got strides [1] with rank 1 and got input shape {1, 3, 5, 5} of rank 2"},
      :raises}, "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), strides: [1])"},
    {:quiet, "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), strides: [1, 2])"},
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "bad argument in arithmetic expression (ArithmeticError), from a stride of 0"}, :raises},
     "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), strides: [1, 0])"},
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "must specify dilation for each spatial dimension of the input or specify an integer dilation factor"},
      :raises}, "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), input_dilation: [1])"},
    {:quiet, "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), input_dilation: [1, 2])"},
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "kernel dilation of each dimension must be a positive integer, got 0"}, :raises},
     "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), kernel_dilation: 0)"},
    {:quiet, "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), kernel_dilation: 2)"},
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "padding must be a list of {high, low} tuples, where each element is an integer. Got: [{1, 1, 0}, {0, 0, 0}]"},
      :raises},
     "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), padding: [{1, 1, 0}, {0, 0, 0}])"},
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "invalid padding configuration, rank of padding configuration and shape must match"},
      :raises}, "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), padding: [{1, 1}])"},
    {:quiet, "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), padding: [{1, 1}, {0, 0}])"},
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "either batch groups or feature groups must be 1, got batch_groups = 2 and feature_groups = 2"},
      :raises},
     "Nx.conv(Nx.iota({2, 4, 5, 5}), Nx.iota({4, 2, 2, 2}), feature_group_size: 2, batch_group_size: 2)"},
    {:quiet, "Nx.conv(Nx.iota({2, 4, 5, 5}), Nx.iota({4, 2, 2, 2}), feature_group_size: 2)"},
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "batch groups must evenly divide input batch size got rem(2, 3) != 0"}, :raises},
     "Nx.conv(Nx.iota({3, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), batch_group_size: 2)"},
    {:quiet, "Nx.conv(Nx.iota({4, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), batch_group_size: 2)"},
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "size of kernel output channels must be evenly divisible by feature groups got rem(4, 3) != 0 for kernel with shape {4, 2, 2, 2}"},
      :raises}, "Nx.conv(Nx.iota({2, 6, 5, 5}), Nx.iota({4, 2, 2, 2}), feature_group_size: 3)"},
    {:quiet, "Nx.conv(Nx.iota({2, 6, 5, 5}), Nx.iota({3, 2, 2, 2}), feature_group_size: 3)"},
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "bad argument in arithmetic expression (ArithmeticError), from a feature_group_size of 0"},
      :raises}, "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), feature_group_size: 0)"},
    # groups of a permuted conv, against its operands as it reads them
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "size of kernel output channels must be evenly divisible by feature groups got rem(4, 3) != 0 for kernel with shape {4, 2, 2, 2}"},
      :raises},
     "Nx.conv(Nx.iota({2, 5, 5, 6}), Nx.iota({2, 2, 2, 4}), feature_group_size: 3, input_permutation: [0, 3, 1, 2], kernel_permutation: [3, 2, 0, 1])"},
    {:quiet,
     "Nx.conv(Nx.iota({2, 5, 5, 6}), Nx.iota({2, 2, 2, 3}), feature_group_size: 3, input_permutation: [0, 3, 1, 2], kernel_permutation: [3, 2, 0, 1])"},
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "batch groups must evenly divide input batch size got rem(2, 3) != 0"}, :raises},
     "Nx.conv(Nx.iota({5, 5, 3, 3}), Nx.iota({4, 3, 2, 2}), batch_group_size: 2, input_permutation: [3, 2, 0, 1])"},
    {{:finds,
      {"tensor_shape_mismatch", "conv_options",
       "either batch groups or feature groups must be 1, got batch_groups = 2 and feature_groups = 2"},
      :raises},
     "Nx.conv(Nx.iota({2, 5, 5, 4}), Nx.iota({4, 2, 2, 2}), feature_group_size: 2, batch_group_size: 2, input_permutation: [0, 3, 1, 2])"},
    # a window operation's window and options
    {{:finds,
      {"tensor_shape_mismatch", "window_options",
       "invalid window_dimensions. window_dimensions is a n-element tuple with the size of each dimension. Got: [2, 2]"},
      :raises}, "Nx.window_sum(Nx.iota({4, 4}), [2, 2])"},
    {:quiet, "Nx.window_sum(Nx.iota({4, 4}), {2, 2})"},
    {{:finds, {"tensor_shape_mismatch", "window_options", :any}, :raises},
     "Nx.window_scatter_max(Nx.iota({4, 4}), Nx.iota({2, 2}), 0, [2, 2], strides: [2, 2])"},
    {{:finds,
      {"tensor_shape_mismatch", "window_options",
       "invalid dimension in axis 0 found in window_dimensions. Each dimension must be a positive integer, got 0 in shape {0, 2}"},
      :raises}, "Nx.window_max(Nx.iota({4, 4}), {0, 2})"},
    {{:finds,
      {"tensor_shape_mismatch", "window_options",
       "invalid padding configuration, rank of padding configuration and shape must match"},
      :raises}, "Nx.window_sum(Nx.iota({4, 4}), {2, 2}, window_dilations: [1])"},
    {:quiet, "Nx.window_sum(Nx.iota({4, 4}), {2, 2}, window_dilations: [1, 2])"},
    {{:finds,
      {"tensor_shape_mismatch", "window_options",
       "dilation rates must be greater than or equal to 1 got [0, 1]"}, :raises},
     "Nx.window_sum(Nx.iota({4, 4}), {2, 2}, window_dilations: [0, 1])"},
    {{:finds,
      {"tensor_shape_mismatch", "window_options",
       "invalid padding configuration, rank of padding configuration and shape must match"},
      :raises}, "Nx.window_sum(Nx.iota({1, 2, 4, 4}), {1, 1, 2, 2}, padding: [{1, 1}, {1, 1}])"},
    {:quiet,
     "Nx.window_sum(Nx.iota({1, 2, 4, 4}), {1, 1, 2, 2}, padding: [{0, 0}, {0, 0}, {1, 1}, {1, 1}])"},
    {{:finds,
      {"tensor_shape_mismatch", "window_options",
       "bad argument in arithmetic expression (ArithmeticError), from a stride of 0"}, :raises},
     "Nx.window_mean(Nx.iota({4, 4}), {2, 2}, strides: [1, 0])"},
    {{:finds,
      {"tensor_shape_mismatch", "window_options",
       "no function clause matching in Nx.window_reduce/5 (FunctionClauseError), from window dimensions [2, 2]"},
      :raises},
     "Nx.window_reduce(Nx.iota({4, 4}), 0, [2, 2], fn element, accumulator -> Nx.add(element, accumulator) end)"},
    {{:finds,
      {"tensor_shape_mismatch", "window_options",
       "no match of right hand side value (MatchError), from a window of size 0 at axis 0"},
      :raises},
     "Nx.window_reduce(Nx.iota({4, 4}), 0, {0, 2}, [strides: [1, 1]], fn element, accumulator -> Nx.add(element, accumulator) end)"},
    {:quiet,
     "Nx.window_reduce(Nx.iota({4, 4}), 0, {2, 2}, [strides: [1, 1]], fn element, accumulator -> Nx.add(element, accumulator) end)"},
    # negative strides, which step back from the first window
    {{:finds,
      {"tensor_shape_mismatch", "conv",
       "conv would result in empty tensor which is not currently supported in Nx"}, :raises},
     "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), strides: [1, -1])"},
    {{:finds, {"tensor_shape_mismatch", "conv", :any}, :raises},
     "Nx.conv(Nx.iota({1, 5, 5, 3}), Nx.iota({4, 3, 2, 2}), strides: -2, input_permutation: [0, 3, 1, 2])"},
    {:quiet, "Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 5, 5}), strides: [-1, -1])"},
    {{:finds,
      {"tensor_shape_mismatch", "window",
       "window dimensions would result in empty tensor which is not currently supported in Nx"},
      :raises}, "Nx.window_sum(Nx.iota({4, 4}), {2, 2}, strides: [1, -1])"},
    {{:finds,
      {"tensor_shape_mismatch", "reshape",
       "cannot reshape, current shape {3, 1} is not compatible with new shape {4}"}, :raises},
     "Nx.window_sum(Nx.iota({4, 4}), {2, 2}, strides: [1, -5])\n|> Nx.reshape({4})"},
    {:quiet, "Nx.window_sum(Nx.iota({4, 4}), {2, 2}, strides: [1, -5])\n|> Nx.reshape({3})"},
    # padding that crops an axis to nothing
    {{:finds,
      {"tensor_shape_mismatch", "cropped_axis",
       "no match of right hand side value (MatchError), from padding axis 1 of size 3 to 0"},
      :raises}, "Nx.pad(Nx.iota({2, 3}), 0, [{0, 0, 0}, {0, -3, 0}])"},
    {:quiet, "Nx.pad(Nx.iota({2, 3}), 0, [{0, 0, 0}, {0, -2, 0}])"},
    {{:finds, {"tensor_shape_mismatch", "cropped_axis", :any}, :raises},
     "Nx.pad_outer(Nx.iota({2, 3}), 0, [{0, 0}, {-2, -1}])"},
    # a padding configuration the code builds, in each context
    {{:finds,
      {"tensor_shape_mismatch", "broadcast",
       "cannot broadcast tensor of dimensions {2, 5} to {2, 4}"}, :raises},
     "x = Nx.iota({2, 3})\nNx.pad(x, 0, [{0, 0, 0}, {0, 5 - Nx.axis_size(x, 1), 0}])\n|> Nx.add(Nx.iota({2, 4}))"},
    {:quiet,
     "x = Nx.iota({2, 3})\nNx.pad(x, 0, [{0, 0, 0}, {0, 5 - Nx.axis_size(x, 1), 0}])\n|> Nx.add(Nx.iota({2, 5}))"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "x = Nx.iota({2, 3})\nNx.pad_outer(x, 0, [{0, 0}, {Nx.axis_size(x, 0), 0}])\n|> Nx.add(Nx.iota({2, 4}))"},
    {{:finds,
      {"tensor_shape_mismatch", "cropped_axis",
       "no match of right hand side value (MatchError), from padding axis 1 of size 3 to 0"},
      :raises},
     "x = Nx.iota({2, 3})\nNx.pad(x, 0, [{0, 0, 0}, {-4, Nx.axis_size(x, 0) - 1, 0}])"},
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "Nx.pad(Nx.iota({2, 3}), 0, [{0, 0, 0}, {0, config.a, 0}])\n|> Nx.add(Nx.iota({3, 7}))"},
    {{:finds,
      {"tensor_shape_mismatch", "pad",
       "invalid padding configuration, rank of padding configuration and shape must match"},
      :raises}, "Nx.pad(Nx.iota({2, 3}), 0, [{0, config.a, 0}])"},
    {{:misaligned, "size_variables"},
     "Nx.pad(Nx.iota({config.a, 3}), 0, [{0, 0, 0}, {1, 1, 0}])\n|> Nx.add(Nx.iota({config.b, 5}))"},
    {:quiet,
     "Nx.pad(Nx.iota({config.a, 3}), 0, [{0, 0, 0}, {1, 1, 0}])\n|> Nx.add(Nx.iota({config.a, 5}))"},
    # sizes, counts, lengths and strides below 1
    {{:finds,
      {"tensor_shape_mismatch", "nonpositive",
       "repetitions must be a list of integers, got: [0, 1]"}, :raises},
     "Nx.tile(Nx.iota({2, 3}), [0, 1])"},
    {:quiet, "Nx.tile(Nx.iota({2, 3}), [2, 1])"},
    {{:finds,
      {"tensor_shape_mismatch", "nonpositive",
       "invalid dimension in axis 0 found in shape. Each dimension must be a positive integer, got 0 in shape {0, 3}"},
      :raises}, "Nx.iota({0, 3})"},
    {{:finds, {"tensor_shape_mismatch", "nonpositive", :any}, :raises},
     "Nx.broadcast(1, {2, 0})"},
    {{:finds, {"tensor_shape_mismatch", "nonpositive", :any}, :raises},
     "Nx.Random.uniform_split(Nx.Random.key(1), 0.0, 1.0, shape: {3, 0})"},
    {{:finds,
      {"tensor_shape_mismatch", "reshape",
       "cannot reshape, current shape {6} is not compatible with new shape {0, 6}"}, :raises},
     "Nx.reshape(Nx.iota({6}), {0, 6})"},
    {{:finds_none, :any}, "Nx.template({0, 3}, :f32)"},
    # a dimension below 1 written into a shape the code builds
    {{:finds,
      {"tensor_shape_mismatch", "nonpositive",
       "invalid dimension in axis 1 found in shape. Each dimension must be a positive integer, got 0"},
      :raises}, "Nx.iota({config.a, 0})"},
    {{:finds, {"tensor_shape_mismatch", "nonpositive", :any}, :raises},
     "Nx.broadcast(1, {config.a, -2})"},
    {{:finds,
      {"tensor_shape_mismatch", "reshape",
       "cannot reshape, current shape {6} is not compatible with new shape with 0 at axis 1"},
      :raises}, "Nx.reshape(Nx.iota({6}), {config.a, 0})"},
    {:quiet, "Nx.iota({config.a, 2})"},
    {{:finds_none, :any}, "Nx.template({config.a, 0}, :f32)"},
    {{:finds,
      {"tensor_shape_mismatch", "nonpositive",
       "top_k k must be an integer greater than or equal to 1, got k=0"}, :raises},
     "Nx.top_k(Nx.iota({3}), k: 0)"},
    {:quiet, "Nx.top_k(Nx.iota({3}), k: 2)"},
    {{:finds,
      {"tensor_shape_mismatch", "nonpositive",
       "stride at axis 0 must be greater than or equal to 1, got: 0"}, :raises},
     "Nx.slice(Nx.iota({5}), [0], [3], strides: [0])"},
    {{:finds,
      {"tensor_shape_mismatch", "nonpositive",
       "stride at axis 1 must be greater than or equal to 1, got: 0"}, :raises},
     "Nx.slice_along_axis(Nx.iota({2, 5}), 0, 2, axis: 1, strides: 0)"},
    {{:finds, {"tensor_shape_mismatch", "nonpositive", :any}, :raises},
     "Nx.fft(Nx.iota({4}), length: 0)"},
    {{:finds, {"tensor_shape_mismatch", "nonpositive", :any}, :raises},
     "Nx.fft2(Nx.iota({4, 4}), lengths: [0, 2])"},
    {{:finds,
      {"tensor_shape_mismatch", "nonpositive", "expected n to be a non-negative integer, got: 0"},
      :raises}, "Nx.linspace(0, 10, n: 0)"},
    {{:finds,
      {"tensor_shape_mismatch", "nonpositive", "expected a positive integer as length, got: 0"},
      :raises}, "Nx.irfft(Nx.iota({3}, type: :f32), length: 0)"},
    {{:finds, {"tensor_shape_mismatch", "no_tensors", "no tensors were given to concatenate"},
      :raises}, "Nx.concatenate([])"},
    {{:finds, {"tensor_shape_mismatch", "no_tensors", "no tensors were given to stack"}, :raises},
     "Nx.stack([])"},
    {{:finds,
      {"tensor_shape_mismatch", "axis",
       "new axis position for shape {3} must be a number between -2 and 1, got: 3"}, :raises},
     "Nx.stack([Nx.iota({3}), Nx.iota({3})], axis: 3)"},
    {:quiet, "Nx.stack([Nx.iota({3}), Nx.iota({3})], axis: -2)"},
    # a dot product that batches an axis it contracts
    {{:finds,
      {"tensor_shape_mismatch", "dot",
       "dot batch axes for left tensor ([0]) cannot be in contract axes ([0])"}, :raises},
     "Nx.dot(Nx.iota({2, 3, 4}), [0], [0], Nx.iota({2, 4, 5}), [1], [0])"},
    {:quiet, "Nx.dot(Nx.iota({2, 3, 4}), [2], [0], Nx.iota({2, 4, 5}), [1], [0])"},
    # a weighted mean's weights, reshaped and swapped as Nx arranges them
    {{:finds,
      {"tensor_shape_mismatch", "weighted_mean",
       "cannot broadcast tensor of dimensions {5, 2, 3} to {1, 3, 2}"}, :raises},
     "Nx.weighted_mean(Nx.iota({5, 2, 3}), Nx.iota({2, 3}), axes: [1, 2])"},
    {{:finds, {"tensor_call_error", "transposed_weights", "{2, 3} over axes [1, 2]"}, :accepted},
     "Nx.weighted_mean(Nx.iota({5, 3, 2}), Nx.iota({2, 3}), axes: [1, 2])"},
    {{:finds,
      {"tensor_shape_mismatch", "broadcast", "cannot broadcast tensor of dimensions {5} to {4}"},
      :raises},
     "Nx.weighted_mean(Nx.iota({5, 3, 2}), Nx.iota({2, 3}), axes: [1, 2])\n|> Nx.add(Nx.iota({4}))"},
    {{:finds,
      {"tensor_shape_mismatch", "weighted_mean",
       "cannot broadcast tensor of dimensions {2, 3} to {1, 2}"}, :raises},
     "Nx.weighted_mean(Nx.iota({2, 3}), Nx.iota({2}), axes: [1])"},
    {:quiet,
     "Nx.weighted_mean(Nx.iota({2, 3}), Nx.iota({2}), axes: [0])\n|> Nx.add(Nx.iota({3}))"},
    {:quiet, "Nx.weighted_mean(Nx.iota({2, 3}, names: [:a, :b]), Nx.iota({2, 3}))"},
    # ragged literal data
    {{:finds,
      {"tensor_shape_mismatch", "ragged_data",
       "cannot build tensor because lists have different shapes, got {2} at position 0 and {1} at position 2"},
      :raises}, "Nx.tensor([[1, 2], [3]])"},
    {{:finds,
      {"tensor_shape_mismatch", "ragged_data", "invalid value given to Nx.tensor/1, got: [2]"},
      :raises}, "Nx.tensor([1, [2]])"},
    {{:finds, {"tensor_shape_mismatch", "ragged_data", :any}, :raises}, "Nx.tensor([[1, 2], 3])"},
    {{:finds, {"tensor_shape_mismatch", "ragged_data", :any}, :raises},
     "Nx.tensor([[[1], [2]], [[3]]])"},
    {:quiet, "Nx.tensor([[1, 2], [3, 4]])"},
    {{:finds,
      {"tensor_shape_mismatch", "names", "cannot merge name :x on axis 0 with name :y on axis 0"},
      :raises}, "Nx.tensor(Nx.iota({2}, names: [:x]), names: [:y])"},
    {:quiet, "Nx.tensor(Nx.iota({2}, names: [:x]), names: [:x])"},
    # base-2 and base-10 logarithms keep their operand's shape
    {{:finds, {"tensor_shape_mismatch", "broadcast", :any}, :raises},
     "Nx.add(Nx.log2(Nx.add(Nx.iota({2, 3}), 1)), Nx.iota({2}))"},
    {:quiet, "Nx.add(Nx.log10(Nx.add(Nx.iota({2, 3}), 1)), Nx.iota({3}))"},
    # vectorized axes of one name and two sizes, lined up
    {{:finds,
      {"tensor_shape_mismatch", "vectorized_axes",
       "expected vectorized axis :x to have the same size in both tensors or to one of them to have size 1, got 3 and 2"},
      :raises},
     "Nx.broadcast_vectors([Nx.vectorize(Nx.iota({2, 1}), :x), Nx.vectorize(Nx.iota({3, 1}), :x)])"},
    {:quiet,
     "Nx.broadcast_vectors([Nx.vectorize(Nx.iota({2, 1}), :x), Nx.vectorize(Nx.iota({1, 1}), :x)])"},
    # an iota of a tensor, or of a number
    {{:finds, {"tensor_shape_mismatch", "tensor_as_shape", :any}, :raises},
     "Nx.iota(Nx.iota({2}))"},
    {{:finds, {"tensor_call_error", "scalar_iota", "5"}, :accepted}, "Nx.iota(5)"},
    {:quiet, "Nx.iota({5})"},
    # an inverse real transform: too short, or of an odd signal
    {{:finds,
      {"tensor_shape_mismatch", "irfft",
       "length at axis 0 must be greater than or equal to 1, got: -1"}, :raises},
     "Nx.irfft(Nx.iota({1}, type: :f32))"},
    {{:finds,
      {"tensor_shape_mismatch", "irfft",
       "length at axis 0 must be greater than or equal to 1, got: 0"}, :raises},
     "Nx.irfft(Nx.iota({3}, type: :f32), length: 2)"},
    {{:finds, {"tensor_call_error", "odd_irfft", "5"}, :accepted},
     "Nx.irfft(Nx.rfft(Nx.iota({5}, type: :f32)))"},
    {{:finds, {"tensor_call_error", "odd_irfft", "5"}, :accepted},
     "Nx.iota({5}, type: :f32)\n|> Nx.rfft()\n|> Nx.multiply(2)\n|> Nx.irfft()"},
    {:quiet, "Nx.irfft(Nx.rfft(Nx.iota({5}, type: :f32)), length: 5)"},
    {:quiet, "Nx.irfft(Nx.rfft(Nx.iota({4}, type: :f32)))"},
    # a log-sum-exp scaled by a factor of more axes than the tensor
    {{:finds, {"tensor_call_error", "scaling_factor_rank", "{2, 3} scaling {3}"}, :accepted},
     "Nx.logsumexp(Nx.iota({3}, type: :f32), axes: [0], exp_scaling_factor: Nx.iota({2, 3}, type: :f32))"},
    {:quiet,
     "Nx.logsumexp(Nx.iota({2, 3}, type: :f32), axes: [1], exp_scaling_factor: Nx.iota({2, 3}, type: :f32))"},
    # its result, the scaled sum broadcast with the tensor's maximum
    {{:finds,
      {"tensor_shape_mismatch", "broadcast", "cannot broadcast tensor of dimensions {3} to {2}"},
      :raises},
     "Nx.logsumexp(Nx.iota({3}, type: :f32), axes: [0], exp_scaling_factor: Nx.iota({2, 3}, type: :f32))\n|> Nx.add(Nx.iota({2}))"},
    {{:finds,
      {"tensor_shape_mismatch", "broadcast", "cannot broadcast tensor of dimensions {2} to {3}"},
      :raises},
     "Nx.logsumexp(Nx.iota({1, 3}, type: :f32), axes: [1], exp_scaling_factor: Nx.iota({2, 3}, type: :f32))\n|> Nx.add(Nx.iota({3}))"},
    {:quiet,
     "Nx.logsumexp(Nx.iota({1, 3}, type: :f32), axes: [1], exp_scaling_factor: Nx.iota({2, 3}, type: :f32))\n|> Nx.add(Nx.iota({2}))"},
    {:quiet,
     "Nx.logsumexp(Nx.iota({2, 3}, type: :f32), axes: [1], exp_scaling_factor: 2.0)\n|> Nx.add(Nx.iota({2}))"}
  ]
  @fixture_modules_shape_gaps """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.ShapeGaps do
    def conv_channels_last,
      do: Nx.conv(Nx.iota({1, 5, 7, 3}, names: [:batch, :height, :width, :channels]), Nx.iota({6, 3, 2, 3}), input_permutation: [0, 3, 1, 2], output_permutation: [0, 3, 1, 2])

    def conv_kernel_last,
      do: Nx.conv(Nx.iota({1, 5, 7, 3}), Nx.iota({2, 3, 3, 6}), input_permutation: [0, 3, 1, 2], kernel_permutation: [3, 2, 0, 1], output_permutation: [0, 3, 1, 2])

    def conv_output_last,
      do: Nx.conv(Nx.iota({1, 3, 5, 7}, names: [:batch, :channels, :height, :width]), Nx.iota({6, 3, 2, 3}), output_permutation: [0, 2, 3, 1])

    def conv_named_permutation,
      do: Nx.conv(Nx.iota({1, 5, 7, 3}, names: [:batch, :height, :width, :channels]), Nx.iota({6, 3, 2, 3}), input_permutation: [:batch, :channels, :height, :width])

    def weighted_transposed, do: Nx.weighted_mean(Nx.iota({5, 3, 2}), Nx.iota({2, 3}), axes: [1, 2])

    def weighted_transposed_kept,
      do: Nx.weighted_mean(Nx.iota({5, 3, 2}, names: [:a, :b, :c]), Nx.iota({2, 3}), axes: [1, 2], keep_axes: true)

    def weighted_leading, do: Nx.weighted_mean(Nx.iota({2, 3}), Nx.iota({2}), axes: [0])
    def weighted_named, do: Nx.weighted_mean(Nx.iota({2, 3}, names: [:a, :b]), Nx.iota({2, 3}), axes: [1])
    def base_two, do: Nx.log2(Nx.iota({2, 3}, names: [:a, :b]))
    def base_ten, do: Nx.log10(Nx.iota({2, 3}))
  end
  """

  # ── ReshapeOrder: priv/tensor_shapes/reshape_order.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_reshape_order [
    # heads split off with a reshape that also moves them ahead of the rows
    {{:misaligned, "reshape_order"},
     "Nx.reshape(Nx.iota({config.a, config.rows, config.heads * config.head_dim}), {config.a, config.heads, config.rows, config.head_dim})"},
    {{:misaligned, "reshape_order"},
     "Nx.reshape(Nx.iota({config.rows, config.heads * config.head_dim}), {config.heads, config.rows, :auto})"},
    {{:misaligned, "reshape_order"},
     "heads_by(config, Nx.iota({config.rows, config.heads * config.dim}))"},
    {:quiet,
     "Nx.iota({config.a, config.rows, config.heads * config.head_dim}) |> Nx.reshape({config.a, config.rows, config.heads, config.head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])"},
    # heads merged back without moving them behind the rows first
    {{:misaligned, "reshape_order"},
     "Nx.reshape(Nx.iota({config.a, config.heads, config.rows, config.head_dim}), {config.a, config.rows, config.heads * config.head_dim})"},
    {{:misaligned, "reshape_order"},
     "Nx.reshape(Nx.iota({config.a, config.heads, config.rows, config.head_dim}), {config.a, config.rows, :auto})"},
    {:quiet,
     "Nx.iota({config.a, config.heads, config.rows, config.head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({config.a, config.rows, :auto})"},
    # neighboring axes split or merged in place
    {:quiet,
     "Nx.reshape(Nx.iota({config.rows, config.heads * config.head_dim}), {config.rows, config.heads, config.head_dim})"},
    {:quiet,
     "Nx.reshape(Nx.iota({config.rows, config.heads * config.head_dim}), {config.rows, config.head_dim, config.heads})"},
    {:quiet,
     "Nx.reshape(Nx.iota({config.rows, config.heads, config.head_dim}), {config.rows, :auto})"},
    {:quiet,
     "Nx.reshape(Nx.iota({config.a, config.rows, config.cols}), {config.a * config.rows, config.cols})"},
    {:quiet,
     "Nx.reshape(Nx.iota({config.rows, config.heads * config.head_dim}), {config.rows * config.heads, config.head_dim})"},
    # literal factors moved past other axes
    {:quiet,
     "Nx.reshape(Nx.iota({config.a, config.dim * 4, config.rows, config.cols}), {config.a, config.dim, 2, 2, config.rows, config.cols})"},
    {:quiet,
     "Nx.reshape(Nx.iota({config.rows, 2 * config.cols}), {2, config.rows, config.cols})"},
    # variables the order is not compared for: on two axes, or beside a size not known
    {:quiet,
     "Nx.reshape(Nx.iota({config.dim, config.dim, config.rows}), {config.rows, config.dim * config.dim})"},
    {:quiet,
     "Nx.reshape(Nx.iota({config.rows, config.heads * config.head_dim, Enum.count([t])}), {config.heads, config.rows, :auto})"},
    {:quiet,
     "Nx.reshape(Nx.iota({config.rows, Nx.axis_size(t, 0)}), {config.heads, config.rows, :auto})"},
    {:quiet, "Nx.reshape(t, {config.heads, :auto})"}
  ]
  @fixture_modules_reshape_order """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.ReshapeOrder do
    def split_heads(config) do
      config
      |> projection()
      |> Nx.reshape({config.batch, config.heads, config.seq, config.head_dim})
    end

    def split_heads_and_transpose(config) do
      config
      |> projection()
      |> Nx.reshape({config.batch, config.seq, config.heads, config.head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])
    end

    def merge_heads(config) do
      config
      |> split_heads_and_transpose()
      |> Nx.reshape({config.batch, config.seq, :auto})
    end

    def transpose_and_merge_heads(config) do
      config
      |> split_heads_and_transpose()
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({config.batch, config.seq, :auto})
    end

    def projection(config),
      do: Nx.iota({config.batch, config.seq, config.heads * config.head_dim})
  end
  """

  # ── Dtypes: priv/tensor_shapes/dtypes.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_dtypes [
    # an unsigned difference that can go below zero wraps around
    {{:finds, {"tensor_call_error", "unsigned_wraparound", "u8"}, :accepted},
     "Nx.subtract(Nx.greater(t, 0), 1)"},
    {{:finds_none, :accepted}, "Nx.subtract(Nx.as_type(Nx.greater(t, 0), :s32), 1)"},
    {{:finds_none, :accepted}, "Nx.subtract(1, Nx.greater(t, 0))"},
    {{:finds, {"tensor_call_error", "unsigned_wraparound", "u8"}, :accepted},
     "Nx.negate(Nx.greater(t, 0))"},
    {{:finds, {"tensor_call_error", "unsigned_wraparound", "u8"}, :accepted},
     "Nx.subtract(Nx.greater(t, 0), Nx.less(t, 0))"},
    {{:finds_none, :accepted}, "Nx.sign(t)"},
    {{:finds, {"tensor_call_error", "unsigned_wraparound", "u8"}, :accepted},
     "Nx.subtract(Nx.u8([50, 200]), 128)"},
    {{:finds_none, :accepted}, "Nx.add(Nx.u8([50, 200]), -128)"},
    {{:finds, {"tensor_call_error", "unsigned_wraparound", "u32"}, :accepted},
     "Nx.subtract(Nx.u32([0]), 1)"},
    {{:finds, {"tensor_call_error", "unsigned_wraparound", "u8"}, :accepted},
     "Nx.diff(Nx.greater(Nx.tensor([1, 1, 0, 0, 1]), 0))"},
    {{:finds_none, :accepted}, "Nx.diff(Nx.cumulative_sum(Nx.greater(Nx.iota({5}), 2)))"},
    {{:finds, {"tensor_call_error", "unsigned_wraparound", "u8"}, :accepted},
     "Nx.all_close(Nx.u8(1), Nx.u8(2), atol: 2)"},
    {{:finds_none, :accepted}, "Nx.all_close(Nx.s32(1), Nx.s32(2), atol: 2)"},
    {{:finds, {"tensor_call_error", "unsigned_wraparound", "u8"}, :accepted},
     "Nx.linspace(10, 0, n: 3, type: :u8)"},
    {{:finds_none, :accepted}, "Nx.linspace(10, 0, n: 3, type: :s32)"},
    # zeros and ones counted past what a narrow type holds
    {{:finds, {"tensor_call_error", "count_wraparound", "u8 300"}, :accepted},
     "Nx.cumulative_sum(Nx.greater(Nx.iota({300}), -1))"},
    {{:finds_none, :accepted}, "Nx.cumulative_sum(Nx.greater(Nx.iota({200}), -1))"},
    {{:finds_none, :accepted},
     "Nx.cumulative_sum(Nx.as_type(Nx.greater(Nx.iota({300}), -1), :s32))"},
    {{:finds, {"tensor_call_error", "count_wraparound", "u8 300"}, :accepted},
     "Nx.window_sum(Nx.greater(Nx.iota({300}), -1), {300})"},
    {{:finds, {"tensor_call_error", "count_wraparound", "u8 300"}, :accepted},
     "onehot = Nx.equal(Nx.new_axis(Nx.remainder(Nx.iota({300}), 1), -1), Nx.iota({1, 3}))\nNx.dot(Nx.transpose(onehot), onehot)"},
    {{:finds_none, :accepted},
     "onehot = Nx.as_type(Nx.equal(Nx.new_axis(Nx.remainder(Nx.iota({300}), 1), -1), Nx.iota({1, 3})), :s32)\nNx.dot(Nx.transpose(onehot), onehot)"},
    {{:finds, {"tensor_call_error", "count_wraparound", "u8 ?"}, :accepted},
     "Nx.cumulative_sum(Nx.not_equal(t, 0), axis: 0)"},
    # narrow integers summed or multiplied in their own type
    {{:finds, {"tensor_call_error", "narrow_wraparound", "s8"}, :accepted},
     "Nx.dot(Nx.s8([100, 100]), Nx.s8([2, 2]))"},
    {{:finds_none, :accepted}, "Nx.dot(Nx.as_type(Nx.s8([100, 100]), :s32), Nx.s8([2, 2]))"},
    {{:finds, {"tensor_call_error", "narrow_wraparound", "u8"}, :accepted},
     "Nx.median(Nx.u8([[130, 140], [150, 160]]))"},
    {{:finds_none, :accepted}, "Nx.median(Nx.u8([[130, 140], [150, 160]]), axis: 1)"},
    {{:finds, {"tensor_call_error", "narrow_wraparound", "u8"}, :accepted},
     "Nx.window_mean(Nx.u8([200, 100, 50]), {2})"},
    {{:finds, {"tensor_call_error", "narrow_wraparound", "u8"}, :accepted},
     "Nx.weighted_mean(Nx.u8([200, 100]), Nx.u8([2, 2]))"},
    {{:finds, {"tensor_call_error", "narrow_wraparound", "u8"}, :accepted},
     "Nx.outer(Nx.u8([200]), Nx.u8([2]))"},
    {{:finds, {"tensor_call_error", "narrow_wraparound", "u8"}, :accepted},
     "Nx.cumulative_product(Nx.u8([16, 16]))"},
    # an index type too small for the axis searched
    {{:finds, {"tensor_call_error", "index_wraparound", "u8 300"}, :accepted},
     "Nx.argmax(Nx.iota({300}), type: :u8)"},
    {{:finds_none, :accepted}, "Nx.argmax(Nx.iota({300}))"},
    {{:finds, {"tensor_call_error", "index_wraparound", "s8 200"}, :accepted},
     "Nx.argmin(Nx.iota({200}), type: :s8)"},
    # whole numbers counted in a type that does not hold them
    {{:finds, {"tensor_call_error", "sequence_precision", "bf16 1000"}, :accepted},
     "Nx.iota({1000}, type: :bf16)"},
    {{:finds_none, :accepted}, "Nx.iota({256}, type: :bf16)"},
    {{:finds, {"tensor_call_error", "sequence_precision", "u8 300"}, :accepted},
     "Nx.iota({300}, type: :u8)"},
    {{:finds, {"tensor_call_error", "sequence_precision", "bf16 3000"}, :accepted},
     "Nx.as_type(Nx.iota({3000}), :bf16)"},
    {{:finds_none, :accepted}, "Nx.as_type(Nx.iota({3000}), :f32)"},
    {{:finds, {"tensor_call_error", "sequence_precision", "bf16 3000"}, :accepted},
     "Nx.multiply(Nx.iota({3000}), Nx.bf16(1))"},
    {{:finds_none, :accepted}, "Nx.multiply(Nx.iota({3000}), Nx.f32(1))"},
    {{:finds, {"tensor_call_error", "sequence_precision", "bf16 1000"}, :accepted},
     "Nx.linspace(0, 999, n: 1000, type: :bf16)"},
    # values made a type that does not hold them
    {{:finds, {"tensor_call_error", "cast_wraparound", "u8"}, :accepted},
     "Nx.as_type(Nx.tensor([300.0, -1.0, 2.7, -2.7]), :u8)"},
    {{:finds_none, :accepted},
     "Nx.as_type(Nx.round(Nx.clip(Nx.tensor([300.0, -1.0, 2.7, -2.7]), 0, 255)), :u8)"},
    {{:finds, {"tensor_call_error", "cast_wraparound", "u8"}, :accepted},
     "Nx.as_type(Nx.subtract(Nx.iota({3}), 1), :u8)"},
    # negative only where an input is: info
    {{:finds, {"tensor_call_error", "unchecked_cast_wraparound", "u8"}, :accepted},
     "Nx.as_type(Nx.round(Nx.multiply(t, 255)), :u8)"},
    {{:finds, {"tensor_call_error", "cast_wraparound", "u8"}, :accepted},
     "Nx.fill(Nx.u8([1]), -1, type: :u8)"},
    {{:finds_none, :accepted}, "Nx.fill(Nx.u8([1]), 0, type: :u8)"},
    {{:finds, {"tensor_call_error", "cast_wraparound", "u8"}, :accepted},
     "Nx.linspace(-2, 2, n: 3, type: :u8)"},
    {{:finds, {"tensor_call_error", "complex_to_real", "c64 f32"}, :accepted},
     "Nx.as_type(Nx.c64([1]), :f32)"},
    {{:finds_none, :accepted}, "Nx.real(Nx.c64([1]))"},
    {{:finds, {"tensor_call_error", "float_truncation", "f32 s32"}, :accepted},
     "Nx.as_type(Nx.divide(Nx.iota({3}), 2), :s32)"},
    {{:finds_none, :accepted}, "Nx.as_type(Nx.floor(Nx.divide(Nx.iota({3}), 2)), :s32)"},
    {{:finds, {"tensor_nonfinite_result", "cast_overflow", "f16"}, :nonfinite},
     "Nx.as_type(Nx.tensor(-1.0e9), :f16)"},
    {{:finds, {"tensor_nonfinite_result", "cast_overflow", "f16"}, :nonfinite},
     "Nx.as_type(Nx.Constants.min_finite(:f32), :f16)"},
    {{:finds_none, :finite}, "Nx.Constants.min_finite(:f16)"},
    # written numbers a float type makes infinite or zero
    {{:finds, {"tensor_nonfinite_result", "literal_overflow", "f16"}, :nonfinite},
     "Nx.add(Nx.f16([1, 2]), -1.0e9)"},
    {{:finds_none, :finite}, "Nx.add(Nx.f16([1, 2]), -6.0e4)"},
    {{:finds, {"tensor_nonfinite_result", "literal_overflow", "f16"}, :nonfinite},
     "Nx.add(Nx.f16([0.0]), 70000.0)"},
    {{:finds, {"tensor_nonfinite_result", "literal_overflow", "f16"}, :nonfinite},
     "Nx.fill(Nx.f16([1]), -1.0e9)"},
    {{:finds_none, :finite}, "Nx.fill(Nx.f16([1]), Nx.Constants.min_finite(:f16))"},
    {{:finds, {"tensor_nonfinite_result", "literal_overflow", "f16"}, :nonfinite},
     "Nx.linspace(0, 70000, n: 3, type: :f16)"},
    {{:finds, {"tensor_call_error", "literal_underflow", "1.0e-12 f16"}, :nonfinite},
     "Nx.rsqrt(Nx.add(Nx.as_type(Nx.multiply(t, t), :f16), 1.0e-12))"},
    {{:finds_none, :finite}, "Nx.rsqrt(Nx.add(Nx.as_type(Nx.multiply(t, t), :f16), 1.0e-4))"},
    # an epsilon f16 rounds to zero keeps no sum or maximum from zero
    {{:nonfinite, "divide_by_zero", "square"},
     "Nx.divide(1, Nx.add(Nx.as_type(Nx.multiply(t, t), :f16), 1.0e-12))"},
    {:finite, "Nx.divide(1, Nx.add(Nx.as_type(Nx.multiply(t, t), :f16), 1.0e-4))"},
    {{:nonfinite, "divide_by_zero", "clamp"},
     "Nx.divide(1, Nx.max(Nx.as_type(Nx.multiply(t, t), :f16), 1.0e-12))"},
    {:finite, "Nx.divide(1, Nx.max(Nx.as_type(Nx.multiply(t, t), :f16), 1.0e-4))"},
    {:finite, "Nx.divide(1, Nx.add(Nx.as_type(Nx.multiply(t, t), :f32), 1.0e-12))"},
    # f8 keeps an f16's top byte, so a number overflows and flushes in it
    # where it does written as f8 data
    {{:finds_none, :finite}, "Nx.add(Nx.tensor([1.0], type: :f8), 61440.0)"},
    {{:finds, {"tensor_nonfinite_result", "literal_overflow", "f8"}, :nonfinite},
     "Nx.add(Nx.tensor([1.0], type: :f8), 65520.0)"},
    {{:finds, {"tensor_call_error", "literal_underflow", "1.0e-5 f8"}, :finite},
     "Nx.add(Nx.tensor([0.0], type: :f8), 1.0e-5)"},
    {{:finds_none, :finite}, "Nx.add(Nx.tensor([0.0], type: :f8), 1.6e-5)"},
    {{:finds_none, :finite}, "Nx.as_type(Nx.Constants.max_finite(:f16), :f8)"},
    {{:finds_none, :finite}, "Nx.sum(Nx.broadcast(Nx.tensor(1.0, type: :f8), {62000}))"},
    {{:finds, {"tensor_nonfinite_result", "float_sum_overflow", "f8 66000"}, :nonfinite},
     "Nx.sum(Nx.broadcast(Nx.tensor(1.0, type: :f8), {66000}))"},
    # a number meeting a bf16, f32 or f16 tensor overflows and flushes where
    # it does written as data of that type
    {{:finds, {"tensor_nonfinite_result", "literal_overflow", "bf16"}, :nonfinite},
     "Nx.add(Nx.bf16([1.0]), 1.0e39)"},
    {{:finds, {"tensor_call_error", "literal_underflow", "1.0e-46 f32"}, :finite},
     "Nx.add(Nx.f32([0.0]), 1.0e-46)"},
    {{:finds, {"tensor_call_error", "literal_underflow", "2.98e-8 f16"}, :finite},
     "Nx.add(Nx.f16([0.0]), 2.98e-8)"},
    {{:finds_none, :finite}, "Nx.add(Nx.f16([0.0]), 3.0e-8)"},
    # outside traced code Nx makes a float an f32 before it meets an f64 or
    # c128 tensor, so past or below f32's range it is infinite or zero there
    {{:finds, {"tensor_nonfinite_result", "literal_overflow", "f64"}, :nonfinite},
     "Nx.add(Nx.f64([1.0]), 1.0e39)"},
    {{:finds_none, :finite}, "Nx.add(Nx.f64([1.0]), Nx.f64(1.0e39))"},
    {{:finds_none, :finite}, "Nx.Defn.jit(&Nx.add(&1, 1.0e39)).(Nx.f64([1.0]))"},
    {{:finds, {"tensor_call_error", "literal_underflow", "1.0e-50 f64"}, :finite},
     "Nx.multiply(Nx.f64([1.0]), 1.0e-50)"},
    {{:finds, {"tensor_call_error", "literal_underflow", "1.0e-46 f64"}, :finite},
     "Nx.add(Nx.f64([0.0]), 1.0e-46)"},
    {{:finds, {"tensor_nonfinite_result", "literal_overflow", "c128"}, :nonfinite},
     "Nx.add(Nx.c128([0.0]), 1.0e39)"},
    {{:finds, {"tensor_nonfinite_result", "literal_overflow", "f64"}, :nonfinite},
     "Nx.fill(Nx.f64([0.0]), 1.0e39)"},
    {{:finds, {"tensor_nonfinite_result", "literal_overflow", "f64"}, :nonfinite},
     "Nx.fill(Nx.s32([0]), 1.0e39, type: :f64)"},
    # lower-precision floats made f32 by a fixed f32
    {{:finds, {"tensor_type_error", "upcast", "bf16 f32"}, :accepted},
     "Nx.clip(Nx.bf16([1]), -1.0, 1.0)"},
    {{:finds_none, :accepted}, "Nx.clip(Nx.bf16([1]), -1, 1)"},
    {{:finds, {"tensor_type_error", "upcast", "u8 s32"}, :accepted}, "Nx.clip(Nx.u8([1]), 0, 1)"},
    {{:finds_none, :accepted}, "Nx.clip(Nx.u8([1]), Nx.u8(0), Nx.u8(1))"},
    {{:finds, {"tensor_type_error", "upcast", "bf16 f32"}, :accepted},
     "Nx.put_slice(Nx.bf16([1, 2]), [0], Nx.tensor([0.5]))"},
    {{:finds_none, :accepted}, "Nx.put_slice(Nx.bf16([1, 2]), [0], Nx.bf16([0.5]))"},
    {{:finds, {"tensor_type_error", "upcast", "bf16 f32"}, :accepted}, "Nx.log2(Nx.bf16([2]))"},
    {{:finds, {"tensor_type_error", "upcast", "bf16 f32"}, :accepted},
     "Nx.divide(Nx.bf16([1]), Nx.sqrt(64))"},
    {{:finds, {"tensor_type_error", "upcast", "bf16 f32"}, :accepted},
     "Nx.multiply(Nx.bf16([1]), Nx.Constants.pi())"},
    {{:finds_none, :accepted}, "Nx.multiply(Nx.bf16([1]), Nx.Constants.pi(:bf16))"},
    {{:finds, {"tensor_type_error", "upcast", "bf16 f32"}, :accepted},
     "Nx.add(Nx.bf16([1]), Nx.tensor(1.0))"},
    {{:finds_none, :accepted}, "Nx.add(Nx.bf16([1]), 1.0)"},
    {{:finds, {"tensor_type_error", "upcast", "bf16 f32"}, :accepted},
     "Nx.select(Nx.u8([1, 0]), Nx.bf16([1, 2]), 0.0)"},
    {{:finds_none, :accepted}, "Nx.select(Nx.u8([1, 0]), Nx.bf16([1, 2]), Nx.bf16(0))"},
    {{:finds, {"tensor_type_error", "upcast", "f16 f32"}, :accepted},
     "Nx.LinAlg.invert(Nx.f16([[1, 0], [0, 1]]))"},
    {{:finds, {"tensor_call_error", "f32_precision", "f64"}, :accepted},
     "Nx.log2(Nx.f64([2.0]))"},
    # merges into a type that holds less than an operand's
    {{:finds, {"tensor_type_error", "narrowing_merge", "bf16 f16"}, :nonfinite},
     "Nx.add(Nx.bf16([1.0e5]), Nx.f16([1]))"},
    {{:finds_none, :finite},
     "Nx.add(Nx.as_type(Nx.bf16([1.0e5]), :f32), Nx.as_type(Nx.f16([1]), :f32))"},
    {{:finds, {"tensor_type_error", "narrowing_merge", "u64 s64"}, :accepted},
     "Nx.add(Nx.u64([1]), Nx.s8([0]))"},
    {{:finds, {"tensor_type_error", "narrowing_merge", "bf16 f16"}, :nonfinite},
     "Nx.concatenate([Nx.bf16([1.0e5]), Nx.f16([1])])"},
    # a written number in a joined list, whatever else the code writes
    {{:finds, {"tensor_type_error", "upcast", "bf16 f32"}, :accepted},
     "Nx.stack([0.625, Nx.bf16(1.0)])"},
    {{:finds_none, :accepted}, "Nx.stack([Nx.bf16(0.625), Nx.bf16(1.0)])"},
    {{:finds, {"tensor_type_error", "narrowing_merge", "u64 s64"}, :accepted},
     "Nx.stack([77, Nx.u64(18446744073709551615)])"},
    {{:finds, {"tensor_type_error", "narrowing_merge", "f16 f8_e4m3fn"}, :accepted},
     "Nx.add(Nx.tensor([1], type: :f8_e4m3fn), Nx.f16([1]))"},
    {{:finds_none, :accepted}, "Nx.add(Nx.f16([1]), Nx.tensor([1], type: :f8_e4m3fn))"},
    # a pad value of another type than the tensor
    {{:finds, {"tensor_call_error", "pad_type_mismatch", "s32 f32"}, :accepted},
     "Nx.pad(Nx.tensor([1]), 0.5, [{1, 0, 0}])"},
    {{:finds_none, :accepted}, "Nx.pad(Nx.tensor([1]), 0, [{1, 0, 0}])"},
    {{:finds, {"tensor_call_error", "pad_type_mismatch", "u8 s16"}, :accepted},
     "Nx.pad(Nx.u8([1]), -1, [{1, 0, 0}])"},
    # float sums over more elements than the type's largest value
    {{:finds, {"tensor_nonfinite_result", "float_sum_overflow", "f16 100000"}, :nonfinite},
     "Nx.mean(Nx.broadcast(Nx.f16(1.0), {100000}))"},
    {{:finds_none, :finite}, "Nx.mean(Nx.broadcast(Nx.f32(1.0), {100000}))"},
    {{:finds, {"tensor_nonfinite_result", "float_sum_overflow", "f16 70000"}, :nonfinite},
     "Nx.dot(Nx.broadcast(Nx.f16(1.0), {70000}), Nx.broadcast(Nx.f16(1.0), {70000}))"},
    {{:finds, {"tensor_nonfinite_result", "float_sum_overflow", "f16 70000"}, :nonfinite},
     "Nx.sum(Nx.broadcast(Nx.f16(1.0), {2, 70000}), axes: [1])"},
    # a log-sum-exp of an unsigned tensor, a determinant of an integer one
    {{:finds, {"tensor_nonfinite_result", "unsigned_logsumexp", "u8"}, :nonfinite},
     "Nx.logsumexp(Nx.u8([1, 2, 3]))"},
    {{:finds_none, :finite}, "Nx.logsumexp(Nx.as_type(Nx.u8([1, 2, 3]), :f32))"},
    {{:finds, {"tensor_call_error", "integer_determinant", "u8"}, :finite},
     "Nx.LinAlg.determinant(Nx.u8([[1, 2], [3, 4]]))"},
    {{:finds_none, :finite}, "Nx.LinAlg.determinant(Nx.as_type(Nx.u8([[1, 2], [3, 4]]), :f32))"}
  ]
  @fixture_modules_dtypes """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.Dtypes do
    import Nx.Defn

    def mask_minus_one, do: Nx.subtract(Nx.greater(Nx.tensor([1, -1]), 0), 1)
    def mask_bias, do: Nx.tensor([1, -1]) |> Nx.greater(0) |> Nx.subtract(1) |> Nx.multiply(1.0e9)
    def negated_mask, do: Nx.negate(Nx.greater(Nx.tensor([1, -1]), 0))
    def hand_sign, do: Nx.subtract(Nx.greater(Nx.tensor([1, -1, 0]), 0), Nx.less(Nx.tensor([1, -1, 0]), 0))
    def centered_pixels, do: Nx.subtract(Nx.u8([50, 200]), 128)
    def zero_minus_one, do: Nx.subtract(Nx.u32([0]), 1)
    def mask_differences, do: Nx.diff(Nx.greater(Nx.tensor([1, 1, 0, 0, 1]), 0))
    def byte_closeness, do: Nx.all_close(Nx.u8(1), Nx.u8(2), atol: 2)
    def descending_bytes, do: Nx.linspace(10, 0, n: 3, type: :u8)
    def running_count, do: Nx.cumulative_sum(Nx.greater(Nx.iota({300}), -1))
    def window_count, do: Nx.window_sum(Nx.greater(Nx.iota({300}), -1), {300})

    def confusion do
      onehot = Nx.equal(Nx.new_axis(Nx.remainder(Nx.iota({300}), 1), -1), Nx.iota({1, 3}))
      Nx.dot(Nx.transpose(onehot), onehot)
    end

    def byte_dot, do: Nx.dot(Nx.s8([100, 100]), Nx.s8([2, 2]))
    def byte_median, do: Nx.median(Nx.u8([[130, 140], [150, 160]]))
    def byte_window_mean, do: Nx.window_mean(Nx.u8([200, 100, 50]), {2})
    def byte_weighted_mean, do: Nx.weighted_mean(Nx.u8([200, 100]), Nx.u8([2, 2]))
    def byte_argmax, do: Nx.argmax(Nx.iota({300}), type: :u8)
    def signed_byte_argmax, do: Nx.argmax(Nx.iota({200}), type: :s8)
    def bf16_positions, do: Nx.iota({1000}, type: :bf16)
    def byte_positions, do: Nx.iota({300}, type: :u8)
    def cast_to_bytes, do: Nx.as_type(Nx.tensor([300.0, -1.0, 2.7, -2.7]), :u8)
    def filled_bytes, do: Nx.fill(Nx.u8([1]), -1, type: :u8)
    def padded_with_half, do: Nx.pad(Nx.tensor([1]), 0.5, [{1, 0, 0}])
    def padded_with_minus_one, do: Nx.pad(Nx.u8([1]), -1, [{1, 0, 0}])
    def byte_determinant, do: Nx.LinAlg.determinant(Nx.u8([[1, 2], [3, 4]]))
    def unsigned_meets_signed, do: Nx.add(Nx.u64([18446744073709551615]), Nx.s8([0]))
    def log2_of_f64, do: Nx.log2(Nx.f64([2.0]))
    def log2_of_f64_in_f64, do: Nx.divide(Nx.log(Nx.f64([2.0])), Nx.log(Nx.f64(2.0)))
    def defn_bias, do: defn_bias_body(Nx.greater(Nx.tensor([1, -1]), 0))
    defn defn_bias_body(mask), do: (mask - 1) * 1.0e9
  end

  defmodule ArgusNxTensorAnalyses.TensorShapesTest.DtypesConfigured do
    def masked(x, type), do: Nx.add(Nx.as_type(x, type), -1.0e9)
    def scaled(x, type), do: Nx.multiply(Nx.as_type(x, type), Nx.Constants.pi())
    def same_type(x, y, type), do: Nx.add(Nx.as_type(x, type), Nx.as_type(y, type))
    def positions(n, type), do: Nx.multiply(Nx.iota({n}), Nx.tensor(1, type: type))
    def against_f16(x, type), do: Nx.add(Nx.as_type(x, type), Nx.f16([1]))
    def normalized(x, type), do: Nx.rsqrt(Nx.add(Nx.as_type(Nx.multiply(x, x), type), 1.0e-12))
    def written_bf16(x), do: Nx.add(Nx.as_type(x, :bf16), -1.0e9)
    def real_part(x, type), do: Nx.as_type(Nx.as_type(x, :c64), type)
    def tiny_epsilon(x, type), do: Nx.add(Nx.as_type(x, type), 1.0e-46)
  end
  """

  # ── Serving: priv/tensor_shapes/serving.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_serving [
    # a serving's output leaf without the batch as its first axis
    {{:finds, {"tensor_call_error", "serving_scalar_output", :any}, :raises},
     "Nx.Serving.run(Nx.Serving.jit(&Nx.sum/1), Nx.Batch.stack([Nx.tensor([1, 2, 3])]))"},
    {{:finds_none, :accepted},
     "Nx.Serving.run(Nx.Serving.jit(&Nx.sum(&1, axes: [1])), Nx.Batch.stack([Nx.tensor([1, 2, 3])]))"},
    {{:finds, {"tensor_call_error", "serving_scalar_output", :any}, :raises},
     "Nx.Serving.run(Nx.Serving.jit(fn x -> Nx.mean(x) end), Nx.Batch.stack([t]))"},
    {{:finds, {"tensor_call_error", "serving_scalar_output", :any}, :raises},
     "Nx.Serving.run(Nx.Serving.jit(fn x -> {x, Nx.dot(x, x)} end), Nx.Batch.stack([Nx.tensor(1.0)]))"},
    {{:finds, {"tensor_call_error", "serving_output_batch_axis", :any}, :accepted},
     "Nx.Serving.run(Nx.Serving.jit(fn x -> Nx.transpose(x) end), Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])]))"},
    {{:finds_none, :accepted},
     "Nx.Serving.run(Nx.Serving.jit(fn x -> Nx.multiply(x, 2) end), Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])]))"},
    {{:finds, {"tensor_call_error", "serving_output_batch_axis", :any}, :accepted},
     "Nx.Serving.run(Nx.Serving.jit(fn x -> Nx.sum(x, axes: [0]) end), Nx.Batch.stack([Nx.tensor([1, 2]), Nx.tensor([3, 4])]))"},
    {{:finds, {"tensor_call_error", "serving_output_batch_axis", :any}, :accepted},
     "Nx.Serving.new(fn options -> Nx.Defn.compile(fn x -> Nx.transpose(x) end, [Nx.template({4, 3}, :s32)], options) end)"},
    {{:finds_none, :accepted},
     "Nx.Serving.new(fn options -> Nx.Defn.compile(fn x -> Nx.sum(x, axes: [1]) end, [Nx.template({4, 3}, :s32)], options) end)"},
    # operations along the batch axis
    {{:finds, {"tensor_call_error", "serving_mixes_batch", :any}, :accepted},
     "Nx.Serving.jit(fn x -> Nx.subtract(x, Nx.mean(x, axes: [0])) end)"},
    {{:finds_none, :accepted},
     "Nx.Serving.jit(fn x -> Nx.subtract(x, Nx.mean(x, axes: [1], keep_axes: true)) end)"},
    {{:finds, {"tensor_call_error", "serving_mixes_batch", :any}, :accepted},
     "Nx.Serving.jit(fn x -> Nx.sum(x, axes: [0]) end)"},
    {{:finds, {"tensor_call_error", "serving_mixes_batch", :any}, :accepted},
     "Nx.Serving.run(Nx.Serving.jit(fn x -> Nx.sort(x) end), Nx.Batch.stack([Nx.tensor([3, 1]), Nx.tensor([2, 4])]))"},
    {{:finds_none, :accepted},
     "Nx.Serving.run(Nx.Serving.jit(fn x -> Nx.sort(x, axis: 1) end), Nx.Batch.stack([Nx.tensor([3, 1]), Nx.tensor([2, 4])]))"},
    {{:finds, {"tensor_call_error", "serving_mixes_batch", :any}, :accepted},
     "Nx.Serving.run(Nx.Serving.jit(fn x -> Nx.multiply(x, Nx.sum(Nx.dot(x, [0], x, [0]))) end), Nx.Batch.stack([Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])]))"},
    {{:finds_none, :accepted},
     "Nx.Serving.run(Nx.Serving.jit(fn x -> Nx.multiply(x, Nx.sum(x, axes: [1], keep_axes: true)) end), Nx.Batch.stack([Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])]))"},
    # a closure takes the batch, or the request, ahead of what it captured
    {{:finds, {"tensor_call_error", "serving_mixes_batch", :any}, :accepted},
     "scale = Nx.add(t, 1)\nNx.Serving.jit(fn x -> Nx.multiply(Nx.subtract(x, Nx.mean(x, axes: [0])), scale) end)"},
    {{:finds_none, :accepted},
     "scale = Nx.add(t, 1)\nNx.Serving.jit(fn x -> Nx.multiply(Nx.subtract(x, Nx.mean(x, axes: [1], keep_axes: true)), scale) end)"},
    {{:finds_none, :accepted},
     "offsets = Nx.iota({3, 2})\nNx.Serving.jit(fn x -> Nx.add(x, Nx.sum(offsets, axes: [0])) end)"},
    {{:finds, {"tensor_call_error", "serving_output_batch_axis", :any}, :accepted},
     "flip = config.a\nNx.Serving.run(Nx.Serving.jit(fn x -> if flip > 0, do: Nx.transpose(x), else: x end), Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])]))"},
    {{:finds_none, :accepted},
     "flip = config.a\nNx.Serving.run(Nx.Serving.jit(fn x -> if flip > 0, do: Nx.multiply(x, 2), else: x end), Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])]))"},
    {{:finds, {"tensor_call_error", "serving_entry_shape_varies", :any}, :accepted},
     "serving = Nx.Serving.jit(&Nx.multiply(&1, 2))\n|> Nx.Serving.client_preprocessing(fn input -> {Nx.Batch.stack([input]), config} end)\n[{Nx.Serving, serving: serving, name: Lint}]"},
    {{:finds_none, :accepted},
     "serving = Nx.Serving.jit(&Nx.multiply(&1, 2))\n|> Nx.Serving.client_preprocessing(fn input -> {Nx.Batch.stack([Nx.reshape(input, {3})]), config} end)\n[{Nx.Serving, serving: serving, name: Lint}]"},
    # a template compiled for another batch size or type
    {{:finds, {"tensor_call_error", "serving_template_batch_size", :any}, :raises},
     "Nx.Serving.run(Nx.Serving.new(fn options -> Nx.Defn.compile(&Nx.multiply(&1, 2), [Nx.template({4, 3}, :s32)], options) end), Nx.Batch.stack([Nx.tensor([1, 2, 3])]))"},
    {{:finds_none, :accepted},
     "Nx.Serving.run(Nx.Serving.new(fn options -> Nx.Defn.compile(&Nx.multiply(&1, 2), [Nx.template({4, 3}, :s32)], options) end), Nx.Batch.pad(Nx.Batch.stack([Nx.tensor([1, 2, 3])]), 3))"},
    {{:finds, {"tensor_call_error", "serving_template_type", :any}, :raises},
     "Nx.Serving.run(Nx.Serving.new(fn options -> Nx.Defn.compile(&Nx.multiply(&1, 2), [Nx.template({1, 3}, :f32)], options) end), Nx.Batch.stack([Nx.tensor([1, 2, 3])]))"},
    {{:finds, {"tensor_call_error", "serving_template_type", "s64 s32"}, :raises},
     "Nx.Serving.run(Nx.Serving.new(fn options -> Nx.Defn.compile(&Nx.multiply(&1, 2), [Nx.template({1, 3}, :s64)], options) end), Nx.Batch.stack([Nx.tensor([1, 2, 3])]))"},
    {{:finds_none, :accepted},
     "Nx.Serving.run(Nx.Serving.new(fn options -> Nx.Defn.compile(&Nx.multiply(&1, 2), [Nx.template({1, 3}, :s64)], options) end), Nx.Batch.stack([Nx.tensor([1, 2, 3], type: :s64)]))"},
    {{:finds, {"tensor_call_error", "serving_template_type", "f32 bf16"}, :raises},
     "Nx.Serving.run(Nx.Serving.new(fn options -> Nx.Defn.compile(&Nx.multiply(&1, 2), [Nx.template({1, 3}, :f32)], options) end), Nx.Batch.stack([Nx.bf16([1.0, 2.0, 3.0])]))"},
    {{:finds_none, :accepted},
     "Nx.Serving.run(Nx.Serving.new(fn options -> Nx.Defn.compile(&Nx.multiply(&1, 2), [Nx.template({1, 3}, :s32)], options) end), Nx.Batch.stack([Nx.tensor([1, 2, 3])]))"},
    {{:finds, {"tensor_call_error", "serving_template_batch_size", :any}, :accepted},
     "serving = Nx.Serving.new(fn options -> Nx.Defn.compile(&Nx.multiply(&1, 2), [Nx.template({4, 3}, :s32)], options) end)\n[{Nx.Serving, serving: serving, name: Lint}]"},
    # entries a batch cannot join
    {{:finds, {"tensor_call_error", "batch_incompatible_entries", :any}, :raises},
     "Nx.Batch.stack([Nx.tensor([1, 2]), Nx.tensor([1, 2, 3])])"},
    {{:finds_none, :accepted}, "Nx.Batch.stack([Nx.tensor([1, 2]), Nx.tensor([3, 4])])"},
    {{:finds, {"tensor_call_error", "batch_incompatible_entries", :any}, :raises},
     "Nx.Batch.stack([Nx.iota({2}, names: [:x]), Nx.iota({2}, names: [:y])])"},
    {{:finds_none, :accepted}, "Nx.Batch.stack([Nx.iota({2}), Nx.iota({2}, names: [:x])])"},
    {{:finds, {"tensor_call_error", "batch_incompatible_entries", :any}, :raises},
     "Nx.Batch.concatenate([Nx.iota({2, 3}), Nx.iota({1, 2})])"},
    {{:finds_none, :accepted}, "Nx.Batch.concatenate([Nx.iota({2, 3}), Nx.iota({1, 3})])"},
    {{:finds, {"tensor_call_error", "batch_scalar_entry", :any}, :raises},
     "Nx.Batch.concatenate([Nx.tensor(1)])"},
    {{:finds, {"tensor_call_error", "batch_empty", :any}, :raises},
     "Nx.Serving.run(Nx.Serving.jit(&Nx.multiply(&1, 2)), Nx.Batch.new())"},
    {{:finds_none, :accepted},
     "Nx.Serving.run(Nx.Serving.jit(&Nx.multiply(&1, 2)), Nx.Batch.stack([t]))"},
    # the serving API handed what it cannot take
    {{:finds, {"tensor_call_error", "serving_run_input", :any}, :raises},
     "Nx.Serving.run(Nx.Serving.jit(&Nx.multiply(&1, 2)), Nx.tensor([1, 2, 3]))"},
    {{:finds_none, :accepted},
     "Nx.Serving.jit(&Nx.multiply(&1, 2))\n|> Nx.Serving.client_preprocessing(fn input -> {Nx.Batch.stack([input]), :ok} end)\n|> Nx.Serving.run(Nx.tensor([1, 2, 3]))"},
    {{:finds, {"tensor_call_error", "serving_preprocessing_result", :any}, :raises},
     "Nx.Serving.jit(&Nx.multiply(&1, 2))\n|> Nx.Serving.client_preprocessing(fn input -> Nx.Batch.stack([input]) end)\n|> Nx.Serving.run(t)"},
    {{:finds, {"tensor_call_error", "serving_preprocessing_result", :any}, :raises},
     "Nx.Serving.jit(&Nx.multiply(&1, 2))\n|> Nx.Serving.client_preprocessing(fn input -> {Nx.multiply(input, 2), :ok} end)\n|> Nx.Serving.run(t)"},
    {{:finds, {"tensor_call_error", "serving_builder_result", :any}, :raises},
     "Nx.Serving.run(Nx.Serving.new(fn x -> Nx.multiply(x, 2) end), Nx.Batch.stack([t]))"},
    {{:finds_none, :accepted},
     "Nx.Serving.run(Nx.Serving.new(fn options -> Nx.Defn.jit(&Nx.multiply(&1, 2), options) end), Nx.Batch.stack([t]))"},
    {{:finds, {"tensor_call_error", "serving_computation_arity", :any}, :raises},
     "Nx.Serving.run(Nx.Serving.jit(fn x, y -> Nx.add(x, y) end), Nx.Batch.stack([t]))"},
    {{:finds_none, :accepted},
     "Nx.Serving.run(Nx.Serving.jit(fn x -> Nx.add(x, t) end), Nx.Batch.stack([t]))"},
    {{:finds, {"tensor_call_error", "serving_postprocessing_input", :any}, :raises},
     "Nx.Serving.jit(&Nx.multiply(&1, 2))\n|> Nx.Serving.client_postprocessing(fn output, _info -> Nx.argmax(output) end)\n|> Nx.Serving.run(Nx.Batch.stack([t]))"},
    {{:finds_none, :accepted},
     "Nx.Serving.jit(&Nx.multiply(&1, 2))\n|> Nx.Serving.client_postprocessing(fn {output, _metadata}, _info -> Nx.argmax(output) end)\n|> Nx.Serving.run(Nx.Batch.stack([t]))"},
    # a serving process whose entries take the request's shape
    {{:finds, {"tensor_call_error", "serving_entry_shape_varies", :any}, :accepted},
     "serving = Nx.Serving.jit(&Nx.multiply(&1, 2))\n|> Nx.Serving.client_preprocessing(fn input -> {Nx.Batch.stack([input]), :ok} end)\n[{Nx.Serving, serving: serving, name: Lint}]"},
    {{:finds_none, :accepted},
     "serving = Nx.Serving.jit(&Nx.multiply(&1, 2))\n|> Nx.Serving.client_preprocessing(fn input -> {Nx.Batch.stack([Nx.reshape(input, {3})]), :ok} end)\n[{Nx.Serving, serving: serving, name: Lint}]"},
    {{:finds_none, :accepted},
     "serving = Nx.Serving.jit(&Nx.multiply(&1, 2))\n|> Nx.Serving.client_preprocessing(fn input -> {Nx.Batch.key(Nx.Batch.stack([input]), :short), :ok} end)\n[{Nx.Serving, serving: serving, name: Lint}]"}
  ]
  @fixture_modules_serving """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.Servings do
    import Nx.Defn

    defn total(x), do: Nx.sum(x)
    defn per_entry(x), do: Nx.sum(x, axes: [1])
    defn centered(x), do: x - Nx.mean(x, axes: [0])
    defn centered_per_entry(x), do: x - Nx.mean(x, axes: [1], keep_axes: true)

    def total_serving, do: Nx.Serving.jit(&total/1)
    def per_entry_serving, do: Nx.Serving.jit(&per_entry/1)
    def centered_serving, do: Nx.Serving.jit(&centered/1)
    def centered_per_entry_serving, do: Nx.Serving.jit(&centered_per_entry/1)

    def run(serving, batch), do: Nx.Serving.run(serving, batch)
    def run_total(batch), do: run(total_serving(), batch)

    def contracting(entries) do
      serving = Nx.Serving.jit(fn x -> Nx.multiply(x, Nx.sum(Nx.dot(x, [0], x, [0]))) end)
      run(serving, Nx.Batch.stack([Nx.tensor([1.0, 2.0]) | entries]))
    end

    def contracting_known, do: contracting([Nx.tensor([3.0, 4.0])])

    def sorting(entries) do
      run(Nx.Serving.jit(&Nx.sort/1), Nx.Batch.stack([Nx.tensor([3, 1]) | entries]))
    end

    def sorting_known do
      run(Nx.Serving.jit(fn x -> Nx.sort(x) end), Nx.Batch.stack([Nx.tensor([3, 1]), Nx.tensor([2, 4])]))
    end

    def varying_serving do
      Nx.Serving.jit(&Nx.multiply(&1, 2))
      |> Nx.Serving.client_preprocessing(fn input -> {Nx.Batch.stack([input]), :ok} end)
    end

    def varying_child, do: {Nx.Serving, serving: varying_serving(), name: ServingVarying}

    def aot_serving do
      Nx.Serving.new(fn options ->
        Nx.Defn.compile(&Nx.multiply(&1, 2), [Nx.template({4, 3}, :s32)], options)
      end)
    end

    def aot_child, do: {Nx.Serving, serving: aot_serving(), name: ServingAot, batch_size: 4}

    def padded_aot_serving do
      Nx.Serving.new(fn options ->
        compiled = Nx.Defn.compile(&Nx.multiply(&1, 2), [Nx.template({4, 3}, :s32)], options)
        fn batch -> compiled.(Nx.Batch.pad(batch, 4 - batch.size)) end
      end)
    end

    def padded_aot_child,
      do: {Nx.Serving, serving: padded_aot_serving(), name: ServingPaddedAot, batch_size: 4}
  end
  """

  # ── Consumption: priv/tensor_shapes/consumption.dl ──
  # Lint cases, and modules of their own compiled with the fixtures.
  @lint_cases_consumption [
    # a key drawn from twice, and the key threaded through
    {{:finds, {"tensor_call_error", "reused_key", "Nx.Random.uniform after Nx.Random.uniform"},
      :finite},
     "key = Nx.Random.key(42)\n{a, _} = Nx.Random.uniform(key)\n{b, _} = Nx.Random.uniform(key)\nNx.subtract(a, b)"},
    {{:finds_none, :finite},
     "key = Nx.Random.key(42)\n{a, key} = Nx.Random.uniform(key)\n{b, _} = Nx.Random.uniform(key)\nNx.subtract(a, b)"},
    # two samplers on one key, as a variable or a part of a split, and on two parts
    {{:finds, {"tensor_call_error", "reused_key", "Nx.Random.normal after Nx.Random.uniform"},
      :finite},
     "key = Nx.Random.key(42)\n{a, _} = Nx.Random.uniform(key, shape: {4})\n{b, _} = Nx.Random.normal(key, shape: {4})\nNx.add(a, b)"},
    {{:finds, {"tensor_call_error", "reused_key", "Nx.Random.normal after Nx.Random.uniform"},
      :finite},
     "keys = Nx.Random.split(Nx.Random.key(42))\n{a, _} = Nx.Random.uniform(keys[0], shape: {4})\n{b, _} = Nx.Random.normal(keys[0], shape: {4})\nNx.add(a, b)"},
    {{:finds_none, :finite},
     "keys = Nx.Random.split(Nx.Random.key(42))\n{a, _} = Nx.Random.uniform(keys[0], shape: {4})\n{b, _} = Nx.Random.normal(keys[1], shape: {4})\nNx.add(a, b)"},
    # a key split, then drawn from itself
    {{:finds, {"tensor_call_error", "reused_key", "Nx.Random.uniform after Nx.Random.split"},
      :finite},
     "key = Nx.Random.key(42)\nkeys = Nx.Random.split(key)\n{a, _} = Nx.Random.uniform(key)\n{a, keys}"},
    # draws on two branches draw once on any path
    {{:finds_none, :finite},
     "key = Nx.Random.key(42)\nif config.heads > 0, do: elem(Nx.Random.uniform(key), 0), else: elem(Nx.Random.normal(key), 0)"},
    # a helper that draws from the key it is handed, called twice
    {{:finds, {"tensor_call_error", "reused_key", :any}, :finite},
     "key = Nx.Random.key(42)\nNx.subtract(ArgusNxTensorAnalyses.TensorShapesTest.Consumption.sample(key), ArgusNxTensorAnalyses.TensorShapesTest.Consumption.sample(key))"},
    # a loop's fun that captures the key, and one that folds in the pass's index
    {{:finds, {"tensor_call_error", "captured_loop_key", "Nx.Random.uniform"}, :finite},
     "key = Nx.Random.key(42)\nfor _ <- 1..3, do: elem(Nx.Random.uniform(key), 0)"},
    {{:finds_none, :finite},
     "key = Nx.Random.key(42)\nfor index <- 1..3, do: elem(Nx.Random.uniform(Nx.Random.fold_in(key, index)), 0)"},
    # a loop that hands its next pass the key it drew from, and one that hands the new key
    {{:finds, {"tensor_call_error", "passed_back_key", "Nx.Random.uniform"}, :finite},
     "{rows, _key} = Enum.map_reduce(1..3, Nx.Random.key(42), fn _, key -> {elem(Nx.Random.uniform(key), 0), key} end)\nrows"},
    {{:finds_none, :finite},
     "{rows, _key} = Enum.map_reduce(1..3, Nx.Random.key(42), fn _, key -> Nx.Random.uniform(key) end)\nrows"},
    # a function that returns the key it drew from, and one that returns the new key
    {{:finds, {"tensor_call_error", "spent_key_returned", "Nx.Random.uniform"}, :finite},
     "key = Nx.Random.key(42)\n{sample, _} = Nx.Random.uniform(key)\n{sample, key}"},
    {{:finds_none, :finite},
     "key = Nx.Random.key(42)\n{sample, key} = Nx.Random.uniform(key)\n{sample, key}"},
    # a tensor read after a transfer freed it, and the tensor the transfer returns
    {{:finds, {"tensor_call_error", "used_after_transfer", "read"}, :accepted},
     "x = Nx.iota({3})\n_ = Nx.backend_transfer(x)\nNx.add(x, 1)"},
    {{:finds_none, :finite}, "x = Nx.iota({3})\nx = Nx.backend_transfer(x)\nNx.add(x, 1)"},
    {{:finds, {"tensor_call_error", "used_after_transfer", "read"}, :accepted},
     "_ = inspect(Nx.backend_transfer(t))\nNx.multiply(t, 2)"},
    {{:finds, {"tensor_call_error", "used_after_transfer", "read"}, :accepted},
     "_ = Nx.backend_transfer(t, Nx.BinaryBackend)\nNx.sum(t)"},
    {{:finds, {"tensor_call_error", "used_after_transfer", "returned"}, :accepted},
     "_ = Nx.backend_transfer(t)\nt"},
    # its shape is in the struct, which the transfer leaves
    {{:finds_none, :finite}, "_ = Nx.backend_transfer(t)\nNx.shape(t)"},
    # a tensor read after it is deallocated, and read before
    {{:finds, {"tensor_call_error", "used_after_deallocation", "read"}, :accepted},
     "x = Nx.iota({3})\nNx.backend_deallocate(x)\nNx.sum(x)"},
    {{:finds_none, :finite},
     "x = Nx.iota({3})\ntotal = Nx.sum(x)\nNx.backend_deallocate(x)\ntotal"},
    # a tensor read after a JIT call it was donated to, and what the call returns
    {{:finds, {"tensor_call_error", "used_after_donation", "read"}, :accepted},
     "doubled = Nx.Defn.jit(&Nx.multiply(&1, 2))\n_ = doubled.(Nx.donatable(t))\nNx.add(t, 1)"},
    {{:finds_none, :finite},
     "doubled = Nx.Defn.jit(&Nx.multiply(&1, 2))\nt = doubled.(Nx.donatable(t))\nNx.add(t, 1)"}
  ]
  @fixture_modules_consumption """
  defmodule ArgusNxTensorAnalyses.TensorShapesTest.Consumption do
    import Nx.Defn

    def drawn_twice(key) do
      {first, _key} = Nx.Random.uniform(key, shape: {4})
      {second, _key} = Nx.Random.uniform(key, shape: {4})
      {first, second}
    end

    def drawn_threaded(key) do
      {first, key} = Nx.Random.uniform(key, shape: {4})
      {second, _key} = Nx.Random.uniform(key, shape: {4})
      {first, second}
    end

    def normal_after_uniform(key) do
      {uniform, _key} = Nx.Random.uniform(key, shape: {8})
      {normal, _key} = Nx.Random.normal(key, shape: {8})
      {uniform, normal}
    end

    def sample(key), do: elem(Nx.Random.uniform(key, shape: {4}), 0)

    def sampled_twice(key), do: {sample(key), sample(key)}

    def field_drawn_twice(state) do
      {first, _key} = Nx.Random.uniform(state.key, shape: {4})
      {second, _key} = Nx.Random.uniform(state.key, shape: {4})
      {first, second}
    end

    def recursive_rows(_key, 0), do: []

    def recursive_rows(key, count),
      do: [elem(Nx.Random.uniform(key, shape: {2}), 0) | recursive_rows(key, count - 1)]

    def recursive_threaded_rows(_key, 0), do: []

    def recursive_threaded_rows(key, count) do
      {row, key} = Nx.Random.uniform(key, shape: {2})
      [row | recursive_threaded_rows(key, count - 1)]
    end

    def captured_rows(key), do: for(_ <- 1..3, do: elem(Nx.Random.uniform(key, shape: {2}), 0))

    def folded_rows(key) do
      for index <- 1..3,
          do: elem(Nx.Random.uniform(Nx.Random.fold_in(key, index), shape: {2}), 0)
    end

    def passed_back_rows(key) do
      {rows, _key} =
        Enum.map_reduce(1..3, key, fn _, key ->
          {elem(Nx.Random.uniform(key, shape: {2}), 0), key}
        end)

      rows
    end

    def threaded_rows(key) do
      {rows, _key} = Enum.map_reduce(1..3, key, fn _, key -> Nx.Random.uniform(key, shape: {2}) end)
      rows
    end

    defn while_rows(key) do
      {_, rows, _key} =
        while {i = 0, rows = Nx.broadcast(0.0, {3, 2}), key}, i < 3 do
          {row, _next} = Nx.Random.uniform(key, shape: {1, 2})
          {i + 1, Nx.put_slice(rows, [i, 0], row), key}
        end

      rows
    end

    defn while_threaded_rows(key) do
      {_, rows, _key} =
        while {i = 0, rows = Nx.broadcast(0.0, {3, 2}), key}, i < 3 do
          {row, key} = Nx.Random.uniform(key, shape: {1, 2})
          {i + 1, Nx.put_slice(rows, [i, 0], row), key}
        end

      rows
    end

    defn drawn_twice_in_defn(key) do
      {first, _key} = Nx.Random.uniform(key, shape: {4})
      {second, _key} = Nx.Random.uniform(key, shape: {4})
      {first, second}
    end

    defn drawn_threaded_in_defn(key) do
      {first, key} = Nx.Random.uniform(key, shape: {4})
      {second, _key} = Nx.Random.uniform(key, shape: {4})
      {first, second}
    end

    def spent(key) do
      {sample, _key} = Nx.Random.uniform(key, shape: {4})
      {sample, key}
    end

    def spent_in_state(state) do
      {sample, _key} = Nx.Random.uniform(state.key, shape: {4})
      %{state | sample: sample}
    end

    def threaded_state(state) do
      {sample, key} = Nx.Random.uniform(state.key, shape: {4})
      %{state | sample: sample, key: key}
    end

    def sample_reply(state) do
      {sample, _key} = Nx.Random.uniform(state.key, shape: {4})
      {:reply, sample, state}
    end

    def host_reply(state) do
      host = Nx.backend_transfer(state.tensor)
      {:reply, host, state}
    end

    def host_reply_updated(state) do
      host = Nx.backend_transfer(state.tensor)
      {:reply, host, %{state | tensor: host}}
    end

    def transfer_then_helper(tensor) do
      _ = Nx.backend_transfer(tensor)
      doubled(tensor)
    end

    defp doubled(tensor), do: Nx.multiply(tensor, 2)

    defn tripled(tensor), do: tensor * 3

    def deallocate_then_defn(tensor) do
      Nx.backend_deallocate(tensor)
      tripled(tensor)
    end

    def transfer_container(first, second) do
      _ = Nx.backend_transfer({first, second})
      Nx.add(second, 1)
    end
  end
  """

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
    %{directory: directory, source: source, beams: beams} =
      compile_fixtures("tensor_shapes", fixture_source())

    probe = Path.join(directory, "probe.dl")
    File.write!(probe, probe_program())

    configured =
      Path.join(directory, "Elixir.ArgusNxTensorAnalyses.TensorShapesTest.DtypesConfigured.beam")

    solved =
      solve_concurrently("tensor_shapes",
        rows: &TensorShapes.solve(beams, probe, unsupported_types: [:f64], cache: &1),
        placed: &TensorShapes.run(beams, cache: &1),
        float_rows:
          &TensorShapes.solve([configured], TensorShapes.rules_file(),
            float_types: [:f16, :bf16, :f32],
            cache: &1
          )
      )

    Map.put(solved, :source, source)
  end

  for {spec, index} <- Enum.with_index(@cases, 1) do
    {expectation, body} =
      case spec do
        {expectation, body} -> {expectation, body}
        body -> {:agrees, body}
      end

    test "#{index}: #{String.replace(body, "\n", "; ")}", %{rows: rows} do
      index = unquote(index)
      function = function_id(@fixtures, "case_#{index}", 0)

      # the findings in the case's function, and in the functions it calls
      # with the shapes it gives them
      found =
        rows
        |> reached_findings("tensor_shape_mismatch", function, [
          :operation,
          :kind,
          :detail,
          :certainty
        ])
        |> Enum.uniq()

      assert_agrees(unquote(expectation), run_case(index), returned_shapes(rows, function), found)
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

          assert {:returns, _value} = lint_outcome(index)

        {:misaligned, kind} ->
          assert [_ | _] = for({"tensor_axis_misalignment", ^kind, _detail} <- found, do: kind),
                 "expected a #{kind} misalignment, found #{inspect(found)}"

          refute Enum.any?(found, &match?({"tensor_shape_mismatch", _, _}, &1)), inspect(found)
          assert {:returns, _value} = lint_outcome(index)

        {:mismatch, kind} ->
          assert [_ | _] = for({"tensor_shape_mismatch", ^kind, _detail} <- found, do: kind),
                 "expected a #{kind} mismatch, found #{inspect(found)}"

        {:nonfinite, kind, cause} ->
          assert_finding(found, {"tensor_nonfinite_result", kind, cause})
          assert classify_outcome(lint_outcome(index), :nonfinite) == :nonfinite

        {:hazard, kind, cause} ->
          assert_finding(found, {"tensor_nonfinite_result", kind, cause})
          assert {:returns, _value} = lint_outcome(index)

        {:unchecked, kind, cause} ->
          assert_finding(found, {"tensor_nonfinite_result", kind, cause})

          refute Enum.any?(found, fn {_relation, found_kind, _cause} ->
                   not String.starts_with?(found_kind, "unchecked_")
                 end),
                 "expected no definite finding, found #{inspect(found)}"

          assert {:returns, _value} = lint_outcome(index)

        {:type_error, kind, subject} ->
          assert_finding(found, {"tensor_type_error", kind, subject})
          assert {:raises, %ArgumentError{}} = lint_outcome(index)

        {:unsupported, type} ->
          assert_finding(found, {"tensor_type_error", "unsupported_type", type})
          assert {:returns, _value} = lint_outcome(index)

        :finite ->
          assert found == [],
                 "the math keeps the result finite, and the analysis finds #{inspect(found)}"

          assert classify_outcome(lint_outcome(index), :nonfinite) == :finite

        {:finds, {relation, kind, subject}, outcome} ->
          assert Enum.any?(found, fn {found_relation, found_kind, found_subject} ->
                   found_relation == relation and found_kind == kind and
                     subject_matches?(subject, found_subject)
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
    function = function_id(@lint_fixtures, :divided, 2)
    kinds = rows |> findings_for("tensor_nonfinite_result", function, :kind) |> Enum.uniq()

    assert kinds == ["divide_by_zero"]
  end

  # `scaled_by/2` divides by what its callers compute, an absolute value,
  # and `shrunk_by/2` by one they keep positive.
  test "a helper's parameter is what the program's calls hand it", %{rows: rows} do
    kinds = fn name ->
      rows
      |> findings_for("tensor_nonfinite_result", function_id(@lint_fixtures, name, 2), [
        :kind,
        :cause
      ])
      |> Enum.uniq()
    end

    assert kinds.(:scaled_by) == [{"divide_by_zero", "absolute"}]
    assert kinds.(:shrunk_by) == []
  end

  # `magnitude/1` is differentiated where `differentiates_magnitude/1`
  # hands a capture of it to a grad.
  test "a function a grad is handed is differentiated", %{rows: rows} do
    function = function_id(@lint_fixtures, :magnitude, 1)
    kinds = rows |> findings_for("tensor_nonfinite_result", function, :kind) |> Enum.uniq()

    assert kinds == ["infinite_gradient"]
  end

  # The lint case that samples integers of a float type it is given.
  @float_randint Enum.find_index(@lint_cases, fn {_expectation, body} ->
                   body == "Nx.Random.randint(Nx.Random.key(1), 0, 5, type: :f32)"
                 end) + 1

  # With the default options, a sampler's type is checked whether or not
  # anything else asks for the call's type.
  test "run/2 with the default options checks the type a sampler makes", %{placed: placed} do
    titles =
      for %{finding: finding} <- placed_in(placed, @lint_fixtures),
          finding.mfa == {@lint_fixtures, :"lint_#{@float_randint}", 4},
          do: finding.title

    assert "Nx.Random.randint/4 samples integers of a float type" in titles, inspect(titles)
  end

  # ── Options: tests of their own ──
  @options_fixtures ArgusNxTensorAnalyses.TensorShapesTest.Options

  # A call Nx raises for gives no shape: `Nx.sum(t, axis: 1)` is no full
  # reduction, and `Nx.transpose(t, [0, 2, 1])` no full reversal.
  test "options Nx rejects give no shape", %{rows: rows} do
    returned = &returned_shapes(rows, function_id(@options_fixtures, &1, 0))

    assert returned.(:summed_by_axis) == []
    assert returned.(:transposed_by_list) == []
    assert returned.(:summed_by_axes) == ["{2}[nil]"]
  end

  test "a shape Nx never computes meets nothing downstream", %{rows: rows} do
    function = function_id(@options_fixtures, :reshaped_after_axis, 0)

    found =
      findings_for(rows, ["tensor_shape_mismatch", "tensor_call_error"], function, [
        :relation,
        :kind
      ])

    assert found == [{"tensor_call_error", "unknown_option"}]
  end

  # `sums/2` hands on what its callers give it: `[axis: 0]` from one of
  # them, directly, and `[dim: 0]` from another through `sums_through/2`;
  # each is reported where the list is written, with the Nx call as its
  # origin. `sorts/2` gets an atom one of its callers gives. `sums_axes/2`
  # tests its argument's type before it builds the option, so the integer
  # its caller gives does not reach the call.
  test "options a caller hands down are checked at the call", %{rows: rows} do
    call_errors =
      &findings_for(rows, "tensor_call_error", function_id(@options_fixtures, &1, &2), [
        :operation,
        :kind,
        :detail,
        :origin_operation
      ])

    sums = "#{inspect(@options_fixtures)}.sums/2"
    through = "#{inspect(@options_fixtures)}.sums_through/2"

    assert call_errors.(:sums, 2) == []
    assert call_errors.(:sums_through, 2) == []
    assert call_errors.(:sums_by_axis, 1) == [{sums, "unknown_option", "Nx.sum axis", "Nx.sum/2"}]
    assert call_errors.(:sums_by_axes, 1) == []

    assert call_errors.(:sums_through_by_dim, 1) == [
             {through, "unknown_option", "Nx.sum dim", "Nx.sum/2"}
           ]

    assert call_errors.(:sums_through_by_axes, 1) == []

    assert call_errors.(:sorts, 2) == [
             {"Nx.sort/2", "option_value", "direction: :descending", ""}
           ]

    assert call_errors.(:sums_axes, 2) == []
  end

  # Nx raises for options it rejects before it makes a tensor, so the type
  # they name is not reported too. `iota_typed/0` makes an f64 tensor, a
  # type the fixtures are solved as unsupported.
  test "a call whose options Nx rejects makes no tensor of their type", %{rows: rows} do
    found =
      &findings_for(
        rows,
        ["tensor_call_error", "tensor_type_error"],
        function_id(@options_fixtures, &1, 0),
        [:relation, :kind]
      )

    assert found.(:pi_typed_by_option) == [{"tensor_call_error", "unknown_option"}]
    assert found.(:iota_typed_with_typo) == [{"tensor_call_error", "unknown_option"}]
    assert found.(:iota_typed) == [{"tensor_type_error", "unsupported_type"}]
  end

  test "a finding at the call that hands the options down names the Nx function" do
    handed =
      TensorShapes.finding(:tensor_call_error, [
        "M:g/1#1",
        "M:g/1",
        "M.sums/2",
        "unknown_option",
        "Nx.sum axis",
        "M:sums/2#2",
        "Nx.sum/2"
      ])

    assert handed.title == "M.sums/2 hands Nx.sum :axis, an option it does not take"
    assert handed.at_label == "hands Nx.sum :axis here"
    assert handed.detail =~ "Nx.sum checks every key of its options"
    assert handed.detail =~ "(:axes and :keep_axes)"
    assert handed.severity == :error

    assert handed.help == [
             "use :axes, which takes a list of axes: axes: [...] rather than axis: ..."
           ]

    assert [%{label: "raises for :axis in Nx.sum/2"}] = handed.related

    form =
      TensorShapes.finding(:tensor_call_error, [
        "M:g/1#1",
        "M:g/1",
        "M.sums/2",
        "option_form",
        "Nx.sum axes: 1",
        "M:sums/2#2",
        "Nx.sum/2"
      ])

    assert form.title == "M.sums/2 hands Nx.sum :axes as one axis rather than a list"
    assert [%{label: "raises for axes: 1 in Nx.sum/2"}] = form.related
  end

  test "a finding says what Nx takes instead" do
    unknown =
      TensorShapes.finding(:tensor_call_error, [
        "M:f/1#1",
        "M:f/1",
        "Nx.sum/2",
        "unknown_option",
        "axis",
        "",
        ""
      ])

    assert unknown.title == "Nx.sum/2 gets :axis, an option it does not take"
    assert unknown.detail =~ "(:axes and :keep_axes)"

    assert unknown.help == [
             "use :axes, which takes a list of axes: axes: [...] rather than axis: ..."
           ]

    value =
      TensorShapes.finding(:tensor_call_error, [
        "M:f/1#1",
        "M:f/1",
        "Nx.sort/2",
        "option_value",
        "direction: :descending",
        "",
        ""
      ])

    assert value.help == ["use :desc, the atom Nx has for this"]

    form =
      TensorShapes.finding(:tensor_call_error, [
        "M:f/1#1",
        "M:f/1",
        "Nx.sum/2",
        "option_form",
        "axes: 1",
        "",
        ""
      ])

    assert form.help == ["wrap the axis in a list: axes: [1]"]
  end

  # (end of Options tests)

  # ── Traced: tests of their own ──
  # `logs_loss/1`'s transform hands the loss to a helper that reads its
  # data, which raises while Nx traces the `defn`; `calls_doubled/1` calls
  # another `defn`, whose wrapper runs `jit_apply` only outside a trace,
  # and a transform that reads only a shape.
  test "a data read a defn reaches through a transform and a helper", %{rows: rows} do
    traced = ArgusNxTensorAnalyses.TensorShapesTest.TracedDefn
    module = inspect(traced)

    found =
      rows
      |> findings_for("tensor_call_error", &String.starts_with?(&1, module <> ":"), [
        :func,
        :kind,
        :detail
      ])
      |> Enum.uniq()

    assert found == [{module <> ":log_value/1", "data_read_in_trace", "defn"}]

    assert {:raises, %ArgumentError{} = error} =
             outcome_on_binary_backend(traced, :logs_loss, [Nx.iota({2})])

    assert Exception.message(error) =~ ~r/Nx.Defn.Expr/

    assert {:returns, %Nx.Tensor{}} =
             outcome_on_binary_backend(traced, :calls_doubled, [Nx.iota({2})])
  end

  # (end of Traced tests)

  # ── DefnFlow: tests of their own ──
  # Each test runs `DefnFlow`'s `defn`s on BinaryBackend for what Nx does,
  # and reads what the analysis finds in their bodies.
  @defn_flow ArgusNxTensorAnalyses.TensorShapesTest.DefnFlow

  test "a cond on a tensor that is not a scalar raises", %{rows: rows} do
    assert {"predicate_shape", "{3}"} in defn_flow_findings(rows, :positive_part, 1)
    assert defn_flow_raises?(:run_positive_part, [])

    assert defn_flow_findings(rows, :positive_all, 1) == []
    refute defn_flow_raises?(:run_positive_all, [])
  end

  test "a cond on an option the options lack raises", %{rows: rows} do
    assert {"predicate_value", "nil"} in defn_flow_findings(rows, :training_scale, 2)
    assert defn_flow_raises?(:training_scale, [Nx.iota({3})])

    assert defn_flow_findings(rows, :dropout_scale, 2) == []
    refute defn_flow_raises?(:dropout_scale, [Nx.iota({3})])
  end

  test "branches whose shapes do not broadcast raise", %{rows: rows} do
    for {name, detail} <- [
          branch_sizes: "cannot broadcast tensor of dimensions {3} to {2}",
          branch_elements: "cannot broadcast tensor of dimensions {2} to {3}"
        ] do
      assert {"branch_shapes", detail} in defn_flow_findings(rows, name, 1)
      assert defn_flow_raises?(name, [Nx.iota({3})])
    end

    for name <- [:branch_sizes_agree, :branch_scalar] do
      assert defn_flow_findings(rows, name, 1) == []
      refute defn_flow_raises?(name, [Nx.iota({3})])
    end
  end

  test "branches broadcast to a shape neither gives without a word", %{rows: rows} do
    findings = defn_flow_findings(rows, :branch_grows, 1)
    assert {"branch_broadcast", "{3} and {3, 1} broadcast to {3, 3}"} in findings

    assert {:returns, %Nx.Tensor{shape: {3, 3}}} =
             outcome_on_binary_backend(@defn_flow, :branch_grows, [Nx.iota({3})])
  end

  test "branches of different structures raise", %{rows: rows} do
    findings = defn_flow_findings(rows, :branch_forms, 1)
    assert {"branch_structure", "a tuple of 2 and a tensor"} in findings
    assert defn_flow_raises?(:run_branch_forms, [])

    assert defn_flow_findings(rows, :branch_pairs, 1) == []
    refute defn_flow_raises?(:run_branch_pairs, [])
  end

  test "a cond on a tensor with no clause that always holds raises", %{rows: rows} do
    assert {"cond_fallthrough", ""} in defn_flow_findings(rows, :sign_of, 1)
    assert defn_flow_raises?(:sign_of, [Nx.iota({3})])

    assert defn_flow_findings(rows, :sign_or_zero, 1) == []
    refute defn_flow_raises?(:sign_or_zero, [Nx.iota({3})])
  end

  test "a raise in a branch of a cond on a tensor runs whatever the data", %{rows: rows} do
    assert {"traced_raise", ""} in defn_flow_findings(rows, :checked_root, 1)
    assert defn_flow_raises?(:checked_root, [Nx.iota({3})])

    assert defn_flow_findings(rows, :runtime_checked_root, 1) == []
    refute defn_flow_raises?(:runtime_checked_root, [Nx.iota({3})])

    assert defn_flow_findings(rows, :strict_root, 2) == []
    refute defn_flow_raises?(:strict_root, [Nx.iota({3})])
  end

  test "a while body that changes its state's shape raises", %{rows: rows} do
    findings = defn_flow_findings(rows, :scalar_accumulator, 1)
    detail = "element 1 of the state starts as {} and the body makes it {3}"
    assert {"while_shape", detail} in findings
    assert defn_flow_raises?(:scalar_accumulator, [Nx.iota({3})])

    assert defn_flow_findings(rows, :vector_accumulator, 1) == []
    refute defn_flow_raises?(:vector_accumulator, [Nx.iota({3})])
  end

  test "a while body that makes an integer state a float raises", %{rows: rows} do
    detail = "element 1 of the state starts as an integer and the body makes it a float"

    for name <- [:integer_total, :integer_product] do
      assert {"while_type", detail} in defn_flow_findings(rows, name, 1)
      assert defn_flow_raises?(name, [Nx.iota({3})])
    end

    assert defn_flow_findings(rows, :float_total, 1) == []
    refute defn_flow_raises?(:float_total, [Nx.iota({3})])
  end

  test "a while on a tensor that is not a scalar raises", %{rows: rows} do
    assert {"predicate_shape", "{3}"} in defn_flow_findings(rows, :vector_condition, 1)
    assert defn_flow_raises?(:vector_condition, [Nx.iota({3})])

    assert defn_flow_findings(rows, :scalar_condition, 1) == []
    refute defn_flow_raises?(:scalar_condition, [Nx.iota({3})])
  end

  test "a while or reduce closure that captures a tensor raises", %{rows: rows} do
    for name <- [:captured_step, :captured_reduce] do
      assert {"closure_captures_tensor", ""} in defn_flow_findings(rows, name, 2)
      assert defn_flow_raises?(name, [Nx.iota({3}), Nx.iota({3})])
    end

    for name <- [:carried_step, :captured_length, :separate_reduce] do
      assert defn_flow_findings(rows, name, 2) == []
      refute defn_flow_raises?(name, [Nx.iota({3}), Nx.iota({3})])
    end
  end

  test "an operator on an atom raises", %{rows: rows} do
    assert {"atom_operand", ":train"} in defn_flow_findings(rows, :mode_scale, 2)
    assert defn_flow_raises?(:mode_scale, [Nx.iota({3})])

    assert {"atom_operand", "nil"} in defn_flow_findings(rows, :biased, 2)
    assert defn_flow_raises?(:biased, [Nx.iota({3})])

    for name <- [:mode_case, :shifted] do
      assert defn_flow_findings(rows, name, 2) == []
      refute defn_flow_raises?(name, [Nx.iota({3})])
    end
  end

  test "a public defn's argument where Nx takes an integer raises", %{rows: rows} do
    x = Nx.iota({3})

    for {name, arguments, slot} <- [
          {:iota_of, [3], "the shape"},
          {:broadcast_rows, [x, 2], "the shape"},
          {:sum_along, [Nx.iota({2, 3}), 1], "the :axes option"},
          {:slice_first, [x, 2], "the slice lengths"},
          {:top_of, [x, 2], "the :k option"},
          {:iota_in_tuple, [{x, 3}], "the shape"},
          {:range_to, [x, 2], "a range's bound"}
        ] do
      findings = defn_flow_findings(rows, name, length(arguments))
      assert {"tensor_as_integer", slot} in findings, "#{name}: #{inspect(findings)}"
      assert defn_flow_raises?(name, arguments)
    end

    for {name, arguments} <- [iota_of_option: [x], helper_three: [x], range_along: [x]] do
      assert defn_flow_findings(rows, name, length(arguments)) == [], inspect(name)
      refute defn_flow_raises?(name, arguments)
    end
  end

  # Read as sizes, `{n}` and `{m}` would be two size variables that
  # differ where they meet; `defn` makes them tensors, and Nx raises
  # before any sizes meet.
  test "a public defn's argument is no size variable", %{rows: rows} do
    function = defn_id(@defn_flow, :iota_pair, 2)

    assert findings_for(rows, "tensor_axis_misalignment", function, :func) == []
    assert {"tensor_as_integer", "the shape"} in defn_flow_findings(rows, :iota_pair, 2)
    assert defn_flow_raises?(:iota_pair, [2, 3])
  end

  test "a print whose result the defn never uses", %{rows: rows} do
    assert {"dropped_print", ""} in defn_flow_findings(rows, :printed_dropped, 1)
    printed = capture_io(fn -> refute defn_flow_raises?(:printed_dropped, [Nx.iota({3})]) end)
    assert printed == ""

    assert defn_flow_findings(rows, :printed_kept, 1) == []
    printed = capture_io(fn -> refute defn_flow_raises?(:printed_kept, [Nx.iota({3})]) end)
    assert printed =~ "s32[3]"
  end

  test "elem/2 and print_value/3 give the shapes they are handed", %{rows: rows} do
    for name <- [:second_wrong, :watched_wrong] do
      findings = defn_flow_findings(rows, name, 1)
      assert {"broadcast", "cannot broadcast tensor of dimensions {3} to {2}"} in findings
      assert defn_flow_raises?(name, [Nx.iota({3})])
    end

    assert defn_flow_findings(rows, :second_right, 1) == []
    refute defn_flow_raises?(:second_right, [Nx.iota({3})])
  end

  # The call to `defn_cond/2` carries no line, so a finding about a `cond`
  # is placed at its clause.
  test "run/2 places a cond finding at its clause", %{placed: placed, source: source} do
    title = "An `if` or `cond` has no clause that always holds"

    assert [line] =
             for(
               %{finding: %{title: ^title}, line: line} <- placed_in(placed, @defn_flow),
               do: line
             )

    assert source |> source_line(line) |> String.trim() == "Nx.all(x <= 0) -> -1"
  end

  # The call errors and shape mismatches in a `defn`'s body, as `{kind,
  # detail}`.
  defp defn_flow_findings(rows, name, arity) do
    rows
    |> findings_for(
      ["tensor_call_error", "tensor_shape_mismatch"],
      defn_id(@defn_flow, name, arity),
      [:kind, :detail]
    )
    |> Enum.uniq()
  end

  defp defn_flow_raises?(name, arguments),
    do: match?({:raises, _error}, outcome_on_binary_backend(@defn_flow, name, arguments))

  # (end of DefnFlow tests)

  # ── Containers: tests of their own ──
  @containers_prefix "ArgusNxTensorAnalyses.TensorShapesTest"
  @containers_defns "#{@containers_prefix}.ContainerDefns"
  @containers_use "#{@containers_prefix}.ContainerUse"

  # Nx dispatches to a struct's `Nx.Container` implementation through the
  # protocol consolidated at compile time, which knows none of the fixtures'
  # structs, so these tests consolidate it again with them before running
  # Nx.
  describe "containers" do
    setup %{source: source} do
      containers_consolidate(Path.dirname(source))
    end

    # `causal` and `heads` are in neither `containers:` nor `keep:` of
    # `DroppingLayer`, and in `keep:` of `KeepingLayer`.
    test "a defn reads a field its struct's container drops", %{rows: rows} do
      findings =
        containers_findings(rows, "#{@containers_defns}:__defn:branch__/2", "dropped_field_read")

      assert findings == ["#{@containers_prefix}.DroppingLayer.causal=false"]

      tensor = Nx.tensor([1])
      assert containers_run(:dropping_branch, [tensor]) == {:returns, Nx.tensor([1])}
      assert containers_run(:keeping_branch, [tensor]) == {:returns, Nx.tensor([100])}

      findings =
        containers_findings(
          rows,
          "#{@containers_defns}:__defn:by_heads__/2",
          "dropped_field_read"
        )

      assert findings == ["#{@containers_prefix}.DroppingLayer.heads=nil"]
      assert {:raises, %ArithmeticError{}} = containers_run(:dropping_heads, [tensor])
      assert {:returns, %Nx.Tensor{shape: {2, 2}}} = containers_run(:keeping_heads, [tensor])
    end

    # A struct a defn builds and hands another is not traversed.
    test "a defn reads a field of a struct a defn built", %{rows: rows} do
      assert containers_findings(
               rows,
               "#{@containers_defns}:__defn:branch_inside__/2",
               "dropped_field_read"
             ) ==
               []

      assert containers_run(:built_inside, [Nx.tensor([1])]) == {:returns, Nx.tensor([100])}
    end

    # A wrapper called while a defn is traced runs its body directly, over
    # what it is handed as it is.
    test "a function a defn calls hands a defn a container", %{rows: rows} do
      assert containers_findings(
               rows,
               "#{@containers_defns}:through_transform/1",
               "container_leaf"
             ) ==
               []

      assert containers_run(:traced_caller, [Nx.tensor([1])]) == {:returns, Nx.tensor([1])}
    end

    test "a function jit runs reads a field its struct's container drops", %{rows: rows} do
      findings =
        containers_findings(
          rows,
          "#{@containers_defns}:__defn:heads_of__/2",
          "dropped_field_read"
        )

      assert findings == ["#{@containers_prefix}.DroppingLayer.heads=nil"]

      tensor = Nx.tensor([1])
      assert {:raises, %ArithmeticError{}} = containers_run(:dropping_jitted, [tensor])
      assert {:returns, %Nx.Tensor{shape: {2, 2}}} = containers_run(:keeping_jitted, [tensor])
    end

    test "code reads a field its struct's container drops of what a defn or jit returns", %{
      rows: rows
    } do
      assert containers_findings(
               rows,
               "#{@containers_use}:dropping_returned/1",
               "dropped_field_returned"
             ) ==
               ["#{@containers_prefix}.DroppingLayer.causal=false"]

      assert containers_findings(
               rows,
               "#{@containers_use}:keeping_returned/1",
               "dropped_field_returned"
             ) ==
               []

      assert containers_findings(
               rows,
               "#{@containers_use}:dropping_jit_returned/1",
               "dropped_field_returned"
             ) == ["#{@containers_prefix}.DroppingLayer.heads=nil"]

      assert containers_findings(
               rows,
               "#{@containers_use}:keeping_jit_returned/1",
               "dropped_field_returned"
             ) ==
               []

      tensor = Nx.tensor([1])
      assert containers_run(:dropping_returned, [tensor]) == {:returns, false}
      assert containers_run(:keeping_returned, [tensor]) == {:returns, true}
      assert containers_run(:dropping_jit_returned, [tensor]) == {:returns, nil}
      assert containers_run(:keeping_jit_returned, [tensor]) == {:returns, 2}
    end

    test "a struct's container field left at a nil default is handed to defn", %{rows: rows} do
      assert containers_findings(rows, "#{@containers_use}:unset_bias/1", "container_leaf") ==
               ["nil at argument 1.bias"]

      assert containers_findings(rows, "#{@containers_use}:set_bias/1", "container_leaf") == []

      tensor = Nx.tensor([1])
      assert {:raises, %Protocol.UndefinedError{}} = containers_run(:unset_bias, [tensor])
      assert {:returns, %{bias: %Nx.Tensor{}}} = containers_run(:set_bias, [tensor])
    end

    test "a struct with no container implementation is handed to defn", %{rows: rows} do
      assert containers_findings(rows, "#{@containers_use}:plain_settings/1", "container_leaf") ==
               ["a #{@containers_prefix}.PlainSettings struct at argument 1{1}"]

      assert {:raises, %Protocol.UndefinedError{}} =
               containers_run(:plain_settings, [Nx.tensor([1])])
    end

    test "an implementation's traverse/3 and reduce/3 visit fields in two orders", %{rows: rows} do
      assert containers_findings(
               rows,
               "Nx.Container.#{@containers_prefix}.MisorderedPair:traverse/3",
               "container_order"
             ) == [
               "#{@containers_prefix}.MisorderedPair.first",
               "#{@containers_prefix}.MisorderedPair.second"
             ]

      assert containers_findings(
               rows,
               "Nx.Container.#{@containers_prefix}.OrderedPair:traverse/3",
               "container_order"
             ) == []

      tensor = Nx.tensor([1.0, 2.0, 3.0])
      assert {:returns, misordered} = containers_run(:misordered_loop, [tensor])
      refute Nx.shape(misordered.first) == {3}
      assert {:returns, ordered} = containers_run(:ordered_loop, [tensor])
      assert ordered.first == tensor
    end
  end

  # The details of the call errors of a kind in a function.
  defp containers_findings(rows, function, kind) do
    for {^kind, detail} <- findings_for(rows, "tensor_call_error", function, [:kind, :detail]),
        uniq: true,
        do: detail
  end

  defp containers_run(name, arguments),
    do:
      outcome_on_binary_backend(Module.concat(@containers_prefix, ContainerUse), name, arguments)

  # Consolidates `Nx.Container` again with the implementations compiled into
  # `directory`, once.
  defp containers_consolidate(directory) do
    layer = struct(Module.concat(@containers_prefix, DroppingLayer))

    if Nx.Container.impl_for(layer) == nil do
      implementations = Protocol.extract_impls(Nx.Container, [directory | :code.get_path()])
      {:ok, binary} = Protocol.consolidate(Nx.Container, implementations)
      :code.purge(Nx.Container)
      {:module, Nx.Container} = :code.load_binary(Nx.Container, ~c"consolidated", binary)
    end

    :ok
  end

  # (end of Containers tests)

  # ── Gradients: tests of their own ──
  @gradients_fixtures ArgusNxTensorAnalyses.TensorShapesTest.GradientsFixtures

  # The kinds of the results that can be infinite or NaN in a function of
  # the gradients' fixtures.
  defp gradient_kinds(rows, function) do
    rows
    |> findings_for(
      "tensor_nonfinite_result",
      "#{inspect(@gradients_fixtures)}:#{function}",
      :kind
    )
    |> Enum.uniq()
  end

  # Whether the gradient a fixture computes at zero holds an infinity or a
  # NaN.
  defp gradient_nonfinite?(name) do
    {:returns, gradient} =
      outcome_on_binary_backend(@gradients_fixtures, name, [Nx.tensor([0.0])])

    nonfinite?(gradient)
  end

  # `safe_norm/1` wraps its norm in a custom_grad, as the usual fix does.
  test "a custom_grad replaces the gradient of the norm it wraps", %{rows: rows} do
    assert gradient_kinds(rows, "__defn:safe_norm__/1") == []
    assert gradient_kinds(rows, "__defn:plain_norm__/1") == ["infinite_gradient"]
    refute gradient_nonfinite?(:differentiates_safe_norm)
    assert gradient_nonfinite?(:differentiates_plain_norm)
  end

  test "an exponent an option or a quotient of written numbers holds is a root", %{rows: rows} do
    assert "infinite_gradient" in gradient_kinds(rows, "__defn:root_of__/2")
    assert gradient_kinds(rows, "__defn:cube_root__/1") == ["infinite_gradient"]
    assert gradient_kinds(rows, "__defn:cube__/1") == []
    assert gradient_nonfinite?(:differentiates_root_option)
    assert gradient_nonfinite?(:differentiates_cube_root)
    refute gradient_nonfinite?(:differentiates_cube)
  end

  test "a defn's select of a logarithm its predicate guards is a double where", %{rows: rows} do
    assert gradient_kinds(rows, "-__defn:masked_log__/1-fun-0-/1") == ["masked_gradient"]
    assert gradient_nonfinite?(:masked_log)
  end

  test "every kind the gradient rules report has wording of its own" do
    alias ArgusNxTensorAnalyses.TensorShapes.Wording

    hazards = [
      {"infinite_gradient", "spread", "Nx.standard_deviation/1"},
      {"gradient_at_origin", "square", "Nx.atan2/2"},
      {"gradient_at_origin", "input", "Nx.atan2/2"},
      {"gradient_at_edge", "clip", "Nx.acos/1"},
      {"gradient_at_edge", "square", "Nx.acosh/1"},
      {"masked_gradient", "logarithm", "Nx.log/1"},
      {"exponent_gradient", "written", "Nx.pow/2"},
      {"sigmoid_gradient_overflow", "half_precision", "Nx.sigmoid/1"},
      {"degenerate_gradient", "rank_one", "Nx.LinAlg.pinv/1"}
    ]

    call_errors = [
      {"singular_gradient", "low_rank", "Nx.LinAlg.cholesky/1"},
      {"no_gradient", "", "Nx.reduce/4"},
      {"complex_gradient", "", "Nx.rfft/1"},
      {"custom_grad_not_list", "", "Nx.Defn.Kernel.custom_grad/3"},
      {"custom_grad_short", "1 of 2", "Nx.Defn.Kernel.custom_grad/3"},
      {"custom_grad_unlisted", "", "Nx.Defn.Kernel.custom_grad/3"},
      {"gradient_of_container", "tuple", "Nx.Defn.grad/2"}
    ]

    for {kind, cause, operation} <- hazards do
      assert %{title: _, detail: _, label: _, help: _, frame: _} =
               Wording.hazard(kind, cause, operation)
    end

    for {kind, detail, operation} <- call_errors do
      assert %{title: _, detail: _, label: _, help: _, frame: _, severity: _} =
               Wording.call_error(kind, detail, operation)
    end

    # a variance is left to the core's wording, a standard deviation is not
    assert Wording.Gradients.hazard("infinite_gradient", "spread", "Nx.variance/1") == nil

    assert %{title: "has an infinite gradient where its result is zero"} =
             Wording.hazard("infinite_gradient", "spread", "Nx.variance/1")
  end

  # (end of Gradients tests)

  # ── Math: tests of their own ──

  # Inside a `defn` the same math compiles to `Nx.Defn.Kernel`'s operators.
  test "a softplus or logistic written out in a defn can overflow", %{rows: rows} do
    kinds = fn name ->
      rows
      |> findings_for(
        "tensor_nonfinite_result",
        defn_id(ArgusNxTensorAnalyses.TensorShapesTest.MathDefn, name, 1),
        [:kind, :cause]
      )
      |> Enum.uniq()
    end

    assert kinds.(:softplus) == [{"exp_overflow", "softplus"}]
    assert kinds.(:logistic) == [{"exp_overflow", "logistic"}]
    assert kinds.(:stable_softplus) == []
  end

  # Inside a `defn`, a sampler compiles to `Nx.Defn.Compiler.__remote__/4`,
  # which hands it its bounds in a list, a literal where they are written.
  test "a sample drawn in a defn has the signs its bounds give it", %{rows: rows} do
    math_defn = ArgusNxTensorAnalyses.TensorShapesTest.MathDefn

    findings = fn name ->
      rows
      |> findings_for("tensor_nonfinite_result", defn_id(math_defn, name, 1), [
        :kind,
        :cause,
        :origin_operation
      ])
      |> Enum.uniq()
      |> Enum.sort()
    end

    assert findings.(:log_of_sample) == [{"log_of_zero", "sample", "Nx.Random.uniform/2"}]

    assert findings.(:log_of_centered_sample) == [
             {"unchecked_logarithm", "cancel", "Nx.Random.uniform/4"},
             {"unchecked_logarithm", "negative", ""}
           ]

    assert findings.(:log_of_shifted_sample) == []

    key = Nx.Random.key(42)

    assert {:returns, centered} =
             outcome_on_binary_backend(math_defn, :log_of_centered_sample, [key])

    assert nonfinite?(centered)

    assert {:returns, shifted} =
             outcome_on_binary_backend(math_defn, :log_of_shifted_sample, [key])

    refute nonfinite?(shifted)
  end

  # (end of Math tests)

  # ── Indices: tests of their own ──
  # A negative index names the call whose math makes it so: the select that
  # writes an ignored label, and the remainder in another module's `defn`.
  test "a negative index names the call that takes it below zero", %{placed: placed} do
    labels = fn title, cause ->
      for %{finding: finding} <- placed,
          finding.title == title,
          finding.detail =~ cause,
          frame <- finding.related,
          do: {finding.severity, frame.label}
    end

    assert {:warning, "the index can go negative because of this Nx.select/3"} in labels.(
             "Nx.take_along_axis/3 can get a negative index",
             "a written negative number"
           )

    assert {:warning, "the index can go negative because of this Nx.Defn.Kernel.rem/2"} in labels.(
             "Nx.take/2 can get a negative index",
             "a remainder"
           )
  end

  # Inside a `defn`, `Nx.Random.randint` compiles to
  # `Nx.Defn.Compiler.__remote__/4`, which hands it its written bounds and
  # options in the one literal the compiler folds them into.
  test "a random range written in a defn is checked as one written outside", %{rows: rows} do
    helpers = ArgusNxTensorAnalyses.TensorShapesTest.IndicesHelpers

    findings =
      &findings_for(rows, "tensor_call_error", defn_id(helpers, &1, 1), [
        :operation,
        :kind,
        :detail
      ])

    assert findings.(:empty_range) == [{"Nx.Random.randint/3", "random_range_empty", "5 to 5"}]

    assert findings.(:narrow_range) == [
             {"Nx.Random.randint/4", "random_range_outside_type", "0 to 300 as u8"}
           ]

    assert findings.(:fitting_range) == []

    key = Nx.Random.key(42)
    assert {:raises, _error} = outcome_on_binary_backend(helpers, :empty_range, [key])
    assert {:returns, _sample} = outcome_on_binary_backend(helpers, :narrow_range, [key])
    assert {:returns, _sample} = outcome_on_binary_backend(helpers, :fitting_range, [key])
  end

  # (end of Indices tests)

  # ── Literals: tests of their own ──

  # Each call hands a function of `Nx.Type` a short atom type; Nx raising
  # for it is "atom_type_rejected", and an answer other than the one for
  # the type as a tuple (taken as a type, where it is one) is
  # "atom_type_misread".
  test "a short atom type is reported where Nx.Type raises or answers otherwise", %{rows: rows} do
    prefix = "#{inspect(@literal_type_fixtures)}:"

    found =
      rows
      |> findings_for("tensor_type_error", &String.starts_with?(&1, prefix), [:func, :kind])
      |> Map.new()

    disagreements =
      for {{function, arguments}, index} <- @literal_type_calls,
          expected = atom_type_outcome(function, arguments),
          reported = Map.get(found, "#{prefix}call_#{index}/0"),
          reported != expected,
          do: {function, arguments, expected, reported}

    assert disagreements == []
  end

  # `floating?/1` hands `Nx.Type.float?/1` the atom its caller writes,
  # `valid_config?/1` and `state_type/1` the field of a map their callers
  # build, `masks_through_helper/1` masks with the integer `wide_mask/0`
  # returns, and a `defn` adds an integer literal.
  test "a literal is followed to the call it misleads", %{rows: rows} do
    prefix = "#{inspect(@literal_fixtures)}:"

    found =
      for {function, operation, kind, subject} <-
            findings_for(rows, "tensor_type_error", &String.starts_with?(&1, prefix), [
              :func,
              :operation,
              :kind,
              :subject
            ]),
          do: {String.replace_prefix(function, prefix, ""), operation, kind, subject}

    assert {"floating?/1", "Nx.Type.float?/1", "atom_type_rejected", ":f32"} in found
    assert {"valid_config?/1", "Nx.Type.float?/1", "atom_type_rejected", ":f32"} in found
    assert {"state_type/1", "Nx.Type.merge/2", "atom_type_rejected", ":bf16"} in found

    assert {"masks_through_helper/1", "Nx.bitwise_and/2", "integer_past_s32",
            "1099511627775 as s32"} in found

    assert {"__defn:offset_past_s32__/1", "Nx.Defn.Kernel.+/2", "integer_past_s32",
            "3000000000 as s32"} in found
  end

  # A Kernel operator on two numbers computes a number, as Kernel's own
  # does, and Nx makes no tensor of them: the index is 1.
  test "a wide integer a defn computes with a number is no tensor", %{rows: rows} do
    function = function_id(@literal_fixtures, "__defn:picks_past_s32__", 1)

    assert findings_for(rows, "tensor_type_error", function, :kind) == []

    assert {:returns, value} =
             outcome_on_binary_backend(@literal_fixtures, :picks_past_s32, [Nx.iota({4})])

    assert Nx.to_number(value) == 1
  end

  # What Nx makes of each literal these findings name, on the binary
  # backend, is what their wording says it becomes.
  test "a literal a type cannot hold becomes the value its finding names" do
    for {kind, subject, operation, make} <- [
          {"integer_past_s32", "1700000000000 as s32", "Nx.add/2",
           fn -> Nx.add(Nx.tensor(0, type: :s64), 1_700_000_000_000) end},
          {"integer_past_s32", "3000000000 as s32", "Nx.Defn.Kernel.+/2",
           fn -> apply(@literal_fixtures, :offset_past_s32, [Nx.tensor(0, type: :s64)]) end},
          {"integer_past_s32", "3000000000 as s32", "Nx.tensor/1",
           fn -> Nx.tensor(3_000_000_000) end},
          {"integer_past_s32", "5000000000 as s32", "Nx.max/2",
           fn -> Nx.max(Nx.tensor(0, type: :s64), 5_000_000_000) end},
          {"literal_wraps", "300 as s8", "Nx.tensor/2",
           fn -> Nx.tensor([300, 301], type: :s8) end},
          {"literal_wraps", "-1 as u8", "Nx.tensor/2", fn -> Nx.tensor(-1, type: :u8) end},
          {"literal_wraps", "256 as u8", "Nx.u8/1", fn -> Nx.u8([256, -1]) end},
          {"literal_wraps", "18446744073709551616 as u64", "Nx.tensor/2",
           fn -> Nx.tensor(18_446_744_073_709_551_616, type: :u64) end},
          {"literal_overflows", "7.0e4 as f16", "Nx.tensor/2",
           fn -> Nx.tensor(70_000.0, type: :f16) end},
          {"literal_flushes", "1.0e-8 as f16", "Nx.tensor/2",
           fn -> Nx.tensor(1.0e-8, type: :f16) end}
        ] do
      %{label: label} =
        TensorShapes.Wording.Literals.type_error(kind, subject, operation, "0", "1")

      made =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          make.() |> Nx.flatten() |> Nx.to_flat_list() |> hd()
        end)

      shown = if made in [:infinity, :neg_infinity], do: "an infinity", else: to_string(made)

      assert label =~ ~r/ becomes #{Regex.escape(shown)} here$/,
             "#{subject}: #{label}, and Nx makes #{inspect(made)}"
    end
  end

  # The kind of finding a call of `Nx.Type` with short atom types expects:
  # where it raises, or answers otherwise than with the types as tuples.
  defp atom_type_outcome(function, arguments) do
    tuples = Enum.map(arguments, &if(is_atom(&1), do: Nx.Type.normalize!(&1), else: &1))

    case {call_nx_type(function, arguments), call_nx_type(function, tuples)} do
      {:raises, _tuple} -> "atom_type_rejected"
      {same, same} -> nil
      {atom, tuple} -> if as_type(atom) == tuple, do: nil, else: "atom_type_misread"
    end
  end

  defp call_nx_type(function, arguments) do
    {:returns, apply(Nx.Type, function, arguments)}
  rescue
    _error -> :raises
  end

  # An answer that is a type as an atom, as its tuple.
  defp as_type({:returns, atom}) when is_atom(atom) and atom not in [true, false, nil] do
    {:returns, Nx.Type.normalize!(atom)}
  rescue
    ArgumentError -> {:returns, atom}
  end

  defp as_type(answer), do: answer

  # (end of Literals tests)

  # ── Access: tests of their own ──

  @access_fixtures ArgusNxTensorAnalyses.TensorShapesTest.AccessFixtures

  for {{body, expected}, index} <- Enum.with_index(@access_shapes, 1) do
    test "access shape #{index}: #{String.replace(body, "\n", "; ")}", %{rows: rows} do
      index = unquote(index)
      derived = returned_shapes(rows, function_id(@access_fixtures, "shape_#{index}", 0))

      {:returns, value} = outcome_on_binary_backend(@access_fixtures, :"shape_#{index}", [])
      computed = spell(value)

      case unquote(expected) do
        :nx ->
          assert derived == [computed],
                 "Nx gives #{computed}, and the analysis derives #{inspect(derived)}"

        shape ->
          assert derived == [shape], "expected #{shape}, the analysis derives #{inspect(derived)}"
          assert rank(shape) == rank(computed), "Nx gives #{computed}"
      end
    end
  end

  # The slices Nx takes draw no finding.
  test "an access Nx accepts is not reported", %{rows: rows} do
    shape = "#{inspect(@access_fixtures)}:shape_"

    reported =
      findings_for(rows, "tensor_call_error", &String.starts_with?(&1, shape), [:func, :kind])

    assert reported == []
  end

  for {{body, kind, detail, outcome}, index} <- Enum.with_index(@access_findings, 1) do
    test "access finding #{index}: #{String.replace(body, "\n", "; ")}", %{rows: rows} do
      index = unquote(index)
      function = function_id(@access_fixtures, "finding_#{index}", 0)
      found = findings_for(rows, "tensor_call_error", function, [:kind, :detail])

      assert found == [{unquote(kind), unquote(detail)}]

      assert_access_outcome(
        unquote(outcome),
        unquote(detail),
        outcome_on_binary_backend(@access_fixtures, :"finding_#{index}", [])
      )
    end
  end

  # In a `defn`, `k - 1` of a number `k` is a number, so `0..(k - 1)` of a
  # `k` of 0 steps down and `k..(k - 1)//1` of a `k` of 2 holds nothing,
  # while their neighbors give ranges Nx takes.
  test "a range a defn computes of numbers is checked", %{rows: rows} do
    found =
      &findings_for(
        rows,
        "tensor_call_error",
        function_id(@access_fixtures, "__defn:#{&1}__", 1),
        [
          :kind,
          :detail
        ]
      )

    assert found.(:head_before) == [
             {"access_negative_step", "range step must be positive, got range: 0..-1//-1"}
           ]

    assert found.(:span_before) == [
             {"access_empty_range", "slicing a tensor requires a non-empty range, got: 2..1//1"}
           ]

    assert found.(:head_through) == []
    assert found.(:span_through) == []

    assert {:raises, %ArgumentError{}} =
             outcome_on_binary_backend(@access_fixtures, :heads_before, [])

    assert {:raises, error} = outcome_on_binary_backend(@access_fixtures, :spans_before, [])
    assert Exception.message(error) == "slicing a tensor requires a non-empty range, got: 2..1//1"
    assert {:returns, _value} = outcome_on_binary_backend(@access_fixtures, :heads_through, [])
    assert {:returns, _value} = outcome_on_binary_backend(@access_fixtures, :spans_through, [])
  end

  test "run/2 words and places an access's findings", %{placed: placed, source: source} do
    by_title = placed |> placed_in(@access_fixtures) |> Enum.group_by(& &1.finding.title)

    [bounds | _] = by_title["Access.get/2 indexes past the end of an axis"]
    assert bounds.finding.severity == :error
    assert source_line(source, bounds.line) == "Nx.iota({4, 5})[4]"

    [clamped | _] = by_title["Access.get/2 clamps a scalar tensor index into its axis"]
    assert clamped.finding.severity == :warning

    [tuple] = by_title["Nx.sum/1 gets a tuple of tensors where it takes a tensor"]
    assert tuple.finding.detail =~ "Its first argument is a tuple of tensors"
    assert [%{label: "returns the tuple: Nx.split/2"}] = tuple.finding.related
  end

  defp assert_access_outcome(:message, detail, outcome) do
    assert {:raises, error} = outcome
    assert Exception.message(error) == detail
  end

  defp assert_access_outcome(:raises, _detail, outcome), do: assert({:raises, _error} = outcome)

  defp assert_access_outcome(:accepted, _detail, outcome),
    do: assert({:returns, _value} = outcome)

  defp rank(spelled) do
    case spelled |> String.split(["{", "}"]) |> Enum.at(1) do
      "" -> 0
      sizes -> sizes |> String.split(",") |> length()
    end
  end

  # (end of Access tests)

  # ── Tuples: tests of their own ──
  # What the findings of `priv/tensor_shapes/tuples.dl` say is what Nx
  # raises, or what a sampler or norm computes instead of what was meant.
  test "the tuples rules say what Nx raises or computes", %{rows: rows} do
    details =
      for {kind, detail} <-
            findings_for(
              rows,
              ["tensor_shape_mismatch", "tensor_call_error"],
              fn _function -> true end,
              [:kind, :detail]
            ),
          kind in ["key", "sampler_parameters", "shared_draw", "norm_axes"],
          uniq: true,
          do: {kind, detail}

    for expected <- [
          {"key", "expected key to have shape {2}, got tensor with shape {2, 2}"},
          {"sampler_parameters", "cannot broadcast tensor of dimensions {2} to {3}"},
          {"sampler_parameters",
           "cannot reshape, current shape {3} is not compatible with new shape {}"},
          {"sampler_parameters", "cannot broadcast tensor of dimensions {3} to {}"},
          {"shared_draw", "draws {} and returns {2, 3}"},
          {"norm_axes", "1"}
        ] do
      assert_finding(details, expected)
    end
  end

  test "run/2 warns of a normal sampler that repeats its draw", %{placed: placed} do
    shared =
      placed
      |> placed_in(@lint_fixtures)
      |> Enum.find(&(&1.finding.title =~ "repeats one draw"))

    assert shared.finding.severity == :warning
    assert shared.finding.title =~ "Nx.Random.normal/3"
    assert shared.finding.detail =~ "draws {} and returns {2, 3}"
  end

  # Inside a `defn`, `Nx.Random`'s functions compile to
  # `Nx.Defn.Compiler.__remote__/4`, which hands them their arguments in a
  # list (a single literal for `Nx.Random.key(42)`), and `Nx.LinAlg`'s to
  # direct calls.
  test "samplers and decompositions in a defn give the shapes they give outside one", %{
    rows: rows
  } do
    helpers = ArgusNxTensorAnalyses.TensorShapesTest.TupleHelpers
    key = Nx.Random.key(42)
    scalar = Nx.tensor(1.0)

    findings = fn name ->
      rows
      |> findings_for("tensor_shape_mismatch", defn_id(helpers, name, 1), [
        :operation,
        :kind,
        :detail
      ])
      |> Enum.uniq()
    end

    for {name, argument, finding} <- [
          {:key_added, scalar,
           {"Nx.add/2", "broadcast", "cannot broadcast tensor of dimensions {2} to {3}"}},
          {:sample_added, key,
           {"Nx.Defn.Kernel.+/2", "broadcast",
            "cannot broadcast tensor of dimensions {2, 3} to {4}"}},
          {:split_drawn, key,
           {"Nx.Random.uniform/1", "key",
            "expected key to have shape {2}, got tensor with shape {2, 2}"}},
          {:uniform_over, key,
           {"Nx.Random.uniform/4", "sampler_parameters",
            "cannot broadcast tensor of dimensions {2} to {3}"}},
          {:parts_added, scalar,
           {"Nx.Defn.Kernel.+/2", "broadcast",
            "cannot broadcast tensor of dimensions {4, 4} to {4, 2}"}}
        ] do
      assert findings.(name) == [finding]
      assert {:raises, _error} = outcome_on_binary_backend(helpers, name, [argument])
    end

    for {name, argument} <- [
          key_added_fits: scalar,
          sample_added_fits: key,
          parts_multiplied: scalar
        ] do
      assert findings.(name) == []
      assert {:returns, _value} = outcome_on_binary_backend(helpers, name, [argument])
    end
  end

  # (end of Tuples tests)

  # ── ShapeGaps: tests of their own ──

  @shape_gaps ArgusNxTensorAnalyses.TensorShapesTest.ShapeGaps

  # Each function of the module returns a tensor of a shape only these
  # rules derive, for a call the core leaves without one or gives another.
  test "shape gaps: a newly modeled call gets the shape Nx gives it", %{rows: rows} do
    for {name, 0} <- apply(@shape_gaps, :__info__, [:functions]) do
      {:returns, value} = outcome_on_binary_backend(@shape_gaps, name, [])
      expected = spell(value)
      derived = returned_shapes(rows, function_id(@shape_gaps, name, 0))

      assert derived == [expected],
             "#{name}: Nx gives #{expected}, the analysis derives #{inspect(derived)}"
    end
  end

  # (end of ShapeGaps tests)

  # ── ReshapeOrder: tests of their own ──
  @reshape_order ArgusNxTensorAnalyses.TensorShapesTest.ReshapeOrder

  # Real sizes, batch 1, seq 2 and heads and head_dim 2 each: a reshape
  # alone gives the shape the transposing version does and other data.
  test "a reshape that moves heads past the sequence scrambles the data", %{rows: rows} do
    sizes = %{batch: 1, seq: 2, heads: 2, head_dim: 2}

    run = fn name ->
      {:returns, value} = outcome_on_binary_backend(@reshape_order, name, [sizes])
      value
    end

    split = run.(:split_heads)
    split_and_transposed = run.(:split_heads_and_transpose)
    assert Nx.shape(split) == Nx.shape(split_and_transposed)
    assert Nx.to_flat_list(split) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert Nx.to_flat_list(split_and_transposed) == [0, 1, 4, 5, 2, 3, 6, 7]

    projection = run.(:projection)
    merged = run.(:merge_heads)
    assert Nx.shape(merged) == Nx.shape(projection)
    assert Nx.to_flat_list(merged) == [0, 1, 4, 5, 2, 3, 6, 7]
    assert Nx.to_flat_list(run.(:transpose_and_merge_heads)) == Nx.to_flat_list(projection)

    assert reshape_order_functions(rows) == ["merge_heads/1", "split_heads/1"]
  end

  test "a reshape that moves a size says which sizes it moves", %{rows: rows} do
    function = function_id(@reshape_order, :split_heads, 1)

    [finding] =
      for [_id, ^function, _operation, "reshape_order" | _rest] = row <-
            Map.get(rows, "tensor_axis_misalignment", []),
          do: TensorShapes.finding(:tensor_axis_misalignment, row)

    assert finding.severity == :warning

    assert finding.title ==
             "Nx.reshape/2 moves a size past another axis, which scrambles the data"

    assert finding.detail =~
             "the tensor has seq on axis 1 before heads on axis 2, and the result has heads on axis 1 before seq on axis 2"
  end

  defp reshape_order_functions(rows) do
    prefix = "#{inspect(@reshape_order)}:"

    functions =
      for {function, "reshape_order"} <-
            findings_for(rows, "tensor_axis_misalignment", &String.starts_with?(&1, prefix), [
              :func,
              :kind
            ]),
          uniq: true,
          do: String.replace_prefix(function, prefix, "")

    Enum.sort(functions)
  end

  # (end of ReshapeOrder tests)

  # ── Dtypes: tests of their own ──
  @dtypes ArgusNxTensorAnalyses.TensorShapesTest.Dtypes
  @dtypes_configured ArgusNxTensorAnalyses.TensorShapesTest.DtypesConfigured

  # Each function of `Dtypes`, the wrong value Nx gives it on the binary
  # backend, and the finding that says why.
  @dtypes_wrong_values [
    {:mask_minus_one, [0, 255], {"tensor_call_error", "unsigned_wraparound", "u8"}},
    {:mask_bias, [0.0, 255_000_002_560.0], {"tensor_call_error", "unsigned_wraparound", "u8"}},
    {:negated_mask, [255, 0], {"tensor_call_error", "unsigned_wraparound", "u8"}},
    {:hand_sign, [1, 255, 0], {"tensor_call_error", "unsigned_wraparound", "u8"}},
    {:centered_pixels, [178, 72], {"tensor_call_error", "unsigned_wraparound", "u8"}},
    {:zero_minus_one, [4_294_967_295], {"tensor_call_error", "unsigned_wraparound", "u32"}},
    {:mask_differences, [0, 255, 0, 1], {"tensor_call_error", "unsigned_wraparound", "u8"}},
    {:byte_closeness, [0], {"tensor_call_error", "unsigned_wraparound", "u8"}},
    {:descending_bytes, [10, 133, 0], {"tensor_call_error", "unsigned_wraparound", "u8"}},
    {:window_count, [44], {"tensor_call_error", "count_wraparound", "u8 300"}},
    {:confusion, [44, 0, 0, 0, 0, 0, 0, 0, 0],
     {"tensor_call_error", "count_wraparound", "u8 300"}},
    {:byte_dot, [-112], {"tensor_call_error", "narrow_wraparound", "s8"}},
    {:byte_median, [17.0], {"tensor_call_error", "narrow_wraparound", "u8"}},
    {:byte_window_mean, [22.0, 75.0], {"tensor_call_error", "narrow_wraparound", "u8"}},
    {:byte_weighted_mean, [86.0], {"tensor_call_error", "narrow_wraparound", "u8"}},
    {:byte_argmax, [43], {"tensor_call_error", "index_wraparound", "u8 300"}},
    {:signed_byte_argmax, [-57], {"tensor_call_error", "index_wraparound", "s8 200"}},
    {:cast_to_bytes, [44, 255, 2, 254], {"tensor_call_error", "cast_wraparound", "u8"}},
    {:filled_bytes, [255], {"tensor_call_error", "cast_wraparound", "u8"}},
    {:padded_with_half, [1_056_964_608, 1],
     {"tensor_call_error", "pad_type_mismatch", "s32 f32"}},
    {:padded_with_minus_one, [255, 255], {"tensor_call_error", "pad_type_mismatch", "u8 s16"}},
    {:byte_determinant, [4_294_967_296.0], {"tensor_call_error", "integer_determinant", "u8"}},
    {:unsigned_meets_signed, [-1], {"tensor_type_error", "narrowing_merge", "u64 s64"}},
    {:defn_bias, [0.0, 255_000_002_560.0], {"tensor_call_error", "unsigned_wraparound", "u8"}}
  ]

  for {function, wrong, finding} <- @dtypes_wrong_values do
    test "dtypes: #{function} gives #{inspect(wrong)}, and the analysis says why", %{rows: rows} do
      function = unquote(function)
      wrong = unquote(Macro.escape(wrong))
      finding = unquote(Macro.escape(finding))

      assert dtypes_value(function) == wrong
      assert_finding(dtypes_findings(rows, @dtypes, function), finding)
    end
  end

  # A long run of whole numbers wraps or repeats where the type runs out.
  test "dtypes: counts and positions past a type's limit", %{rows: rows} do
    assert dtypes_value(:running_count) |> Enum.slice(254, 4) == [255, 0, 1, 2]

    assert_finding(
      dtypes_findings(rows, @dtypes, :running_count),
      {"tensor_call_error", "count_wraparound", "u8 300"}
    )

    assert dtypes_value(:bf16_positions) |> Enum.slice(255, 4) == [255.0, 256.0, 256.0, 258.0]

    assert_finding(
      dtypes_findings(rows, @dtypes, :bf16_positions),
      {"tensor_call_error", "sequence_precision", "bf16 1000"}
    )

    assert dtypes_value(:byte_positions) |> Enum.slice(254, 4) == [254, 255, 0, 1]

    assert_finding(
      dtypes_findings(rows, @dtypes, :byte_positions),
      {"tensor_call_error", "sequence_precision", "u8 300"}
    )
  end

  # A logarithm to a base of an f64 tensor, and the same computed in f64.
  test "dtypes: an f64 logarithm to a base is only as accurate as an f32", %{rows: rows} do
    assert dtypes_value(:log2_of_f64) == [0.9999999972521647]

    assert_finding(
      dtypes_findings(rows, @dtypes, :log2_of_f64),
      {"tensor_call_error", "f32_precision", "f64"}
    )

    assert dtypes_value(:log2_of_f64_in_f64) == [1.0]

    refute Enum.any?(
             dtypes_findings(rows, @dtypes, :log2_of_f64_in_f64),
             &match?({_relation, "f32_precision", _detail}, &1)
           )
  end

  # With the float types the code may run at, a type read from
  # configuration is each of them in turn.
  test "dtypes: a configured type is checked as each float type", %{
    rows: rows,
    float_rows: float_rows
  } do
    findings = &dtypes_findings(float_rows, @dtypes_configured, &1)

    assert_finding(findings.(:masked), {"tensor_nonfinite_result", "literal_overflow", "f16"})
    assert_finding(findings.(:scaled), {"tensor_type_error", "upcast", "bf16 f32"})
    assert_finding(findings.(:scaled), {"tensor_type_error", "upcast", "f16 f32"})
    assert findings.(:same_type) == []

    # one finding for a call, naming each float type that breaks it
    assert findings.(:positions) == [
             {"tensor_call_error", "sequence_precision", "f16/bf16 arg0"}
           ]

    assert findings.(:real_part) == [{"tensor_call_error", "complex_to_real", "c64 f16/bf16/f32"}]

    assert findings.(:tiny_epsilon) == [
             {"tensor_call_error", "literal_underflow", "1.0e-46 f16/bf16/f32"}
           ]

    titles = Enum.map(Argus.Findings.build(TensorShapes, float_rows), & &1.title)
    assert "Nx.multiply/2 counts past what f16 or bf16 holds exactly" in titles

    assert_finding(findings.(:against_f16), {"tensor_type_error", "narrowing_merge", "bf16 f16"})

    assert_finding(
      findings.(:normalized),
      {"tensor_call_error", "literal_underflow", "1.0e-12 f16"}
    )

    assert findings.(:written_bf16) == []

    # without them, a type read from configuration is not known
    for function <- [:masked, :scaled, :positions, :against_f16, :normalized],
        do: assert(dtypes_findings(rows, @dtypes_configured, function) == [])
  end

  test "dtypes: a finding reads as what Nx does", %{rows: rows} do
    findings = Argus.Findings.build(TensorShapes, rows)

    wrap =
      Enum.find(
        findings,
        &(&1.title == "Nx.negate/1 can go below zero in u8, which wraps around" and
            &1.related != [])
      )

    assert wrap.severity == :warning
    assert [%{label: "makes it u8: Nx.greater/2"}] = wrap.related

    count = Enum.find(findings, &(&1.title == "Nx.window_sum/2 counts past what u8 holds"))
    assert count.detail =~ "over 300 elements"
    assert count.severity == :warning

    unknown = Enum.find(findings, &(&1.title == "Nx.cumulative_sum/2 counts past what u8 holds"))
    assert unknown.detail =~ "an axis whose length the code does not show"
    assert unknown.severity == :info
  end

  defp dtypes_value(function) do
    {:returns, value} = outcome_on_binary_backend(@dtypes, function, [])
    Nx.to_flat_list(value)
  end

  # A function's findings as `{relation, kind, detail}`, in it and in the
  # `defn` named for it that it runs.
  defp dtypes_findings(rows, module, function) do
    prefix = "#{inspect(module)}:"
    name = Atom.to_string(function)

    owns? = fn found_in ->
      String.starts_with?(found_in, prefix <> name <> "/") or
        String.starts_with?(found_in, prefix <> "__defn:" <> name)
    end

    [tensor_call_error: :detail, tensor_nonfinite_result: :cause, tensor_type_error: :subject]
    |> Enum.flat_map(fn {relation, detail} ->
      findings_for(rows, Atom.to_string(relation), owns?, [:relation, :kind, detail])
    end)
    |> Enum.uniq()
  end

  # (end of Dtypes tests)

  # ── Serving: tests of their own ──
  @servings ArgusNxTensorAnalyses.TensorShapesTest.Servings

  # The call errors found in the functions of the serving fixtures whose
  # names start with `name`, as `{operation, kind, detail}`.
  defp serving_findings(rows, name) do
    prefix = "#{inspect(@servings)}:#{name}"

    rows
    |> findings_for("tensor_call_error", &String.starts_with?(&1, prefix), [
      :operation,
      :kind,
      :detail
    ])
    |> Enum.uniq()
  end

  defp serving_batch(entries) do
    Nx.with_default_backend(Nx.BinaryBackend, fn -> Nx.Batch.stack(entries) end)
  end

  test "a serving's computation defined elsewhere that reduces every axis", %{rows: rows} do
    assert [{"Nx.Serving.jit/1", "serving_scalar_output", _detail}] =
             serving_findings(rows, "total_serving/0")

    assert serving_findings(rows, "per_entry_serving/0") == []
    batch = serving_batch([Nx.tensor([1, 2, 3])])

    assert_raise ArgumentError, ~r/given axis \(0\) invalid for shape with rank 0/, fn ->
      apply(@servings, :run_total, [batch])
    end

    assert apply(@servings, :run, [apply(@servings, :per_entry_serving, []), batch]) ==
             Nx.tensor([6])
  end

  # Nx.Batch.pad's zero rows shift the mean over the batch axis, and with
  # it every entry's result; over each entry's own axis they do not.
  test "a serving's computation that reduces along the batch axis mixes entries", %{rows: rows} do
    assert [{"Nx.mean/2", "serving_mixes_batch", "it reduces the batch axis"}] =
             serving_findings(rows, "__defn:centered__/1")

    assert serving_findings(rows, "__defn:centered_per_entry__/1") == []

    batch = serving_batch([Nx.tensor([1.0, 3.0])])
    padded = Nx.Batch.pad(batch, 1)
    run = &apply(@servings, :run, [apply(@servings, &1, []), &2])

    refute run.(:centered_serving, batch) == run.(:centered_serving, padded)

    assert run.(:centered_per_entry_serving, batch) ==
             run.(:centered_per_entry_serving, padded)
  end

  # The first entry's result changes with the entry batched after it.
  test "a serving's computation that contracts or sorts along the batch axis", %{rows: rows} do
    assert serving_findings(rows, "-sorting_known/0-fun-") ==
             [
               {"Nx.sort/1", "serving_mixes_batch",
                "it sorts or accumulates along the batch axis"}
             ]

    assert serving_findings(rows, "-contracting/1-fun-") ==
             [{"Nx.dot/4", "serving_mixes_batch", "it contracts the batch axis"}]

    first_result = fn function, entry ->
      @servings
      |> apply(function, [[entry]])
      |> Nx.slice_along_axis(0, 1, axis: 0)
    end

    refute first_result.(:contracting, Nx.tensor([3.0, 4.0])) ==
             first_result.(:contracting, Nx.tensor([5.0, 6.0]))

    refute first_result.(:sorting, Nx.tensor([2, 4])) ==
             first_result.(:sorting, Nx.tensor([5, 0]))
  end

  # A serving process runs smaller batches than its batch size when its
  # batch times out: one entry raises against a template of four, and a
  # builder that pads each batch to four takes it.
  test "a serving process compiled ahead of time for a full batch", %{rows: rows} do
    assert [{"Nx.Defn.compile/3", "serving_template_batch_size", _detail}] =
             serving_findings(rows, "-aot_serving/0-fun-")

    assert serving_findings(rows, "-padded_aot_serving/0-fun-") == []
    batch = serving_batch([Nx.tensor([1, 2, 3])])

    assert_raise ArgumentError, ~r/not compatible with compiled function template/, fn ->
      apply(@servings, :run, [apply(@servings, :aot_serving, []), batch])
    end

    assert apply(@servings, :run, [apply(@servings, :padded_aot_serving, []), batch]) ==
             Nx.tensor([[2, 4, 6]])
  end

  # Two requests of different shapes batched together: both callers exit,
  # and the serving's supervisor, which does not restart, dies with them.
  test "a serving process's batches of requests of different shapes cannot merge", %{rows: rows} do
    assert [{"Nx.Serving.client_preprocessing/2", "serving_entry_shape_varies", _detail}] =
             serving_findings(rows, "varying_serving/0")

    parent = self()

    # the serving logs its crash
    ExUnit.CaptureLog.capture_log(fn ->
      runner =
        spawn(fn ->
          Process.flag(:trap_exit, true)

          {:ok, supervisor} =
            Nx.Serving.start_link(
              serving: apply(@servings, :varying_serving, []),
              name: ServingVaryingRun,
              batch_size: 4,
              batch_timeout: 200
            )

          requests =
            for input <- [Nx.tensor([1, 2, 3]), Nx.tensor([1, 2])] do
              Task.async(fn -> catch_exit(Nx.Serving.batched_run(ServingVaryingRun, input)) end)
            end

          results = Task.await_many(requests, 5_000)

          supervisor_exit =
            receive do
              {:EXIT, ^supervisor, reason} -> reason
            after
              5_000 -> :alive
            end

          send(parent, {:served, results, supervisor_exit})
        end)

      assert_receive {:served, results, supervisor_exit}, 15_000
      assert Enum.all?(results, &match?({_reason, {Nx.Serving, _function, _arguments}}, &1))
      refute supervisor_exit == :alive
      Process.exit(runner, :kill)
    end)
  end

  test "run/2 words a serving finding and places it at the serving's call", %{placed: placed} do
    titles =
      for %{finding: finding} <- placed_in(placed, @servings),
          do: {finding.severity, finding.title}

    assert {:error, "Nx.Serving.jit/1 compiles a serving computation whose output is a scalar"} in titles

    assert {:warning,
            "Nx.Defn.compile/3 compiles a serving computation for a batch size its batches do not have"} in titles

    assert {:warning,
            "Nx.Serving.client_preprocessing/2 sets a preprocessing whose batches a serving process cannot merge"} in titles
  end

  # (end of Serving tests)

  # ── Consumption: tests of their own ──
  # Nx raises nothing on a key drawn from twice, so each test runs the
  # fixture and shows the equal samples beside the finding. A freed tensor
  # raises only on EMLX and EXLA, where the BinaryBackend frees nothing, so
  # those tests assert the findings alone.

  @consumption_fixtures ArgusNxTensorAnalyses.TensorShapesTest.Consumption

  @consumption_kinds ~w(reused_key captured_loop_key passed_back_key spent_key_returned
                        used_after_transfer used_after_deallocation used_after_donation)

  test "a key drawn from twice gives equal samples, and a threaded key does not", %{rows: rows} do
    key = Nx.Random.key(42)
    {first, second} = consume(:drawn_twice, [key])
    assert Nx.to_flat_list(first) == Nx.to_flat_list(second)

    assert consumption_findings(rows, "drawn_twice") == [
             {"reused_key", "Nx.Random.uniform after Nx.Random.uniform"}
           ]

    {first, second} = consume(:drawn_threaded, [key])
    refute Nx.to_flat_list(first) == Nx.to_flat_list(second)
    assert consumption_findings(rows, "drawn_threaded") == []
  end

  # Inside a `defn`, `Nx.Random.uniform/2` compiles to
  # `Nx.Defn.Compiler.__remote__/4`, which hands it the key in a list.
  test "a key drawn from twice in a defn is one finding at the sampler it calls", %{rows: rows} do
    key = Nx.Random.key(42)
    {first, second} = consume(:drawn_twice_in_defn, [key])
    assert Nx.to_flat_list(first) == Nx.to_flat_list(second)

    function = defn_id(@consumption_fixtures, :drawn_twice_in_defn, 1)

    assert findings_for(rows, "tensor_call_error", function, [
             :operation,
             :kind,
             :origin_operation
           ]) == [{"Nx.Random.uniform/2", "reused_key", "Nx.Random.uniform/2"}]

    {first, second} = consume(:drawn_threaded_in_defn, [key])
    refute Nx.to_flat_list(first) == Nx.to_flat_list(second)
    assert consumption_findings(rows, "drawn_threaded_in_defn") == []
  end

  test "a normal drawn from a key a uniform drew from orders its samples as the uniform does",
       %{rows: rows} do
    {uniform, normal} = consume(:normal_after_uniform, [Nx.Random.key(42)])
    assert Nx.to_flat_list(Nx.argsort(uniform)) == Nx.to_flat_list(Nx.argsort(normal))

    assert consumption_findings(rows, "normal_after_uniform") == [
             {"reused_key", "Nx.Random.normal after Nx.Random.uniform"}
           ]
  end

  test "a helper that draws from the key it is handed, called twice with one key", %{
    rows: rows
  } do
    {first, second} = consume(:sampled_twice, [Nx.Random.key(42)])
    assert Nx.to_flat_list(first) == Nx.to_flat_list(second)
    assert [{"reused_key", _detail}] = consumption_findings(rows, "sampled_twice")
  end

  test "two reads of one field are one key", %{rows: rows} do
    {first, second} = consume(:field_drawn_twice, [%{key: Nx.Random.key(42)}])
    assert Nx.to_flat_list(first) == Nx.to_flat_list(second)

    assert consumption_findings(rows, "field_drawn_twice") == [
             {"reused_key", "Nx.Random.uniform after Nx.Random.uniform"}
           ]
  end

  test "a recursion that hands itself the key it drew from draws equal rows", %{rows: rows} do
    key = Nx.Random.key(42)
    assert [row, row, row] = :recursive_rows |> consume([key, 3]) |> Enum.map(&Nx.to_flat_list/1)

    assert consumption_findings(rows, "recursive_rows") == [
             {"reused_key",
              "#{inspect(@consumption_fixtures)}.recursive_rows after Nx.Random.uniform"}
           ]

    assert :recursive_threaded_rows
           |> consume([key, 3])
           |> Enum.map(&Nx.to_flat_list/1)
           |> Enum.uniq()
           |> length() == 3

    assert consumption_findings(rows, "recursive_threaded_rows") == []
  end

  test "a loop's fun that captures the key draws equal rows, and one that folds in the pass's index does not",
       %{rows: rows} do
    key = Nx.Random.key(42)
    assert [row, row, row] = Enum.map(consume(:captured_rows, [key]), &Nx.to_flat_list/1)

    assert consumption_findings(rows, "captured_rows") == [
             {"captured_loop_key", "Nx.Random.uniform"}
           ]

    assert :folded_rows
           |> consume([key])
           |> Enum.map(&Nx.to_flat_list/1)
           |> Enum.uniq()
           |> length() == 3

    assert consumption_findings(rows, "folded_rows") == []
  end

  test "a map_reduce whose fun hands back the key it drew from draws equal rows", %{rows: rows} do
    key = Nx.Random.key(42)
    assert [row, row, row] = Enum.map(consume(:passed_back_rows, [key]), &Nx.to_flat_list/1)

    assert consumption_findings(rows, "passed_back_rows") == [
             {"passed_back_key", "Nx.Random.uniform"}
           ]

    assert :threaded_rows
           |> consume([key])
           |> Enum.map(&Nx.to_flat_list/1)
           |> Enum.uniq()
           |> length() == 3

    assert consumption_findings(rows, "threaded_rows") == []
  end

  test "a while whose state hands back the key it drew from draws equal rows", %{rows: rows} do
    key = Nx.Random.key(42)
    assert [row, row, row] = :while_rows |> consume([key]) |> Nx.to_list()

    assert consumption_findings(rows, "while_rows") == [
             {"passed_back_key", "Nx.Random.uniform"}
           ]

    assert :while_threaded_rows |> consume([key]) |> Nx.to_list() |> Enum.uniq() |> length() == 3
    assert consumption_findings(rows, "while_threaded_rows") == []
  end

  test "a function that returns the key it drew from hands its caller the same bits", %{
    rows: rows
  } do
    key = Nx.Random.key(42)
    {sample, returned} = consume(:spent, [key])
    {again, _key} = Nx.Random.uniform(returned, shape: {4})
    assert Nx.to_flat_list(again) == Nx.to_flat_list(sample)
    assert consumption_findings(rows, "spent") == [{"spent_key_returned", "Nx.Random.uniform"}]

    state = consume(:spent_in_state, [%{key: key, sample: nil}])
    {again, _key} = Nx.Random.uniform(state.key, shape: {4})
    assert Nx.to_flat_list(again) == Nx.to_flat_list(state.sample)

    assert consumption_findings(rows, "spent_in_state") == [
             {"spent_key_returned", "Nx.Random.uniform"}
           ]

    state = consume(:threaded_state, [%{key: key, sample: nil}])
    {again, _key} = Nx.Random.uniform(state.key, shape: {4})
    refute Nx.to_flat_list(again) == Nx.to_flat_list(state.sample)
    assert consumption_findings(rows, "threaded_state") == []

    {:reply, sample, state} = consume(:sample_reply, [%{key: key}])
    {again, _key} = Nx.Random.uniform(state.key, shape: {4})
    assert Nx.to_flat_list(again) == Nx.to_flat_list(sample)

    assert consumption_findings(rows, "sample_reply") == [
             {"spent_key_returned", "Nx.Random.uniform"}
           ]
  end

  test "a state handed back holding the tensor a transfer freed", %{rows: rows} do
    assert consumption_findings(rows, "host_reply") == [{"used_after_transfer", "returned"}]
    assert consumption_findings(rows, "host_reply_updated") == []
  end

  test "a freed tensor handed to a helper, a defn, or in a term the transfer freed", %{
    rows: rows
  } do
    assert consumption_findings(rows, "transfer_then_helper") == [
             {"used_after_transfer", "handed_on"}
           ]

    assert consumption_findings(rows, "deallocate_then_defn") == [
             {"used_after_deallocation", "read"}
           ]

    assert consumption_findings(rows, "transfer_container") == [{"used_after_transfer", "read"}]
  end

  test "a read of a freed tensor is an error, and a key drawn from twice a warning", %{
    rows: rows
  } do
    finding = fn function, kind ->
      id = "#{inspect(@consumption_fixtures)}:#{function}"

      row =
        Enum.find(Map.get(rows, "tensor_call_error", []), &match?([_, ^id, _, ^kind | _], &1))

      TensorShapes.finding(:tensor_call_error, row)
    end

    read = finding.("transfer_container/2", "used_after_transfer")
    assert read.severity == :error
    assert read.title == "Nx.add/2 reads a tensor after Nx.backend_transfer freed it"
    assert [%{label: "freed by Nx.backend_transfer/1"}] = read.related

    reused = finding.("drawn_twice/1", "reused_key")
    assert reused.severity == :warning

    assert reused.title ==
             "Nx.Random.uniform/2 draws from a random key that was already drawn from"

    assert [%{label: "the key was first drawn from by Nx.Random.uniform/2"}] = reused.related

    returned = finding.("spent/1", "spent_key_returned")
    assert returned.severity == :warning
    assert returned.title == "returns a random key it has already drawn from"
  end

  defp consume(name, arguments) do
    {:returns, value} = outcome_on_binary_backend(@consumption_fixtures, name, arguments)
    value
  end

  # The findings of priv/tensor_shapes/consumption.dl in a function of the
  # Consumption fixtures, its closures and `defn` body among them, as
  # `{kind, detail}`.
  defp consumption_findings(rows, name) do
    module = "#{inspect(@consumption_fixtures)}:"
    owned = ["#{name}/", "-#{name}/", "__defn:#{name}__/", "-__defn:#{name}__/"]

    owns? = fn function ->
      String.starts_with?(function, module) and
        String.starts_with?(String.replace_prefix(function, module, ""), owned)
    end

    for {kind, detail} <- findings_for(rows, "tensor_call_error", owns?, [:kind, :detail]),
        kind in @consumption_kinds,
        uniq: true,
        do: {kind, detail}
  end

  # (end of Consumption tests)

  # ── LinAlg: tests of their own ──
  # (end of LinAlg tests)

  test "run/2 places a finding at its call", %{placed: placed, source: source} do
    reshape =
      placed
      |> placed_in(@fixtures)
      |> Enum.find(&(&1.finding.detail =~ "shape {2, 3} is not compatible with new shape {4, 2}"))

    assert reshape.file == source
    assert source_line(source, reshape.line) == "Nx.reshape(Nx.iota({2, 3}), {4, 2})"
  end

  test "run/2 places the calls that bring a helper its shapes", %{placed: placed, source: source} do
    callers =
      for %{finding: finding, related: frames} <- placed_in(placed, @fixtures),
          finding.detail =~ "cannot broadcast tensor of dimensions {2, 3} to {4}",
          frame <- frames,
          do: source_line(source, frame.line)

    assert "local_helper(Nx.iota({2, 3}))" in callers
  end

  test "run/2 labels a call with the shapes it gets and the call that makes each", %{
    placed: placed
  } do
    reshape =
      placed
      |> placed_in(@fixtures)
      |> Enum.find(&(&1.finding.detail =~ "shape {2, 3} is not compatible with new shape {4, 2}"))

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

    assert Enum.all?(found, &match?({_operation, _kind, _detail, "on_some_path"}, &1)),
           inspect(found)
  end

  defp assert_agrees(:unknown, {:returns, _shape}, derived, found) do
    assert {derived, found} == {[], []}
  end

  defp assert_agrees(:no_finding, {:returns, _shape}, _derived, found) do
    assert found == []
  end

  # What Nx does with the case: the shape of the tensor it returns, as
  # the probe program spells it (nil for anything else), or the message it
  # raises.
  defp run_case(index) do
    case outcome_on_binary_backend(@fixtures, :"case_#{index}", []) do
      {:returns, %Nx.Tensor{} = tensor} -> {:returns, spell(tensor)}
      {:returns, _other} -> {:returns, nil}
      {:raises, error} -> {:raises, Exception.message(error)}
    end
  end

  # A tensor's shape as the probe program spells it: `{2, 3}[:a, nil]`,
  # then `|:x=2` for each vectorized axis.
  defp spell(%Nx.Tensor{shape: shape, names: names, vectorized_axes: vectorized}) do
    sizes = shape |> Tuple.to_list() |> Enum.map_join(", ", &Integer.to_string/1)
    vectorized = Enum.map_join(vectorized, "", fn {name, size} -> "|#{inspect(name)}=#{size}" end)

    "{#{sizes}}[#{Enum.map_join(names, ", ", &inspect/1)}]" <> vectorized
  end

  # The shapes the probe program derives for what a function returns.
  defp returned_shapes(rows, function),
    do: for([^function, shape] <- Map.get(rows, "returned_shape", []), do: shape)

  # The findings of the shape relations in the lint case's function and in
  # the functions it reaches, as `{relation, kind, detail}`, and of results
  # that can be infinite or NaN in it, as `{relation, kind, cause}`.
  defp lint_findings(rows, index) do
    function = function_id(@lint_fixtures, "lint_#{index}", 4)

    # its closures' too, which a grad differentiates
    closure = "#{inspect(@lint_fixtures)}:-lint_#{index}/4-fun-"
    in_case? = &(&1 == function or String.starts_with?(&1, closure))

    shapes =
      for relation <- ["tensor_shape_mismatch", "tensor_axis_misalignment"],
          found <- reached_findings(rows, relation, function, [:relation, :kind, :detail]),
          uniq: true,
          do: found

    [
      shapes,
      findings_for(rows, "tensor_nonfinite_result", in_case?, [:relation, :kind, :cause]),
      findings_for(rows, "tensor_type_error", function, [:relation, :kind, :subject]),
      findings_for(rows, "tensor_call_error", in_case?, [:relation, :kind, :detail])
    ]
    |> Enum.map(&Enum.uniq/1)
    |> Enum.concat()
  end

  # What running the lint case does where `t` is zero, every size variable
  # is 1 and the shapers are one implementation's.
  defp lint_outcome(index) do
    config = Map.new(~w(heads kv_heads dim head_dim hidden rows cols a b)a, &{&1, 1})
    shaper = struct(ArgusNxTensorAnalyses.TensorShapesTest.Wide)

    outcome_on_binary_backend(@lint_fixtures, :"lint_#{index}", [
      config,
      Nx.iota({1}),
      shaper,
      shaper
    ])
  end

  # A `{:finds, ...}` subject: `:any`, or the finding's own.
  defp subject_matches?(:any, _found_subject), do: true
  defp subject_matches?(subject, found_subject), do: subject == found_subject

  # What running lint case `index` does, as `{:finds, ...}` names it:
  # `:raises`, `:nonfinite` where its result holds an infinity or a NaN,
  # `:finite` otherwise; `:accepted` is anything but `:raises`, `:any`
  # anything at all.
  defp assert_outcome(:any, _index), do: :ok

  defp assert_outcome(:accepted, index),
    do:
      refute(
        classify_outcome(lint_outcome(index), :raises) == :raises,
        "expected Nx to accept lint #{index}"
      )

  defp assert_outcome(outcome, index),
    do: assert(classify_outcome(lint_outcome(index), :raises) == outcome)

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
