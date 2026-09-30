# Checks

This is the reference for the findings `argus_nx_tensor_analyses` reports:
what each kind means, how severe it is, and code that triggers it.

## Reading a finding

Severity:

- **error**: Nx raises, or the result is wrong, on every path the analysis
  sees.
- **warning**: it can happen. Nx raises on some paths only, or it runs the
  call and computes something the code likely does not mean.
- **info**: it depends on an input nothing checks, or on values, sizes or
  types the code does not show.

A finding is **certain** when some chain of calls reaching the call brings
it only values that break the rule. It is **on some path** when the call
also gets values that are fine, along branches the analysis cannot tell
apart, so the bad combination may never run. A kind listed as an error is
reported as a warning when its finding is on some path: a mismatch whose
text says "on some path", a type error whose operand is a float or
complex only on some paths, or a container that "can hold" a bad value.

Each finding carries a kind, the string in the tables below. `tensor_shapes`
reports its findings in `tensor_shape_mismatch`,
`tensor_axis_misalignment`, `tensor_nonfinite_result`, `tensor_type_error`
and `tensor_call_error`, and `tensor_emlx` in `tensor_emlx_divergence` and
`tensor_emlx_mixed_backends`.

In the examples, `t` is a tensor and `config` a map that the function is
handed, whose contents the analysis cannot see. `key` is a random key.

## `tensor_shapes`

### Shapes

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `broadcast` | error | Shapes that do not broadcast. | `Nx.add(Nx.iota({2, 3}), Nx.iota({2}))` |
| `axis` | error | An axis the operand does not have, or one listed twice. | `Nx.sum(Nx.iota({2, 3}), axes: [2])` |
| `dot` | error | Contracted or batch axes of different sizes, or a batch axis that is also contracted. | `Nx.dot(Nx.iota({4, 8}), Nx.iota({6, 16}))` |
| `reshape` | error | A new shape with another element count, or a size below 1. | `Nx.reshape(Nx.iota({2, 3}), {4, 2})` |
| `names` | error | Axes of different names that meet, or names that are not one per axis or that repeat. | `Nx.add(Nx.iota({2}, names: [:x]), Nx.iota({2}, names: [:y]))` |
| `concatenate` | error | Joined tensors that differ on an axis other than the one joined. | `Nx.concatenate([Nx.iota({2, 3}), Nx.iota({2, 4})])` |
| `stack` | error | Stacked tensors of different shapes. | `Nx.stack([Nx.iota({2}), Nx.iota({3})])` |
| `squeeze` | error | Squeezing an axis whose size is not 1. | `Nx.squeeze(Nx.iota({2, 3}), axes: [0])` |
| `transpose` | error | A permutation that does not name every axis once. | `Nx.transpose(Nx.iota({2, 3}), axes: [0])` |
| `flatten` | error | Flattening axes that are not neighbors. | `Nx.flatten(Nx.iota({2, 3, 4}), axes: [0, 2])` |
| `slice` | error | Starts, lengths or strides not one per axis, or a length past its axis. | `Nx.slice(Nx.iota({4}), [0], [5])` |
| `put_slice` | error | A slice larger than the tensor or of another rank, or starts not one per axis. | `Nx.put_slice(Nx.iota({2}), [0], Nx.iota({3}))` |
| `pad` | error | Padding not one `{low, high, interior}` per axis, or negative interior padding. | `Nx.pad(Nx.iota({2, 3}), 0, [{1, 1, 0}])` |
| `split` | error | A split point not strictly inside the axis. | `Nx.split(Nx.iota({4}), 4)` |
| `rank` | error | An operand of a rank the operation does not take. | `Nx.tril(Nx.iota({3}))` |
| `square` | error | A matrix that is not square where the operation needs one. | `Nx.LinAlg.determinant(Nx.iota({2, 3}, type: :f32))` |
| `solve` | error | A right-hand side that does not fit the system. | `Nx.LinAlg.solve(Nx.eye(3), Nx.iota({2}, type: :f32))` |
| `least_squares` | error | A right-hand side with other rows than the system. | `Nx.LinAlg.least_squares(Nx.iota({3, 2}, type: :f32), Nx.iota({4}, type: :f32))` |
| `diagonal` | error | An offset that leaves the diagonal outside the matrix, or a diagonal of another length than the one its offset picks. | `Nx.put_diagonal(Nx.iota({3, 3}), Nx.iota({2}))` |
| `diff` | error | A difference order at or past the axis's size. | `Nx.diff(Nx.iota({3}), order: 3)` |
| `gather` | error | Indices whose last axis addresses more axes than the tensor has. | `Nx.gather(Nx.iota({3}), Nx.iota({2, 2}))` |
| `indexed` | error | Indices or updates of an indexed update that do not fit the tensor. | `Nx.indexed_add(Nx.iota({3}), Nx.iota({2, 1}), Nx.iota({3}))` |
| `take_along_axis` | error | Indices that differ from the tensor on an axis not taken along. | `Nx.take_along_axis(Nx.iota({2, 3}), Nx.iota({3, 3}), axis: 1)` |
| `top_k` | error | A `k` past the last axis's size. | `Nx.top_k(Nx.iota({3}), k: 4)` |
| `conv` | error | Input and kernel of different ranks or channels, a kernel larger than the padded input, or a negative stride that leaves no window. | `Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 2, 2, 2}))` |
| `window` | error | Window sizes or strides not one per axis, a window larger than the padded axis, or a negative stride that leaves no window. | `Nx.window_sum(Nx.iota({4}), {5})` |
| `linspace` | error | Start and stop of different shapes, or no `:n`. | `Nx.linspace(Nx.iota({2}), Nx.iota({3}), n: 3)` |
| `weighted_mean` | error | Weights that do not fit the input once Nx reshapes and swaps them. | `Nx.weighted_mean(Nx.iota({2, 3}), Nx.iota({2}), axes: [1])` |
| `vectorize` | error | Vectorizing axes the tensor lacks, or under a name it already uses. | `Nx.vectorize(Nx.iota({2, 3}), x: 3)` |
| `vectorized_axes` | error | Vectorized axes of one name and different sizes. | `Nx.add(Nx.vectorize(Nx.iota({2}), :x), Nx.vectorize(Nx.iota({3}), :x))` |
| `scalar` | error | A tensor with axes, or a vectorized one, where Nx takes a scalar. | `Nx.to_number(Nx.iota({2}))` |
| `needs_axes` | error | A scalar handed to `Nx.to_list/1`, `Nx.to_heatmap/2` or `Nx.to_batched/3`. | `Nx.to_list(Nx.sum(Nx.iota({3})))` |
| `batch` | error | `Nx.to_batched/3` batches larger than the leading axis. | `Nx.to_batched(Nx.iota({2, 3}), 3)` |
| `batch_size` | error | A batch size `Nx.to_batched/3` has no clause for: below 1, a float or an atom. | `Nx.to_batched(Nx.iota({2}), 0)` |
| `nonpositive` | error | A dimension, repetition, `k`, slice stride, FFT length or `:n` below 1. | `Nx.iota({0, 3})` |
| `no_tensors` | error | Concatenating or stacking an empty list. | `Nx.concatenate([])` |
| `tensor_as_shape` | error | `Nx.iota/2` handed a tensor where it takes a shape. | `Nx.iota(Nx.iota({2}))` |
| `ragged_data` | error | Literal data whose lists differ in length, or mix lists and numbers. | `Nx.tensor([[1, 2], [3]])` |
| `cropped_axis` | error | Negative padding that crops an axis to nothing. | `Nx.pad(Nx.iota({2, 3}), 0, [{0, 0, 0}, {0, -3, 0}])` |
| `irfft` | error | An `Nx.irfft/2` length below 3, given or implied by an axis of size 1 or 2. | `Nx.irfft(Nx.iota({3}, type: :f32), length: 2)` |
| `key` | error | An `Nx.Random` key of a shape other than `{2}`, such as what `Nx.Random.split/2` returns. | `Nx.Random.uniform(Nx.Random.split(key))` |
| `sampler_parameters` | error | Sampler bounds, mean or standard deviation that do not broadcast into `:shape`. | `Nx.Random.uniform(key, Nx.iota({3}, type: :f32), 5.0, shape: {2})` |
| `choice` | error | Choosing from a scalar, fewer than one sample, more samples than elements without replacement, or probabilities that do not fit. | `Nx.Random.choice(key, Nx.iota({4}), samples: 5, replace: false)` |
| `multivariate_normal` | error | A mean that is not a vector, or a covariance that is not square or not of the mean's size. | `Nx.Random.multivariate_normal(key, Nx.tensor([0.0, 0.0]), Nx.eye(3))` |
| `norm` | error | An `Nx.LinAlg.norm/2` order the tensor's rank does not take. | `Nx.LinAlg.norm(Nx.iota({3}, type: :f32), ord: :frobenius)` |
| `scalar_iota` | warning | `Nx.iota/2` of a number, which is the scalar 0, not a range. | `Nx.iota(5)` |
| `odd_irfft` | warning | `Nx.irfft/2` with no `:length` of an odd-length signal's `Nx.rfft/2`, which rebuilds one element short. | `Nx.irfft(Nx.rfft(Nx.iota({5}, type: :f32)))` |
| `transposed_weights` | warning | Weights over two or more axes, which `Nx.weighted_mean/3` applies transposed. | `Nx.weighted_mean(Nx.iota({5, 3, 2}), Nx.iota({2, 3}), axes: [1, 2])` |
| `scaling_factor_rank` | warning | A `logsumexp` scaling factor of more axes than the tensor, which sums over other axes. | `Nx.logsumexp(Nx.iota({3}, type: :f32), axes: [0], exp_scaling_factor: Nx.iota({2, 3}, type: :f32))` |
| `squeeze_input_size` | warning | `Nx.squeeze/1` with no `:axes` over an axis sized by an input's shape, which it removes where that size is 1 (a batch of one). | `Nx.squeeze(Nx.iota({Nx.axis_size(t, 0), 3}))` |

A shape the analysis cannot follow gives no finding. `Nx.template/2` of a
size 0 is not reported: Nx builds it.

### Axis alignment

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `size_variables` | warning | Sizes the code names differently that meet, which fit only while they happen to be equal. | `Nx.add(Nx.iota({config.heads}), Nx.iota({config.kv_heads}))` |
| `unnamed_axis` | warning | An axis with no name meeting a named one. | `Nx.add(Nx.iota({2, 3}, names: [:rows, :cols]), Nx.iota({2, 3}))` |
| `contracted_names` | warning | Contracting axes of different names. | `Nx.dot(Nx.iota({2, 3}, names: [:rows, :cols]), [:cols], Nx.iota({3, 4}, names: [:inner, :out]), [:inner])` |
| `vectorize_name` | warning | Vectorizing a named axis under another name. | `Nx.vectorize(Nx.iota({2, 3}, names: [:rows, :cols]), :heads)` |
| `reshape_order` | warning | A reshape that moves a size past another axis's, where a transpose was meant: heads split out ahead of the sequence, merged back without a transpose, or whole axes listed in another order. | `Nx.reshape(Nx.iota({config.seq, config.heads * config.dim}), {config.heads, config.seq, config.dim})` |

A size only an input's shape decides (`Nx.axis_size(t, 0)`) is none of the
code's variables and meets anything. `reshape_order` leaves out splits and
merges in place, literal factors (a pixel shuffle), and a reshape with a
size it does not know or a variable on several axes of one side.

### Values that turn infinite or NaN

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `divide_by_zero` | warning | A divisor that cannot be negative but can be zero (a sum of squares, a norm, a count, an index, a spread), in a division, quotient, remainder, `rsqrt` or negative power. | `Nx.divide(t, Nx.LinAlg.norm(t))` |
| `log_of_zero` | warning | A logarithm of such a value, or of a softmax written out, a sigmoid of a value that can be negative, a product of many fractions, a sample drawn from zero, or a `logsumexp` scaling factor that can be all zero. | `Nx.log(Nx.sum(Nx.greater(t, 0)))` |
| `root_of_negative` | warning | A square root of a difference of two values that cannot be negative, which rounding takes below zero (a variance as E[x²] - E[x]²). | `Nx.sqrt(Nx.subtract(Nx.mean(Nx.multiply(t, t)), Nx.pow(Nx.mean(t), 2)))` |
| `log_of_negative` | warning | A logarithm of such a difference. | `Nx.log(Nx.subtract(1, Nx.multiply(t, t)))` |
| `exp_overflow` | warning | A softmax or log-sum-exp over values not shifted by their maximum, and a softplus or logistic written out. | `Nx.divide(Nx.exp(t), Nx.sum(Nx.exp(t)))` |
| `outside_domain` | warning | asin, acos, atanh, erf_inv, log1p or acosh of a value its math takes outside the domain, including a cosine similarity that rounding takes past ±1. | `Nx.asin(Nx.multiply(2, Nx.tanh(t)))` |
| `infinite_at_edge` | warning | atanh or erf_inv of a value its math takes to ±1, or log1p of one it takes to -1. | `Nx.atanh(Nx.tanh(t))` |
| `log_base_one` | warning | `Nx.log/2` whose base can be exactly 1. | `Nx.log(Nx.exp(t), Nx.clip(t, 1, 5))` |
| `nan_comparison` | warning | A comparison with NaN, which is always 0 (always 1 for `not_equal`). | `Nx.equal(t, Nx.Constants.nan())` |

Math that keeps the operand away (an epsilon added, `Nx.max` with a
positive number, a clip inside the domain) stays quiet, and so does a
check on the way: a test such as `if n > 0`, or
`Nx.select(Nx.equal(d, 0), 1, d)`. An operand that only an input makes
zero is an unchecked operand instead. A sum, mean or norm is zero only
where every element is, so the zero an iota starts with, or an identity
matrix holds off its diagonal, does not make it zero.

### Unchecked operands

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `unchecked_divisor` | info | A divisor from an input, or a sum or difference whose terms can cancel, that no test and no `Nx.select` keeps from zero. | `Nx.divide(t, config.heads)` |
| `unchecked_logarithm` | info | A logarithm of such a value, or of one that can be negative. | `Nx.log(t)` |
| `unchecked_root` | info | A square root of a value nothing keeps from going negative. | `Nx.sqrt(t)` |
| `unchecked_domain` | info | asin, acos, atanh, erf_inv, log1p or acosh of a value no clip and no test keeps in the domain. | `Nx.asin(t)` |
| `unchecked_cast_wraparound` | info | A cast to an unsigned type of a value only an input can make negative. | `Nx.as_type(Nx.round(Nx.multiply(t, 255)), :u8)` |

A test has to check the operand itself: `if n > 0` checks
`Nx.divide(t, n)`, not `Nx.divide(t, Nx.multiply(t, n))`. A call with a
definite finding gets no unchecked one. A sample of `Nx.Random.uniform` or
`randint` has its bounds' signs, so the logarithm of
`uniform(key, -1.0, 1.0)` is unchecked.

### Types and literals

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `non_integer_operand` | error | A float or complex operand where Nx takes integers: bitwise operations, `quotient`, and the indices of `take`, `gather` and the indexed updates. | `Nx.take(t, Nx.divide(t, 2))` |
| `unsupported_type` | error | A tensor made in a type listed in `unsupported_types`. A call whose options Nx rejects raises before it makes one, and gets only the options finding. | `Nx.as_type(t, :f64)`, with `unsupported_types: [:f64]` |
| `complex_operand` | error | A complex operand to a call that orders or rounds values (`argmax`, `sort`, comparisons, `clip`, `floor`, `Nx.LinAlg.svd`, ...). | `Nx.argmax(Nx.fft(t))` |
| `complex_spread` | warning | A variance or standard deviation of complex values, which squares them rather than their magnitudes. | `Nx.variance(Nx.fft(t))` |
| `integer_past_s32` | warning | An integer literal outside s32 where Nx types it s32, so it wraps. | `Nx.add(Nx.tensor(0, type: :s64), 1_700_000_000_000)` |
| `atom_type_rejected` | error | A short atom type (`:f32`) where Nx takes only the tuple form: most `Nx.Type` functions, and `Nx.Random.gumbel`'s `:type`. | `Nx.Type.float?(:f32)` |
| `atom_type_misread` | warning | A short atom type that an `Nx.Type` function answers as if for another type. | `Nx.Type.to_complex(:f64)` |
| `invalid_type` | error | A type Nx does not have. | `Nx.iota({2}, type: :float32)` |
| `literal_wraps` | warning | Literal integer data outside the integer type written at the call. | `Nx.tensor([300], type: :u8)` |
| `literal_overflows` | warning | Literal float data past the type's largest value, which becomes infinite. f8 overflows from 65520. | `Nx.tensor(70_000.0, type: :f16)` |
| `literal_flushes` | warning | Literal float data too near zero for the type, which becomes 0.0. f8 flushes below 1.5229e-5. | `Nx.tensor(1.0e-8, type: :f16)` |
| `float_as_integer` | error | A float, `:nan` or an infinity as data of an integer type. | `Nx.tensor(1.5, type: :s32)` |
| `unsigned_wraparound` | warning | A subtraction, negation or `diff` whose unsigned result can go below zero, `all_close` of unsigned tensors, or a descending unsigned `linspace`. | `Nx.subtract(Nx.greater(t, 0), 1)` |
| `count_wraparound` | warning; info for a length the code does not write | Zeros and ones counted past the type's limit by a running or window sum, or a `dot`. | `Nx.cumulative_sum(Nx.greater(Nx.iota({300}), -1))` |
| `narrow_wraparound` | warning | 8-bit or narrower integers summed or multiplied in their own type (`dot`, `outer`, running and window operations, `weighted_mean`, `median` with no axis). | `Nx.dot(Nx.s8([100, 100]), Nx.s8([2, 2]))` |
| `index_wraparound` | warning; info for a length the code does not write | An `argmax`, `argmin` or `argsort` `:type` too small for the largest index. | `Nx.argmax(Nx.iota({300}), type: :u8)` |
| `sequence_precision` | warning; info for a length the code does not write | An iota or linspace made in, or brought into, a type that cannot hold its whole numbers. | `Nx.iota({1000}, type: :bf16)` |
| `cast_wraparound` | warning | A cast to an unsigned type of a value the code's math makes negative, or a written number outside an integer type. | `Nx.as_type(Nx.subtract(Nx.iota({3}), 1), :u8)` |
| `complex_to_real` | warning | A cast from complex to real, which drops the imaginary part. | `Nx.as_type(Nx.c64([1]), :f32)` |
| `float_truncation` | info | A cast from float to integer with no rounding first. | `Nx.as_type(Nx.divide(Nx.iota({3}), 2), :s32)` |
| `literal_overflow` | warning | A written number past the tensor's float range where the tensor keeps its type (a `-1.0e9` mask on f16), or past f32's where Nx makes it an f32 before it meets an f64 or c128 tensor, outside traced code. | `Nx.add(Nx.f16([1, 2]), -1.0e9)` |
| `literal_underflow` | warning | A written number the tensor's float type rounds to zero (an epsilon of `1.0e-12` on f16), or f32 does where Nx makes it an f32 before it meets an f64 or c128 tensor, outside traced code. | `Nx.add(Nx.as_type(t, :f16), 1.0e-12)` |
| `cast_overflow` | warning | A cast to f16 or f8 of a value holding a number past the type's range. | `Nx.as_type(Nx.tensor(-1.0e9), :f16)` |
| `float_sum_overflow` | warning; info for a length the code does not write | A sum, mean, variance or `dot` of f16 or f8 over 65520 or more elements. | `Nx.sum(Nx.broadcast(Nx.f16(1.0), {70000}))` |
| `unsigned_logsumexp` | warning | `logsumexp` of an unsigned tensor, which wraps around subtracting the maximum. | `Nx.logsumexp(Nx.u8([1, 2, 3]))` |
| `upcast` | warning | A lower-precision float made f32 by an f32 constant, a float bound of `clip`, `put_slice`, `log2`, `log10`, `Nx.log/2` or `invert`; an integer widened by `clip`'s bounds. | `Nx.multiply(Nx.bf16([1]), Nx.Constants.pi())` |
| `narrowing_merge` | warning | A merge into a type that holds less than an operand (bf16 with f16 is f16, u64 with a signed type is s64). | `Nx.add(Nx.bf16([1.0e5]), Nx.f16([1]))` |
| `pad_type_mismatch` | warning | A pad value of another type than the tensor. | `Nx.pad(Nx.tensor([1]), 0.5, [{1, 0, 0}])` |
| `integer_determinant` | warning for unsigned; info for signed | A 2×2 or 3×3 integer determinant, computed in the matrix's type. | `Nx.LinAlg.determinant(Nx.u8([[1, 2], [3, 4]]))` |
| `f32_precision` | info | `log2`, `log10` or `Nx.log/2` of f64, accurate only to f32. | `Nx.log2(Nx.f64([2.0]))` |
| `constant_type` | error | An `Nx.Constants` value in a type that has none (NaN in s32, `i` in f32, `max_finite` in c64). | `Nx.Constants.nan(:s32)` |
| `integer_negative_power` | warning | An integer power whose exponent can be negative. | `Nx.pow(Nx.add(Nx.iota({3}), 1), -1)` |
| `rounded_logarithm` | warning | A floor, ceiling or integer cast of a base-2 or base-10 logarithm, off by one at exact powers. | `Nx.floor(Nx.log2(Nx.tensor([8192, 32768])))` |

A type the code does not show is not known and gives no finding; list the
float types in `float_types` to check types the code reads from
configuration. A call that breaks a check in several of them is one
finding, which names each (`f16/bf16`); an `upcast` or `narrowing_merge`
is one for each operand, and a `pad_type_mismatch` whose pad value is of
such a type one for each type.
Literals are read by their spelling, so integers of any size compare
exactly, and a value known only at run time is not checked.

### Options

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `unknown_option` | error | An option key the function does not take. The finding names the key likely meant. | `Nx.sum(t, axis: 0)` |
| `options_not_keyword` | error | Options that are a list but not a keyword list. | `Nx.transpose(t, [1, 0])` |
| `option_form` | error | `:axes` as one axis, or `:axis` as a list. | `Nx.sum(t, axes: 1)` |
| `option_value` | error | An atom an option does not take (`direction`, `tie_break`, `padding`, `mode`, `transform_a`, an FFT `length`, ...), nil or false for `max_iter`, or an `Nx.pad_outer/3` padding type Nx lacks. | `Nx.sort(t, direction: :descending)` |
| `conv_options` | error | Conv strides, dilations, padding or permutations not one per axis, a stride of 0, a dilation below 1, or group sizes that do not divide the batch or the kernel. | `Nx.conv(Nx.iota({1, 3, 5, 5}), Nx.iota({4, 3, 2, 2}), strides: [1])` |
| `window_options` | error | A window given as a list or with a size below 1, strides, dilations or padding not one per axis, a stride of 0, or a dilation below 1. | `Nx.window_sum(Nx.iota({4, 4}), [2, 2])` |
| `norm_axes` | warning | A matrix order of `Nx.LinAlg.norm/2` given `:axes`, which still returns one norm of the whole matrix. | `Nx.LinAlg.norm(Nx.iota({2, 3}, type: :f32), ord: 1, axes: [0])` |

Options are read where the code writes them, in the keyword lists it
builds, and in a literal list a caller hands down. A list a caller hands
down is reported at the caller's call that hands it (`sums(t, axis: 0)`,
where `sums/2` calls `Nx.sum(t, opts)`), with a frame at the Nx call
that raises for it. A value the code tests the type of before it builds
the pair
(`if is_list(axes), do: axes, else: [axes]`) is not taken for the value.
`Nx.reshape/3` does not check its options, so it is not reported.

### Indices, slices and ranges

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `negative_index` | warning | An index to `take`, `take_along_axis`, `gather` or an indexed update that the code's math can make negative (one less than an index, a written `-100`, a remainder of a negative). Backends disagree on what it does. | `Nx.take(Nx.iota({3}), Nx.subtract(t, 1))` |
| `negative_slice_start` | warning | A slice start that is written negative or that the math can make negative. Nx moves it to 0. | `Nx.slice_along_axis(Nx.iota({6}), -1, 1)` |
| `slice_past_end` | warning | A start and length that certainly run past the axis. Nx moves the start back. | `Nx.slice(Nx.iota({6}), [5], [3])` |
| `non_integer_start` | error | A float slice start. | `Nx.slice(Nx.iota({6}), [1.0], [2])` |
| `ddof_not_below_count` | warning | A written `ddof` at or past the count a variance, standard deviation or covariance reduces over. | `Nx.variance(Nx.iota({1, 2}, type: :f32), axes: [0], ddof: 1)` |
| `negative_ddof` | warning | A negative `ddof`. | `Nx.variance(t, ddof: -1)` |
| `random_range_empty` | error | `Nx.Random.randint` with equal bounds: the maximum is exclusive. | `Nx.Random.randint(key, 5, 5)` |
| `random_range_reversed` | warning | `randint` or `uniform` with the minimum above the maximum. | `Nx.Random.randint(key, 10, 1)` |
| `random_range_outside_type` | warning | Written bounds the given type cannot hold, or a span as wide as the type. | `Nx.Random.randint(key, 0, 300, type: :u8)` |
| `random_type_not_integer` | error | `randint` of a float type, given or implied by a float bound. | `Nx.Random.randint(key, 0, 5, type: :f32)` |
| `random_bound_truncated` | warning | A float bound to `randint` of an integer type, which truncates it. | `Nx.Random.randint(key, 0, 2.5, type: :s32)` |

A finding says why the value can be negative and points at the call that
makes it so. Integer arithmetic outside Nx (`Nx.axis_size(t, 0) - 1`) is
not checked.

### Tensor access and tuples

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `access_scalar` | error | Indexing a scalar. | `Nx.sum(Nx.iota({4}))[0]` |
| `access_out_of_bounds` | error | A written index or range bound outside its axis. | `Nx.iota({4, 5})[4]` |
| `access_negative_step` | error | A range that steps backwards, as `1..-1` does, and as `0..(k - 1)` does in a `defn` where `k` is 0: a Kernel operator on two numbers computes a number. | `Nx.iota({4, 5})[1..-1//-1]` |
| `access_empty_range` | error | A range that holds no index of its axis. | `Nx.iota({4, 5})[3..1//1]` |
| `access_too_many_indices` | error | More indices than the tensor has axes. | `Nx.iota({4, 5})[[0, 0, 0]]` |
| `access_unknown_name` | error | A name the tensor has no axis for. | `Nx.iota({4, 5}, names: [:a, :b])[c: 1]` |
| `access_duplicate_name` | error | One axis named twice. | `Nx.iota({4, 5}, names: [:a, :b])[[a: 1, a: 0]]` |
| `access_float_index` | error | A float index. | `Nx.iota({4, 5})[1.0]` |
| `access_float_index_tensor` | error | A float tensor as the index. | `Nx.iota({4, 5})[Nx.divide(Nx.iota({2}), 2)]` |
| `access_tensor_in_list` | error | A tensor with axes in a list of indices. | `Nx.iota({4, 5})[[Nx.iota({2})]]` |
| `access_update` | error | `put_in`, `update_in`, `get_and_update_in` or `pop_in` on a tensor. | `x = Nx.iota({4, 5}); put_in(x[0], Nx.iota({5}))` |
| `access_index_clamped` | warning | A written scalar tensor index outside its axis, which Nx clamps rather than raising. | `Nx.iota({4, 5})[Nx.tensor(7)]` |
| `tuple_as_tensor` | error | A tuple of tensors (from `split`, `top_k`, a sampler, a decomposition, or one the code builds) where a tensor goes. | `Nx.add(Nx.top_k(Nx.iota({4}), k: 2), 1)` |
| `tensors_in_tensor_data` | error | `Nx.tensor/2` given a list that holds tensors. | `Nx.tensor([Nx.sum(Nx.iota({2})), Nx.sum(Nx.iota({3}))])` |

`t[key]` gets the shape Nx gives it, inside `defn` and out, and so do the
elements of the tuples Nx returns (the `Nx.Random` samplers, the
`Nx.LinAlg` decompositions, `Nx.split`), so both meet the shape checks.
The halves of a float `Nx.split` have sizes that are not known.

### Traced code and defn control flow

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `data_read_in_trace` | error | Reading a tensor's data (`to_number`, `to_list`, `to_binary`, ...) in code Nx traces: a `defn`, or a function handed to `Nx.Defn.jit`, `compile`, a grad or `EXLA.jit`. | `Nx.Defn.jit(fn x -> Nx.add(x, Nx.to_number(Nx.sum(x))) end).(t)` |
| `jit_in_trace` | error | A jitted or compiled function, or `jit_apply`, run in traced code with no `on_conflict:` in its options. | `inner = Nx.Defn.jit(fn y -> Nx.add(y, 1) end); Nx.Defn.jit(fn x -> inner.(x) end).(t)` |
| `compiled_template` | error | A compiled function called with an argument its template does not fit: other known sizes, another known type, or an axis both name otherwise. | `Nx.Defn.compile(&Nx.exp/1, [Nx.template({2}, :f32)]).(Nx.iota({2}))` |
| `template_computed` | error | A template computed with, or handed to a jitted function. | `Nx.add(Nx.template({1}, :s32), t)` |
| `captured_tensor` | warning | A closure handed to a jit, compile or grad that captures a tensor. EMLX and EXLA raise. | `data = Nx.iota({1}); Nx.Defn.jit(fn x -> Nx.add(x, data) end).(t)` |
| `tensor_arithmetic` | error | Elixir arithmetic (`+`, `*`, `/`, `div`, `band`, ...) on a tensor outside a `defn`. | `x = Nx.iota({2}); x * x` |
| `tensor_boolean` | error | `not`, `and` or `or` on a tensor. | `not Nx.greater(t, 0)` |
| `tensor_comparison` | warning | A tensor compared as an Elixir term, whose answer does not depend on its data. | `Nx.sum(t) == 0` |
| `tensor_truth` | warning | `if`, `unless`, `&&` or `!` on an Nx comparison or reduction, which is always true. | `if Nx.all(Nx.less(t, 0)), do: t, else: Nx.negate(t)` |
| `predicate_shape` | error | An `if`, `cond` or `while` in a `defn` deciding on a tensor that is not a scalar. | `if x > 0` in a `defn` handed a vector |
| `predicate_value` | error | An `if` or `cond` deciding on nil, an atom or a list, such as a key the options lack. | `if opts[:training]` in a `defn` whose options lack it |
| `branch_shapes` | error | Branches whose shapes do not broadcast. | `if(Nx.all(x > 0), do: Nx.iota({2}), else: Nx.iota({3}))` |
| `branch_broadcast` | warning | Branches that broadcast to a shape neither gives. | `if(Nx.all(x > 0), do: Nx.iota({3}), else: Nx.iota({3, 1}))` |
| `branch_structure` | error | Branches that give a tuple and a tensor, or tuples of different sizes. | `if(Nx.all(x > 0), do: {x, x}, else: x)` |
| `cond_fallthrough` | error | A `cond` on a tensor with no last clause that always holds. | a `cond` ending in `Nx.all(x <= 0) -> -1` |
| `traced_raise` | error | A `raise` in a branch of a condition on a tensor, which Nx builds, so it raises on every call. | `if(Nx.all(x >= 0), do: Nx.sqrt(x), else: raise("negative"))` |
| `while_shape` | error | A `while` body that changes a state element's shape. | `while {i = 0, total = 0.0}, i < 3, do: {i + 1, total + Nx.iota({3})}` |
| `while_type` | error | A `while` body that makes an integer state element a float. | `while {i = 0, total = 0, x}, i < 3, do: {i + 1, total + Nx.sum(x) / 2, x}` |
| `closure_captures_tensor` | error | A `while`, `Nx.reduce` or `Nx.window_reduce` closure that computes with a tensor from outside it. | `Nx.reduce(x, 0, fn a, b -> a + b + y end)` in `defn f(x, y)` |
| `atom_operand` | error | An operator or element-wise function given an atom, or nil from a missing key. | `opts[:mode] == :train` in a `defn` |
| `tensor_as_integer` | error | A public `defn`'s argument, a tensor by then, where Nx needs an integer: a shape, an axis, `:k`, a slice length, a range bound. | `defn f(n), do: Nx.iota({n})` |
| `dropped_print` | warning | A `print_value` or `io_call` whose result nothing uses, so it never runs. | `print_value(x)` on a line of its own |

A value counts as a tensor, expression, template or jitted function only
where the analysis sees it made so. In a `defn`, a key that `keyword!/2`
or `assert_keys/2` names counts as given, a `case` on an atom is fine, and
`runtime_raise/1` raises only when reached. `tensor_as_integer` follows a
public `defn`'s own arguments in its own body, so a `defnp` that other
`defn`s hand integers is not reported.

### Gradients

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `infinite_gradient` | warning | A square root, cube root, root power, norm or standard deviation whose result can be zero, where a grad differentiates it. | `Nx.Defn.grad(t, fn x -> Nx.LinAlg.norm(x) end)` |
| `gradient_at_origin` | warning; info if only inputs make both coordinates zero | `atan2`, or the phase of `Nx.complex`, where both coordinates can be zero. | `Nx.Defn.grad(t, fn x -> Nx.sum(Nx.atan2(Nx.multiply(x, x), Nx.abs(x))) end)` |
| `gradient_at_edge` | warning | asin or acos of a value its math takes to ±1, or acosh of one it takes to 1. | `Nx.Defn.grad(t, fn x -> Nx.sum(Nx.acos(Nx.clip(Nx.add(x, 1), -1, 1))) end)` |
| `masked_gradient` | warning | A logarithm, root or division that a select masks out (the "double where"), whose gradient is NaN where masked. | `Nx.Defn.grad(t, fn x -> Nx.sum(Nx.select(Nx.greater(x, 0), Nx.log(x), 0)) end)` |
| `exponent_gradient` | warning if the base is written negative; info otherwise | A power whose differentiated exponent has a base that can be negative. | `Nx.Defn.grad(t, fn x -> Nx.sum(Nx.pow(-2.0, x)) end)` |
| `sigmoid_gradient_overflow` | warning; info if only past f16's range | A sigmoid of an operand plus a mask far below zero. | `Nx.Defn.grad(t, fn x -> Nx.sum(Nx.sigmoid(Nx.add(x, -1.0e9))) end)` |
| `degenerate_gradient` | warning | svd, pinv, eigh, invert or least_squares of a matrix built degenerate (an outer product, a thin product, a scaled identity), or its norm of `ord: :nuclear` or -2, whose gradient is NaN. | `Nx.Defn.grad(t, fn v -> Nx.sum(Nx.LinAlg.pinv(Nx.outer(v, v))) end)` |
| `singular_gradient` | warning | cholesky, qr, lu or solve of a matrix built singular, whose gradient raises. | `Nx.Defn.grad(t, fn v -> Nx.sum(Nx.LinAlg.cholesky(Nx.outer(v, v))) end)` |
| `no_gradient` | error | `reduce`, `window_reduce`, `window_product`, `quotient`, `div` or `map` that a gradient flows through. | `Nx.Defn.grad(t, fn x -> Nx.reduce(x, 0, fn a, b -> Nx.add(a, b) end) end)` |
| `complex_gradient` | warning | A Fourier transform of a real differentiated value, whose gradient comes back complex. | `Nx.Defn.grad(Nx.as_type(t, :f32), fn x -> Nx.sum(Nx.abs(Nx.rfft(x))) end)` |
| `custom_grad_not_list` | error | A `custom_grad` whose gradient function returns no list. | `custom_grad(Nx.multiply(x, x), [x], fn g -> g end)` |
| `custom_grad_short` | warning | A `custom_grad` that returns fewer gradients than it lists inputs. | `custom_grad(Nx.multiply(x, y), [x, y], fn g -> [g] end)` |
| `custom_grad_unlisted` | warning | A `custom_grad` expression made from a differentiated value it does not list. | `custom_grad(Nx.multiply(a, b), [a], fn g -> [Nx.multiply(g, b)] end)` |
| `gradient_of_container` | error | A grad of a function that returns a tuple or a map. | `Nx.Defn.grad(t, fn x -> {Nx.sum(x), x} end)` |

These follow the values made from the variable a grad differentiates, in
the function it is handed and what that calls. Nothing behind `stop_grad`
is reported, nor the double select that masks the operand too, nor a norm
or root inside a `custom_grad`.

### Containers

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `dropped_field_read` | warning | A `defn`, or a function a `jit` runs, reading a struct field its derived `Nx.Container` lists in neither `containers:` nor `keep:`, which holds its default there. | `if layer.causal` in a `defn` handed `%Layer{weight: t, causal: true}` |
| `dropped_field_returned` | warning | Reading such a field of a struct a `defn` or `jit` returns. | `pass(%Layer{weight: t, causal: true}).causal`, `pass` a `defn` |
| `container_leaf` | error | A container handed to a `defn`, `jit`, `jit_apply` or `Nx.Batch` holding nil, a boolean, an atom, a list, a string, a fun, or a struct with no container implementation. | `Nx.Defn.jit(fn x, _ -> x end).(t, %{w: t, b: nil})` |
| `captured_gradient` | warning | A grad whose function captures the value it differentiates, or part of it, which gets a zero gradient. | `Nx.Defn.grad(t, fn v -> Nx.sum(Nx.multiply(v, t)) end)` |
| `container_order` | warning | A hand-written `Nx.Container` whose `traverse/3` and `reduce/3` visit fields in different orders, reported at the implementation. | `traverse/3` visits `first` then `second`, `reduce/3` the reverse |

A list or keyword list passed as an argument itself is not checked, as Nx
takes it as it is, and a `defn` called while Nx traces traverses nothing.
A struct or container is known only where the code builds it or writes
it as a literal.

### Randomness and spent values

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `reused_key` | warning | Two draws from one key in a function, one after the other on some path: one variable, field or split part, or a helper handed the key twice. | `{a, _} = Nx.Random.uniform(key); {b, _} = Nx.Random.uniform(key)` |
| `reused_seed` | warning | Two keys `Nx.Random.key/1` makes of one written seed, each drawn from in a function, one after the other on some path. | `a = Nx.Random.key(42); b = Nx.Random.key(42); {x, _} = Nx.Random.uniform(a); {y, _} = Nx.Random.uniform(b)` |
| `captured_loop_key` | warning | A loop's function (`for`, `Enum`, `Stream`, a `defn`'s `while`) that draws from a key captured from outside. | `for _ <- 1..3, do: elem(Nx.Random.uniform(key), 0)` |
| `passed_back_key` | warning | A loop's function that draws from its key and hands that same key to the next pass. | `Enum.map_reduce(1..3, key, fn _, key -> {elem(Nx.Random.uniform(key), 0), key} end)` |
| `spent_key_returned` | warning | A function that returns a key it drew from: alone, in a term, or in the state it read it from. | `{sample, _} = Nx.Random.uniform(key); {sample, key}` |
| `shared_draw` | warning | A normal sampler whose mean or standard deviation has more elements than `:shape`, which repeats one draw across them. | `Nx.Random.normal(key, Nx.iota({2, 3}, type: :f32), 1.0)` |
| `used_after_transfer` | error for a read by an Nx call; warning when handed on or returned | A tensor used after `Nx.backend_transfer/1`, or `/2` to `Nx.BinaryBackend`, freed it. | `_ = Nx.backend_transfer(t); Nx.add(t, 1)` |
| `used_after_deallocation` | error for a read by an Nx call; warning when handed on or returned | A tensor used after `Nx.backend_deallocate/1`. | `Nx.backend_deallocate(t); Nx.sum(t)` |
| `used_after_donation` | warning | A tensor marked with `Nx.donatable/1` read, handed on or returned after the jitted call it was donated to. | `_ = doubled.(Nx.donatable(t)); Nx.add(t, 1)` |

`Nx.Random.fold_in/2` is no draw, and draws on two branches are one draw
on either path. EMLX and EXLA raise reading a freed tensor; the
BinaryBackend frees nothing, so tests there pass. A draw inside a `defn`
is checked as one outside it is.

### Servings and batches

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `serving_scalar_output` | error | A serving computation whose output is a scalar, which Nx cannot slice per request. | `Nx.Serving.jit(&Nx.sum/1)`, run on a batch |
| `serving_output_batch_axis` | warning | An output leaf whose first axis is not the batch, so each caller gets part of something else. | `Nx.Serving.jit(fn x -> Nx.transpose(x) end)`, run on a batch of vectors |
| `serving_mixes_batch` | warning | A reduction, sort, cumulative operation or contraction along the batch axis, which mixes requests and padding. | `Nx.Serving.jit(fn x -> Nx.subtract(x, Nx.mean(x, axes: [0])) end)` |
| `serving_template_batch_size` | error; warning for a serving process, whose batches are smaller when they time out | A computation compiled ahead of time for a batch size its batches do not have. | `Nx.Defn.compile(fun, [Nx.template({4, 3}, :s32)], options)`, run on a batch of 1 |
| `serving_template_type` | error | A compiled template of another type than the batch's entries, exactly where both are known (s64 against s32), else of another class (float against integer). | an `:s64` template run on a batch of s32 entries |
| `batch_incompatible_entries` | error | `Nx.Batch` entries of different shapes, ranks or axis names. | `Nx.Batch.stack([Nx.tensor([1, 2]), Nx.tensor([1, 2, 3])])` |
| `batch_scalar_entry` | error | A scalar in `Nx.Batch.concatenate/2`. | `Nx.Batch.concatenate([Nx.tensor(1)])` |
| `batch_empty` | error | A serving run on an empty batch. | `Nx.Serving.run(serving, Nx.Batch.new())` |
| `serving_entry_shape_varies` | warning | A serving process whose preprocessing batches each request as it comes, so requests of different shapes crash the batch and the serving's supervisor. | `client_preprocessing(fn input -> {Nx.Batch.stack([input]), :ok} end)` |
| `serving_run_input` | error | A tensor, or a list of tensors, handed to a serving with no client preprocessing. | `Nx.Serving.run(serving, Nx.tensor([1, 2, 3]))` |
| `serving_run_name` | error | `Nx.Serving.run/2` handed a name, where it takes the serving itself. | `Nx.Serving.run(MyServing, batch)` |
| `serving_batched_run_struct` | error | `Nx.Serving.batched_run/2,3` handed a serving, where it takes a serving process's name. | `Nx.Serving.batched_run(Nx.Serving.jit(&Nx.exp/1), batch)` |
| `serving_batch_size_conflict` | error | A serving process started with another `batch_size:` than the one `Nx.Serving.batch_size/2` sets on its serving. | `{Nx.Serving, serving: Nx.Serving.batch_size(serving, 4), name: MyServing, batch_size: 8}` |
| `serving_preprocessing_result` | error | A client preprocessing that returns no `{batch, info}` pair. | `client_preprocessing(fn input -> Nx.Batch.stack([input]) end)` |
| `serving_builder_result` | error | An `Nx.Serving.new/2` builder that returns no compiled function. | `Nx.Serving.new(fn x -> Nx.multiply(x, 2) end)` |
| `serving_computation_arity` | error | A serving computation of more than one argument. | `Nx.Serving.jit(fn x, y -> Nx.add(x, y) end)` |
| `serving_postprocessing_input` | error | A two-argument client postprocessing that takes `{output, metadata}` for a tensor. | `client_postprocessing(fn output, _info -> Nx.argmax(output) end)` |

Where the code shows a batch's shape (an `Nx.template`, or entries it
builds), the computation is analyzed over that batch; elsewhere only
operations on the batch along its first axis are seen. A computation that
captures values, `fn batch -> predict(params, batch) end`, takes the batch
as its argument. `serving_entry_shape_varies` covers servings started as
processes (`{Nx.Serving, serving: serving, ...}`), and `Nx.Batch.key/2` on
the batch quiets it.

## `tensor_emlx`

`ArgusNxTensorAnalyses.EMLX` reports what [EMLX](https://github.com/elixir-nx/emlx)
computes differently from the BinaryBackend and EXLA. It is not a default
analysis: run `mix argus tensor_emlx`, or add it to `analyses`.

| Kind | Severity | Catches | Example |
|---|---|---|---|
| `narrowed_type` | warning | A tensor made on EMLX in f64 or c128, which EMLX keeps as f32 or c64 while it still says f64 or c128; a c128 raises when read back. Not a call whose options Nx rejects, which makes none. | `Nx.iota({3}, type: :f64)` |
| `narrowed_transfer` | warning | An f64 or c128 tensor made off EMLX that `Nx.backend_transfer/2` or `Nx.backend_copy/2` moves onto it: EMLX keeps an f64 as f32 while it still says f64, and raises taking a c128 in. One cast to f32 first is not reported. | `Nx.backend_transfer(Nx.iota({3}, type: :f64, backend: EXLA.Backend), EMLX.Backend)` |
| `negative_remainder` | warning; info if only an input can be negative | A remainder whose divisor or dividend can be negative, which EMLX computes wrongly. | `Nx.remainder(Nx.subtract(Nx.iota({4}), 2), 3)` |
| `negative_integer_power` | warning; info if only an input can be negative | An integer power whose exponent can be negative, which hangs EMLX's CPU device when run eagerly and gives 0 elsewhere. | `Nx.pow(Nx.iota({3}), -1)` |
| `round_half` | info | `Nx.round` of a value the code's math puts on a half (an integer halved, an integer plus 0.5, a mean of integers), which EMLX rounds to even. | `Nx.round(Nx.divide(Nx.iota({5}), 2))` |
| `wrapped_shift` | warning | A shift by a written amount at or past the width EMLX shifts the type in (32 bits, 64 for u32 and the 64-bit types), which EMLX takes modulo that width where the BinaryBackend and EXLA give 0. | `Nx.left_shift(Nx.tensor(1), 33)` |
| `tensor_emlx_mixed_backends` | warning | An Nx call handed tensors of two backends, neither of them the BinaryBackend. Nx raises. | `Nx.add(Nx.tensor([1.0], backend: EXLA.Backend), Nx.iota({1}))` |

`tensor_emlx_mixed_backends` is a relation's name: its findings carry the
two backends rather than a kind. A tensor's backend is followed from where
the code picks one (`backend:`, `Nx.backend_transfer/2`,
`Nx.with_default_backend/2`, a jit's or compile's `compiler:`, `EXLA.jit`,
`Nx.default_backend/1`), and EMLX is the default otherwise. Code Nx traces
is not checked for mixed backends: the compiler, configured outside the
code, decides. A type listed in `unsupported_types` is reported as
`unsupported_type` by `tensor_shapes`, not as `narrowed_type`.

## Project options

A project configures the analyses in its `mix.exs`, under
`argus_nx_tensor_analyses:` in `project/0`:

```elixir
def project do
  [
    # ...
    argus_nx_tensor_analyses: [
      analyses: [:default, :tensor_emlx],
      unsupported_types: [:f64],
      float_types: [:f16, :bf16, :f32]
    ]
  ]
end
```

- `analyses`: the analyses a run that names none runs, by name
  (`:tensor_shapes`, `:tensor_emlx`) or as the sets `:default`
  (`tensor_shapes`) and `:all`. Default: `[:default]`. `mix argus --all`,
  or analyses named on the command line, take its place.
- `unsupported_types`: the types the backend lacks, as Nx names them
  (`:f64`, `{:f, 64}`). Every call that makes a tensor of one is an
  `unsupported_type` error, and `tensor_emlx` leaves such a type to it.
  Default: none.
- `float_types`: the float types the code may run at. A type the code reads
  from configuration (a `type:` or type argument it does not write) is
  checked as each of them, for kinds such as `literal_overflow`,
  `literal_underflow`, `upcast`, `narrowing_merge` and
  `sequence_precision`. Default: none, and such a type is not known.

[How it works](how-it-works.md) explains how the analysis finds these, and
its limits.
