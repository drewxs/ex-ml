defmodule Tensor do
  @moduledoc """
  Module for working with n-dimensional tensors.
  """

  @enforce_keys [:dims, :data]
  defstruct dims: [1, 1], data: [[0.0]]

  @typedoc "An n-dimensional matrix of numbers."
  @type t :: %Tensor{
          dims: [integer],
          data: [[number]]
        }

  @doc """
  Checks if the given value is a tensor.
  """
  defguard is_tensor(t) when is_struct(t, Tensor)

  @doc """
  Checks if two values are tensors.
  """
  defguard is_tensor(t1, t2) when is_tensor(t1) and is_tensor(t2)

  @doc """
  Creates a new tensor of zeros.

  ## Examples

    iex> Tensor.zeros([2, 3])
    %Tensor{dims: [2, 3], data: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}

  """
  @spec zeros([integer]) :: t
  def zeros(dims) when is_list(dims) do
    new(dims, gen_matrix(dims, 0.0))
  end

  @doc """
  Creates a new tensor of ones.

  ## Examples

    iex> Tensor.ones([2, 3])
    %Tensor{dims: [2, 3], data: [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]}

  """
  @spec ones([integer]) :: t
  def ones(dims) when is_list(dims) do
    new(dims, gen_matrix(dims, 1.0))
  end

  @doc """
  Generates a matrix with the specified dimensions and fills it with the provided value.

  ## Examples

    iex> Tensor.gen_matrix([2, 3], 0.0)
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    iex> t1 = Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    iex> Tensor.new(Tensor.gen_matrix(t1.dims, 0.0))
    %Tensor{dims: [2, 3], data: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}

  """
  @spec gen_matrix([integer], number) :: [[number]]
  def gen_matrix([n], v) when is_integer(n) and n > 0 do
    List.duplicate(v, n)
  end

  @spec gen_matrix([integer | integer], number) :: [[number]]
  def gen_matrix([head | tail], v) when is_integer(head) and head > 0 do
    List.duplicate(gen_matrix(tail, v), head)
  end

  @doc """
  Creates a new tensor.

  ## Examples

    iex> Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    %Tensor{dims: [2, 3], data: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
    iex> Tensor.new(Initializer.identity(2))
    %Tensor{dims: [2, 2], data: [[1.0, 0.0], [0.0, 1.0]]}

  """
  @spec new([number]) :: t
  def new(data) when is_list(data) do
    dims = infer_dims(data)
    %Tensor{dims: dims, data: data}
  end

  @spec new([integer], [number]) :: t
  def new(dims, data) when is_list(data) do
    %Tensor{dims: dims, data: data}
  end

  @doc """
  Element-wise add a tensor with another tensor or number.

  ## Examples

    iex> t1 = Tensor.new([[1.0, 2.0, 3.0]])
    iex> t2 = Tensor.new([[1.0, 1.0, 1.0]])
    iex> Tensor.add(t1, t2)
    %Tensor{dims: [1, 3], data: [[2.0, 3.0, 4.0]]}
    iex> Tensor.add(t1, 2)
    %Tensor{dims: [1, 3], data: [[3.0, 4.0, 5.0]]}

  """
  @spec add(t, t) :: t
  def add(t1, t2) when is_tensor(t1, t2) do
    valid_2d?(t1, t2)
    same_dims?(t1, t2)

    {rows, cols} = shape(t1)

    data =
      for i <- 0..(rows - 1) do
        for j <- 0..(cols - 1) do
          Tensor.at(t1, i, j) + Tensor.at(t2, i, j)
        end
      end

    new(data)
  end

  @spec add(t, number) :: t
  def add(t, x) when is_tensor(t) and is_number(x) do
    valid_2d?(t)

    {rows, cols} = shape(t)

    data =
      for i <- 0..(rows - 1) do
        for j <- 0..(cols - 1) do
          Tensor.at(t, i, j) + x
        end
      end

    new(data)
  end

  @doc """
  Element-wise subtract a tensor with another tensor or number.

  ## Examples

    iex> t1 = Tensor.new([[1.0, 2.0, 3.0]])
    iex> t2 = Tensor.new([[1.0, 1.0, 1.0]])
    iex> Tensor.sub(t1, t2)
    %Tensor{dims: [1, 3], data: [[0.0, 1.0, 2.0]]}
    iex> Tensor.sub(t1, 1)
    %Tensor{dims: [1, 3], data: [[0.0, 1.0, 2.0]]}

  """
  @spec sub(t, t) :: t
  def sub(t1, t2) when is_tensor(t1, t2) do
    valid_2d?(t1, t2)
    same_dims?(t1, t2)

    {rows, cols} = shape(t1)

    data =
      for i <- 0..(rows - 1) do
        for j <- 0..(cols - 1) do
          Tensor.at(t1, i, j) - Tensor.at(t2, i, j)
        end
      end

    new(data)
  end

  @spec sub(t, number) :: t
  def sub(t, x) when is_tensor(t) and is_number(x) do
    valid_2d?(t)

    {rows, cols} = shape(t)

    data =
      for i <- 0..(rows - 1) do
        for j <- 0..(cols - 1) do
          Tensor.at(t, i, j) - x
        end
      end

    new(data)
  end

  @doc """
  Element-wise multiply a tensor with another tensor or number.

  ## Examples

    iex> t1 = Tensor.new([[1.0, 2.0, 3.0]])
    iex> t2 = Tensor.new([[2.0, 3.0, 4.0]])
    iex> Tensor.mul(t1, t2)
    %Tensor{dims: [1, 3], data: [[2.0, 6.0, 12.0]]}
    iex> Tensor.mul(t1, 2)
    %Tensor{dims: [1, 3], data: [[2.0, 4.0, 6.0]]}

  """
  @spec mul(t, t) :: t
  def mul(t1, t2) when is_tensor(t1, t2) do
    valid_2d?(t1, t2)
    same_dims?(t1, t2)

    {rows, cols} = shape(t1)

    data =
      for i <- 0..(rows - 1) do
        for j <- 0..(cols - 1) do
          Tensor.at(t1, i, j) * Tensor.at(t2, i, j)
        end
      end

    new(data)
  end

  @spec mul(t, number) :: t
  def mul(t, x) when is_tensor(t) and is_number(x) do
    valid_2d?(t)
    valid_2d?(t)

    {rows, cols} = shape(t)

    data =
      for i <- 0..(rows - 1) do
        for j <- 0..(cols - 1) do
          Tensor.at(t, i, j) * x
        end
      end

    new(data)
  end

  @doc """
  Element-wise divide a tensor with another tensor or number.

  ## Examples

    iex> t1 = Tensor.new([[4.0, 6.0, 8.0]])
    iex> t2 = Tensor.new([[2.0, 3.0, 4.0]])
    iex> Tensor.div(t1, t2)
    %Tensor{dims: [1, 3], data: [[2.0, 2.0, 2.0]]}
    iex> Tensor.div(t1, 2)
    %Tensor{dims: [1, 3], data: [[2.0, 3.0, 4.0]]}

  """
  @spec div(t, t) :: t
  def div(t1, t2) when is_tensor(t1, t2) do
    valid_2d?(t1, t2)
    same_dims?(t1, t2)

    {rows, cols} = shape(t1)

    data =
      for i <- 0..(rows - 1) do
        for j <- 0..(cols - 1) do
          Tensor.at(t1, i, j) / Tensor.at(t2, i, j)
        end
      end

    new(data)
  end

  @spec div(t, t) :: t
  def div(t, x) when is_tensor(t) and is_number(x) do
    valid_2d?(t)

    {rows, cols} = shape(t)

    data =
      for i <- 0..(rows - 1) do
        for j <- 0..(cols - 1) do
          Tensor.at(t, i, j) / x
        end
      end

    new(data)
  end

  @doc """
  Matrix multiplies two tensors.

  ## Examples

    iex> t1 = Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    iex> t2 = Tensor.new([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    iex> Tensor.matmul(t1, t2)
    %Tensor{dims: [2, 2], data: [[58.0, 64.0], [139.0, 154.0]]}

  """
  @spec matmul(t, t) :: t
  def matmul(t1, t2) when is_tensor(t1, t2) do
    valid_2d?(t1, t2)
    valid_for_matmul?(t1, t2)

    {t1_rows, t1_cols} = shape(t1)
    {_, t2_cols} = shape(t2)

    data =
      for i <- 0..(t1_rows - 1) do
        for j <- 0..(t2_cols - 1) do
          Enum.reduce(0..(t1_cols - 1), 0, fn k, acc ->
            acc + Tensor.at(t1, i, k) * Tensor.at(t2, k, j)
          end)
        end
      end

    new(data)
  end

  @doc """
  Matrix multiplication for sparse/dense matrices.

  ## Examples

    iex> t1 = Tensor.new([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    iex> t2 = Tensor.new([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    iex> Tensor.sparse_matmul(t1, t2)
    [[1.0, 2.0], [0.0, 0.0]]

  """
  def sparse_matmul(t1, t2) when is_tensor(t1, t2) do
    valid_2d?(t1, t2)
    valid_for_matmul?(t1, t2)

    sparse_t1 = sparse(t1)
    sparse_t2 = sparse(t2)

    {t1_rows, t1_cols} = Tensor.shape(t1)
    {_, t2_cols} = Tensor.shape(t2)

    data =
      for i <- 0..(t1_rows - 1) do
        for j <- 0..(t2_cols - 1) do
          Enum.reduce(0..(t1_cols - 1), 0, fn k, acc ->
            acc + Map.get(sparse_t1, {i, k}, 0) * Map.get(sparse_t2, {k, j}, 0)
          end)
        end
      end

    data
  end

  @doc """
  Returns the transpose of a tensor.

  ## Examples

    iex> t = Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    iex> Tensor.transpose(t)
    %Tensor{dims: [3, 2], data: [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]}

  """
  @spec transpose(t) :: t
  def transpose(t) when is_tensor(t) do
    valid_2d?(t)

    {rows, cols} = shape(t)

    data =
      for i <- 0..(cols - 1) do
        for j <- 0..(rows - 1) do
          Tensor.at(t, j, i)
        end
      end

    new([cols, rows], data)
  end

  @doc """
  Returns a new tensor with the given tensor flattened.

  ## Examples

    iex> t = Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    iex> Tensor.flatten(t)
    %Tensor{dims: [1, 6], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}

  """
  @spec flatten(t) :: t
  def flatten(t) when is_tensor(t) do
    valid_2d?(t)

    {rows, cols} = shape(t)

    data = Enum.reduce(t.data, [], fn x, acc -> acc ++ x end)

    new([1, rows * cols], data)
  end

  @doc """
  Returns a new tensor with a function applied to each element.

  ## Examples

    iex> t = Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    iex> Tensor.map(t, fn x -> x * 2 end)
    %Tensor{dims: [2, 3], data: [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]}

  """
  @spec map(t, (number -> number)) :: t
  def map(t, f) when is_tensor(t) do
    valid_2d?(t)

    {rows, cols} = shape(t)

    data =
      for i <- 0..(rows - 1) do
        for j <- 0..(cols - 1) do
          f.(Tensor.at(t, i, j))
        end
      end

    new(data)
  end

  @doc """
  Returns a new tensor with each element raised to the given power.

  ## Examples

    iex> t = Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    iex> Tensor.pow(t, 3)
    %Tensor{dims: [2, 3], data: [[1.0, 8.0, 27.0], [64.0, 125.0, 216.0]]}

  """
  @spec pow(t, number) :: t
  def pow(t, p) when is_tensor(t) do
    Tensor.map(t, fn x -> :math.pow(x, p) end)
  end

  @doc """
  Returns a new tensor with each element squared.

  ## Examples

    iex> t = Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    iex> Tensor.square(t)
    %Tensor{dims: [2, 3], data: [[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]]}

  """
  @spec square(t) :: t
  def square(t) when is_tensor(t) do
    Tensor.pow(t, 2)
  end

  @doc """
  Returns a new tensor with the square root of each element.

  ## Examples

    iex> t = Tensor.new([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]])
    iex> Tensor.sqrt(t)
    %Tensor{dims: [2, 3], data: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}

  """
  @spec sqrt(t) :: t
  def sqrt(t) when is_tensor(t) do
    Tensor.map(t, fn x -> :math.sqrt(x) end)
  end

  @doc """
  Returns a new tensor with the natural logarithm of each element.

  ## Examples

    iex> t = Tensor.new([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]])
    iex> Tensor.ln(t)
    %Tensor{dims: [2, 3], data: [[0.0, 1.3862943611198906, 2.1972245773362196], [2.772588722239781, 3.2188758248682006, 3.58351893845611]]}

  """
  @spec ln(t) :: t
  def ln(t) when is_tensor(t) do
    Tensor.map(t, fn x -> :math.log(x) end)
  end

  @doc """
  Returns a clipped tensor with all elements between the given min and max.

  ## Examples

    iex> t = Tensor.new([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]])
    iex> Tensor.clip(t, 5.0, 20.0)
    %Tensor{dims: [2, 3], data: [[5.0, 5.0, 9.0], [16.0, 20.0, 20.0]]}

  """
  @spec clip(t, number, number) :: t
  def clip(t, min, max) when is_tensor(t) and is_number(min) and is_number(max) do
    Tensor.map(t, fn x ->
      cond do
        x < min -> min
        x > max -> max
        true -> x
      end
    end)
  end

  @doc """
  Returns the size of a tensor.

  ## Examples

      iex> t = Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      iex> Tensor.size(t)
      6

  """
  @spec size(t) :: integer
  def size(t) when is_tensor(t) do
    Enum.reduce(t.dims, 1, fn x, acc -> x * acc end)
  end

  @doc """
  Returns the element at the given index.

  ## Examples

    iex> t = Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    iex> Tensor.at(t, 1, 0)
    4.0

  """
  @spec at(t, integer, integer) :: number
  def at(t, i, j) when is_tensor(t) do
    valid_2d?(t)

    Enum.at(Enum.at(t.data, i), j)
  end

  @doc """
  Returns the first element of a tensor.

  ## Examples

      iex> t = Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      iex> Tensor.first(t)
      1.0

  """
  @spec first(t) :: number
  def first(t) when is_tensor(t) do
    valid_2d?(t)

    Enum.at(Enum.at(t.data, 0), 0)
  end

  @doc """
  Returns the shape of a tensor.

  ## Examples

    iex> t = Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    iex> Tensor.shape(t)
    {2, 3}

  """
  @spec shape(t) :: {integer, integer}
  def shape(t) when is_tensor(t) do
    valid_2d?(t)

    {Enum.at(t.dims, 0), Enum.at(t.dims, 1)}
  end

  @doc """
  Returns a map of all non-zero elements in a tensor.

  ## Examples

    iex> t = Tensor.new([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]])
    iex> Tensor.sparse(t)
    %{{0, 0} => 1.0, {0, 2} => 3.0, {1, 1} => 5.0}

  """
  @spec sparse(t) :: map
  def sparse(t) when is_tensor(t) do
    valid_2d?(t)

    {rows, cols} = shape(t)

    non_zero_elements =
      for i <- 0..(rows - 1),
          j <- 0..(cols - 1),
          value = Tensor.at(t, i, j),
          value != 0 do
        {{i, j}, value}
      end

    Map.new(non_zero_elements)
  end

  # Infer the dimensions of a nested list.
  @spec infer_dims([[number]]) :: [integer]
  defp infer_dims(data) do
    rows = length(data)
    cols = length(Enum.at(data, 0))

    [rows, cols]
  end

  # Checks if two tensors have the same dimensions.
  defp same_dims?(t1, t2) do
    if t1.dims != t2.dims do
      raise ArgumentError, message: "Tensors must have the same dimensions"
    end
  end

  # Checks if a tensor is 2-dimensional.
  defp valid_2d?(t) when is_tensor(t) do
    if length(t.dims) != 2 do
      raise ArgumentError, message: "Tensor must be 2-dimensional"
    end
  end

  defp valid_2d?(t1, t2) when is_tensor(t1, t2) do
    if length(t1.dims) != 2 and length(t2.dims) != 2 do
      raise ArgumentError, message: "Tensor must be 2-dimensional"
    end
  end

  # Checks if two tensors are compatible for matrix multiplication.
  defp valid_for_matmul?(a, b) do
    if length(a.dims) != 2 or length(b.dims) != 2 do
      raise ArgumentError, message: "Tensors must be 2-dimensional"
    end

    if Enum.at(a.dims, 1) != Enum.at(b.dims, 0) do
      raise ArgumentError, message: "Tensors must be compatible for matrix multiplication"
    end
  end
end
