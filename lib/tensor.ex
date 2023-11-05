defmodule Tensor do
  defstruct dims: [], data: []

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
  """
  @spec zeros(dims :: [integer]) :: %Tensor{}
  def zeros(dims) when is_list(dims) do
    new(dims, gen_zeros(dims))
  end

  defp gen_zeros([n]) when is_integer(n) and n > 0 do
    List.duplicate(0, n)
  end

  defp gen_zeros([head | tail]) when is_integer(head) and head > 0 do
    List.duplicate(gen_zeros(tail), head)
  end

  @doc """
  Creates a new tensor.

  ## Examples

    iex> Tensor.new([2, 3], [[1, 2, 3], [4, 5, 6]])
    %Tensor{dims: [2, 3], data: [[1, 2, 3], [4, 5, 6]]}

  """
  @spec new(data :: [number]) :: %Tensor{}
  def new(data) when is_list(data) do
    dims = infer_dims(data)
    %Tensor{dims: dims, data: data}
  end

  def new(dims, data) when is_list(data) do
    %Tensor{dims: dims, data: data}
  end

  @doc """
  Adds two tensors element-wise.

  ## Examples

    iex> t1 = Tensor.new([1, 3], [[1, 2, 3]])
    iex> t2 = Tensor.new([1, 3], [[1, 1, 1]])
    iex> Tensor.add(t1, t2)
    %Tensor{dims: [1, 3], data: [2, 3, 4]}

  """
  @spec add(t1 :: %Tensor{}, t2 :: %Tensor{}) :: %Tensor{}
  def add(t1, t2) when is_tensor(t1, t2) do
    same_dims?(t1, t2)

    data = Enum.zip(t1.data, t2.data) |> Enum.map(fn {a, b} -> a + b end)
    new(data)
  end

  @doc """
  Subtracts two tensors element-wise.

  ## Examples

    iex> t1 = Tensor.new([1, 3], [[1, 2, 3]])
    iex> t2 = Tensor.new([1, 3], [[1, 1, 1]])
    iex> Tensor.sub(t1, t2)
    %Tensor{dims: [1, 3], data: [0, 1, 2]}

  """
  @spec sub(t1 :: %Tensor{}, t2 :: %Tensor{}) :: %Tensor{}
  def sub(t1, t2) when is_tensor(t1, t2) do
    same_dims?(t1, t2)

    data = Enum.zip(t1.data, t2.data) |> Enum.map(fn {a, b} -> a - b end)
    new(data)
  end

  @doc """
  Multiplies two tensors element-wise.

  ## Examples

    iex> t1 = Tensor.new([1, 3], [[1, 2, 3]])
    iex> t2 = Tensor.new([1, 3], [[2, 3, 4]])
    iex> Tensor.mul(t1, t2)
    %Tensor{dims: [1, 3], data: [2, 6, 12]}

  """

  @spec mul(t1 :: %Tensor{}, t2 :: %Tensor{}) :: %Tensor{}
  def mul(t1, t2) when is_tensor(t1, t2) do
    same_dims?(t1, t2)

    data = Enum.zip(t1.data, t2.data) |> Enum.map(fn {a, b} -> a * b end)
    new(data)
  end

  @doc """
  Divides two tensors element-wise.

  ## Examples

    iex> t1 = Tensor.new([1, 3], [[4, 6, 8]])
    iex> t2 = Tensor.new([1, 3], [[2, 3, 4]])
    iex> Tensor.div(t1, t2)
    %Tensor{dims: [1, 3], data: [2, 2, 2]}

  """
  @spec div(t1 :: %Tensor{}, t2 :: %Tensor{}) :: %Tensor{}
  def div(t1, t2) when is_tensor(t1, t2) do
    same_dims?(t1, t2)

    data = Enum.zip(t1.data, t2.data) |> Enum.map(fn {a, b} -> a / b end)
    new(data)
  end

  @doc """
  Matrix multiplies two tensors.

  ## Examples

    iex> t1 = Tensor.new([2, 3], [[1, 2, 3], [4, 5, 6]])
    iex> t2 = Tensor.new([3, 2], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    iex> Tensor.matmul(t1, t2)
    %Tensor{dims: [2, 2], data: [[58.0, 64.0], [139.0, 154.0]]}

  """
  @spec matmul(t1 :: %Tensor{}, t2 :: %Tensor{}) :: %Tensor{}
  def matmul(t1, t2) when is_tensor(t1, t2) do
    valid_for_matmul?(t1, t2)

    {rows1, cols1} = {Enum.at(t1.dims, 0), Enum.at(t1.dims, 1)}
    {_rows2, cols2} = {Enum.at(t2.dims, 0), Enum.at(t2.dims, 1)}

    data =
      for i <- 0..(rows1 - 1) do
        for j <- 0..(cols2 - 1) do
          Enum.reduce(0..(cols1 - 1), 0, fn k, acc ->
            acc + Enum.at(Enum.at(t1.data, i), k) * Enum.at(Enum.at(t2.data, k), j)
          end)
        end
      end

    new(data)
  end

  # Infer the dimensions of a nested list.
  @spec infer_dims(data :: [[number]]) :: [integer]
  defp infer_dims(data) do
    rows = length(data)
    cols = length(Enum.at(data, 0))
    [rows, cols]
  end

  # Checks if two tensors have the same dimensions.
  defp same_dims?(a, b) do
    if a.dims != b.dims do
      raise ArgumentError, message: "Tensors must have the same dimensions"
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
