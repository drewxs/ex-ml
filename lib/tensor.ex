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
  """
  @spec zeros([integer]) :: t
  def zeros(dims) when is_list(dims) do
    new(dims, gen_matrix(dims, 0))
  end

  @doc """
  Creates a new tensor of ones.
  """
  @spec ones([integer]) :: t
  def ones(dims) when is_list(dims) do
    new(dims, gen_matrix(dims, 1))
  end

  @doc """
  Generates a matrix with the specified dimensions and fills it with the provided value.

  ## Examples

    iex> Tensor.gen_matrix([2, 3], 0.0)
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    iex> t1 = Tensor.new([[1, 2, 3], [4, 5, 6]])
    iex> t2 = Tensor.new(Tensor.gen_matrix(t1.dims, 0.0))
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

    iex> Tensor.new([2, 3], [[1, 2, 3], [4, 5, 6]])
    %Tensor{dims: [2, 3], data: [[1, 2, 3], [4, 5, 6]]}
    iex> Tensor.new(Initializer.identity(2))
    %Tensor{dims: [2, 2], data: [[1, 0], [0, 1]]}

  """
  @spec new([number]) :: t
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
  @spec add(t, t) :: t
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
  @spec sub(t, t) :: t
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

  @spec mul(t, t) :: t
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
  @spec div(t, t) :: t
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
  @spec matmul(t, t) :: t
  def matmul(t1, t2) when is_tensor(t1, t2) do
    valid_for_matmul?(t1, t2)

    {t1_rows, t1_cols} = {Enum.at(t1.dims, 0), Enum.at(t1.dims, 1)}
    {_t2_rows, t2_cols} = {Enum.at(t2.dims, 0), Enum.at(t2.dims, 1)}

    data =
      for i <- 0..(t1_rows - 1) do
        for j <- 0..(t2_cols - 1) do
          Enum.reduce(0..(t1_cols - 1), 0, fn k, acc ->
            acc + Enum.at(Enum.at(t1.data, i), k) * Enum.at(Enum.at(t2.data, k), j)
          end)
        end
      end

    new(data)
  end

  @doc """
  Returns the size of a tensor.

  ## Examples

      iex> t = Tensor.new([2, 3], [[1, 2, 3], [4, 5, 6]])
      iex> Tensor.size(t)
      6

  """
  @spec size(t :: t) :: integer
  def size(t) when is_tensor(t) do
    Enum.reduce(t.dims, 1, fn x, acc -> x * acc end)
  end

  @doc """
  Returns the first element of a tensor.

  ## Examples

      iex> t = Tensor.new([2, 3], [[1, 2, 3], [4, 5, 6]])
      iex> Tensor.first(t)
      1

  """
  @spec first(t :: t) :: number
  def first(t) when is_tensor(t) do
    Enum.at(Enum.at(t.data, 0), 0)
  end

  # Infer the dimensions of a nested list.
  @spec infer_dims([[number]]) :: [integer]
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
