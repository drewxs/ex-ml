defmodule Ml.Tensor do
  alias Ml.Tensor, as: Tensor

  @spec new(dims :: [integer], data :: [number]) :: %Tensor{}
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
    data = for _ <- 1..Enum.reduce(dims, 1, &(&1 * &2)), do: 0.0
    new(dims, data)
  end

  @doc """
  Creates a new tensor.

  ## Examples

    iex> Tensor.new([2, 3], [[1, 2, 3], [4, 5, 6]])

  """
  def new(dims, data) when is_list(dims) and is_list(data) do
    if length(dims) == 0 do
      raise ArgumentError, message: "dims cannot be empty"
    end

    total_size = Enum.reduce(dims, 1, &(&1 * &2))
    data_size = length(List.flatten(data))

    if data_size != total_size do
      raise ArgumentError, message: "expected #{total_size} data elements, got #{length(data)}"
    end

    %Tensor{dims: dims, data: data}
  end

  @doc """
  Adds two tensors element-wise.
  """
  @spec add(t1 :: %Tensor{}, t2 :: %Tensor{}) :: %Tensor{}
  def add(t1, t2) when is_tensor(t1, t2) do
    same_dims?(t1, t2)

    new_dims = t1.dims
    new_data = Enum.zip(t1.data, t2.data) |> Enum.map(fn {a, b} -> a + b end)
    new(new_dims, new_data)
  end

  @doc """
  Subtracts two tensors element-wise.
  """
  @spec sub(t1 :: %Tensor{}, t2 :: %Tensor{}) :: %Tensor{}
  def sub(t1, t2) when is_tensor(t1, t2) do
    same_dims?(t1, t2)

    new_dims = t1.dims
    new_data = Enum.zip(t1.data, t2.data) |> Enum.map(fn {a, b} -> a - b end)
    new(new_dims, new_data)
  end

  @doc """
  Multiplies two tensors element-wise.
  """
  @spec mul(t1 :: %Tensor{}, t2 :: %Tensor{}) :: %Tensor{}
  def mul(t1, t2) when is_tensor(t1, t2) do
    same_dims?(t1, t2)

    new_dims = t1.dims
    new_data = Enum.zip(t1.data, t2.data) |> Enum.map(fn {a, b} -> a * b end)
    new(new_dims, new_data)
  end

  @doc """
  Divides two tensors element-wise.
  """
  @spec div(t1 :: %Tensor{}, t2 :: %Tensor{}) :: %Tensor{}
  def div(t1, t2) when is_tensor(t1, t2) do
    same_dims?(t1, t2)

    new_dims = t1.dims
    new_data = Enum.zip(t1.data, t2.data) |> Enum.map(fn {a, b} -> a / b end)
    new(new_dims, new_data)

  @doc """
  Matrix multiplies two tensors.

  ## Examples

    iex> t1 = Ml.Tensor.new([2, 3], [[1, 2, 3], [4, 5, 6]])
    iex> t2 = Ml.Tensor.new([3, 2], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    iex> Ml.Tensor.matmul(t1, t2)
    %Ml.Tensor{dims: [2, 2], data: [[58.0, 64.0], [139.0, 154.0]]}

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
  defp infer_dimensions([], dims) do
    {dims, []}
  end

  defp infer_dimensions([head | tail], dims) when is_list(head) do
    {new_dims, remaining} = infer_dimensions(head, dims)
    infer_dimensions(tail, [length(new_dims) | new_dims] ++ remaining)
  end

  defp infer_dimensions([_ | tail], dims) do
    infer_dimensions(tail, [1 | dims])
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
