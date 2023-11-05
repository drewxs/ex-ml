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
  end

  defp same_dims?(a, b) do
    if a.dims != b.dims do
      raise ArgumentError, message: "Tensors must have the same dimensions"
    end
  end
end
