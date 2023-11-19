defmodule Activation do
  @type t :: Tensor.t()

  import Tensor, only: [is_tensor: 1]

  @doc """
  Computes the ReLU activation function on a number or tensor.

    ie> Activation.relu(0.5)
    0.5
    iex> Activation.relu(Tensor.new([[0.0, 1.0], [-1.0, 2.0]]))
    %Tensor{dims: [2, 2], data: [[0.0, 1.0], [0.0, 2.0]]}

  """
  @spec relu(number) :: number
  def relu(x) when is_number(x) do
    case x > 0.0 do
      true -> x
      false -> 0.0
    end
  end

  @spec relu(t) :: t
  def relu(t) when is_tensor(t) do
    Tensor.map(t, &relu/1)
  end

  @doc """
  Computes the Leaky ReLU activation function on a number or tensor.

    ie> Activation.leaky_relu(0.5)
    0.5
    iex> Activation.leaky_relu(Tensor.new([[0.0, 1.0], [-1.0, 2.0]]))
    %Tensor{dims: [2, 2], data: [[0.0, 1.0], [-0.01, 2.0]]}

  """
  @spec leaky_relu(number) :: number
  def leaky_relu(x) when is_number(x) do
    case x > 0.0 do
      true -> x
      false -> 0.01 * x
    end
  end

  @spec leaky_relu(t) :: t
  def leaky_relu(t) when is_tensor(t) do
    Tensor.map(t, &leaky_relu/1)
  end

  @doc """
  Computes the Sigmoid activation function on a number or tensor.

    iex> Activation.sigmoid(0.0)
    0.5
    iex> Activation.sigmoid(Tensor.new([[0.0, 1.0], [2.0, 3.0]]))
    %Tensor{dims: [2, 2], data: [[0.5, 0.7310585786300049], [0.8807970779778823, 0.9525741268224334]]}

  """
  @spec sigmoid(number) :: number
  def sigmoid(x) when is_number(x) do
    1.0 / (1.0 + :math.exp(-x))
  end

  @spec sigmoid(t) :: t
  def sigmoid(t) when is_tensor(t) do
    Tensor.map(t, &sigmoid/1)
  end

  @doc """
  Computes the TanH activation function on a number or tensor.

    ie> Activation.tanh(0.5)
    0.4621171572600098
    iex> Activation.tanh(Tensor.new([[0.0, 1.0], [-1.0, 2.0]]))
    %Tensor{dims: [2, 2], data: [[0.0, 0.7615941559557649], [-0.7615941559557649, 0.9640275800758169]]}

  """
  @spec tanh(number) :: number
  def tanh(x) when is_number(x) do
    :math.tanh(x)
  end

  @spec tanh(t) :: t
  def tanh(t) when is_tensor(t) do
    Tensor.map(t, &tanh/1)
  end
end
