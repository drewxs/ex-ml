defmodule Activation do
  @moduledoc """
  Activation functions and their derivatives.
  """

  @type t :: Tensor.t()

  import Tensor, only: [is_tensor: 1]

  @doc """
  Computes the ReLU activation function on a number or tensor.

  ## Examples

    iex> Activation.relu(0.5)
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
  Computes the derivative of the ReLU activation function on a number or tensor.

  ## Examples

    iex> Activation.d_relu(0.5)
    1.0
    iex> Activation.d_relu(Tensor.new([[0.0, 1.0], [-1.0, 2.0]]))
    %Tensor{dims: [2, 2], data: [[0.0, 1.0], [0.0, 1.0]]}

  """
  @spec d_relu(number) :: number
  def d_relu(x) when is_number(x) do
    case x > 0.0 do
      true -> 1.0
      false -> 0.0
    end
  end

  @spec d_relu(t) :: t
  def d_relu(t) when is_tensor(t) do
    Tensor.map(t, &d_relu/1)
  end

  @doc """
  Computes the Leaky ReLU activation function on a number or tensor.
  
  ## Examples

    iex> Activation.leaky_relu(0.5)
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
  Computes the derivative of the Leaky ReLU activation function on a number or tensor.

  ## Examples

    iex> Activation.d_leaky_relu(0.5)
    1.0
    iex> Activation.d_leaky_relu(Tensor.new([[0.0, 1.0], [-1.0, 2.0]]))
    %Tensor{dims: [2, 2], data: [[0.01, 1.0], [0.01, 1.0]]}

  """
  @spec d_leaky_relu(number) :: number
  def d_leaky_relu(x) when is_number(x) do
    case x > 0.0 do
      true -> 1.0
      false -> 0.01
    end
  end

  @spec d_leaky_relu(t) :: t
  def d_leaky_relu(t) when is_tensor(t) do
    Tensor.map(t, &d_leaky_relu/1)
  end

  @doc """
  Computes the Sigmoid activation function on a number or tensor.

  ## Examples

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
  Computes the derivative of the Sigmoid activation function on a number or tensor.

  ## Examples

    iex> Activation.d_sigmoid(0.0)
    0.25
    iex> Activation.d_sigmoid(Tensor.new([[0.0, 1.0], [2.0, 3.0]]))
    %Tensor{dims: [2, 2], data: [[0.25, 0.19661193324148185], [0.10499358540350662, 0.045176659730912]]}

  """
  @spec d_sigmoid(number) :: number
  def d_sigmoid(x) when is_number(x) do
    y = sigmoid(x)
    y * (1.0 - y)
  end

  @spec d_sigmoid(t) :: t
  def d_sigmoid(t) when is_tensor(t) do
    Tensor.map(t, &d_sigmoid/1)
  end

  @doc """
  Computes the TanH activation function on a number or tensor.

  ## Examples

    iex> Activation.tanh(0.5)
    0.46211715726000974
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

  @doc """
  Computes the derivative of the TanH activation function on a number or tensor.

  ## Examples

    iex> Activation.d_tanh(0.5)
    0.7864477329659274
    iex> Activation.d_tanh(Tensor.new([[0.0, 1.0], [-1.0, 2.0]]))
    %Tensor{dims: [2, 2], data: [[1.0, 0.41997434161402614], [0.41997434161402614, 0.07065082485316443]]}

  """
  @spec d_tanh(number) :: number
  def d_tanh(x) when is_number(x) do
    1.0 - :math.pow(tanh(x), 2)
  end

  @spec d_tanh(t) :: t
  def d_tanh(t) when is_tensor(t) do
    Tensor.map(t, &d_tanh/1)
  end
end
