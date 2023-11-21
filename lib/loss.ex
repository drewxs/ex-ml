defmodule Loss do
  @moduledoc """
  Loss functions used for training neural networks.
  """

  @type t :: Tensor.t()

  @doc """
  Computes the mean squared error between the predicted and actual values.

  ## Examples

    iex> predicted = Tensor.new([[1.0, 2.0, 3.0]])
    iex> actual = Tensor.new([[2.0, 4.0, 6.0]])
    iex> Loss.mse(predicted, actual)
    %Tensor{dims: [1, 3], data: [[1.0, 4.0, 9.0]]}

  """
  @spec mse(t, t) :: t
  def mse(y_hat, y) do
    Tensor.square(Tensor.sub(y_hat, y))
  end

  @doc """
  Computes the binary cross entropy between the predicted and actual values.

  ## Examples

    iex> y_hat = Tensor.new([[0.9, 0.1, 0.2], [0.1, 0.9, 0.8]])
    iex> y = Tensor.new([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    iex> Loss.bce(y_hat, y)
    %Tensor{dims: [2, 3], data: [[0.10536051565782517, 0.10536051565782739, 0.2231435513142111],
                                 [0.10536051565782739, 0.10536051565782517, 0.22314355131420846]]}

  """
  @spec bce(t, t) :: t
  def bce(y_hat, y) do
    epsilon = 1.0e-15
    ones = Tensor.ones(y_hat.dims)

    y_hat = Tensor.add(y_hat, epsilon)
    y_hat = Tensor.clip(y_hat, epsilon, 1.0 - epsilon)

    y_hat_cmp = Tensor.sub(ones, y_hat)
    y_cmp = Tensor.sub(ones, y)

    term1 = Tensor.mul(y, Tensor.ln(y_hat))
    term2 = Tensor.mul(y_cmp, Tensor.ln(y_hat_cmp))

    Tensor.mul(Tensor.add(term1, term2), -1.0)
  end
end
