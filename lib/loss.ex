defmodule Loss do
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
end
