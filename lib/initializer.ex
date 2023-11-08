defmodule Initializer do
  @moduledoc """
  Initialization functions for weight matrices.
  """

  @doc """
  Initializes a weight matrix with zeros."
  """
  @spec zero(non_neg_integer, non_neg_integer) :: [[float]]
  def zero(f_in, f_out) do
    initialize_data(f_in, f_out, 0.0)
  end

  @doc """
  Initializes a weight matrix using Glorot (Xavier) initialization."
  """
  @spec glorot(non_neg_integer, non_neg_integer) :: [[float]]
  def glorot(f_in, f_out) do
    initialize_data(f_in, f_out, :math.sqrt(2.0 / (f_in + f_out)))
  end

  @doc """
  Initializes a weight matrix using He initialization."
  """
  @spec he(non_neg_integer, non_neg_integer) :: [[float]]
  def he(f_in, f_out) do
    initialize_data(f_in, f_out, :math.sqrt(2.0 / f_in))
  end

  @doc """
  Initializes a weight matrix using LeCun initialization."
  """
  @spec lecun(non_neg_integer, non_neg_integer) :: [[float]]
  def lecun(f_in, f_out) do
    initialize_data(f_in, f_out, :math.sqrt(1.0 / f_in))
  end

  @doc """
  Initializes a sparse weight matrix with given sparsity."
  """
  @spec sparse(non_neg_integer, non_neg_integer, float) :: [[float]]
  def sparse(f_in, f_out, sparsity) when sparsity >= 0.0 and sparsity <= 1.0 do
    initialize_data(f_in, f_out, fn ->
      if :rand.uniform() < sparsity, do: 0.0, else: :rand.uniform()
    end)
  end

  @doc """
  Initializes identity matrix."
  """
  @spec identity(non_neg_integer) :: [[float]]
  def identity(f_in) do
    for row <- 1..f_in do
      for col <- 1..f_in do
        if row == col, do: 1.0, else: 0.0
      end
    end
  end

  @doc """
  Initializes weight matrix with uniform random values in a given range."
  """
  @spec uniform(non_neg_integer, non_neg_integer, float) :: [[float]]
  def uniform(f_in, f_out, range) when is_number(range) and range >= 0.0 do
    initialize_data(f_in, f_out, fn ->
      :rand.uniform() * range
    end)
  end

  # Initializes a weight matrix with the given standard deviation.
  @spec initialize_data(non_neg_integer, non_neg_integer, float) :: [[float]]
  defp initialize_data(f_in, f_out, std_dev) when is_float(std_dev) do
    for _ <- 1..f_in, do: initialize_row(f_out, std_dev)
  end

  # Initializes a weight matrix with the given function.
  @spec initialize_data(non_neg_integer, non_neg_integer, function) :: [[float]]
  defp initialize_data(f_in, f_out, func) when is_function(func) do
    for _ <- 1..f_in, do: initialize_row(f_out, func)
  end

  # Initializes a row with the given standard deviation.
  @spec initialize_row(non_neg_integer, float) :: [float]
  defp initialize_row(f_out, std_dev) when is_float(std_dev) do
    for _ <- 1..f_out, do: :rand.uniform() * std_dev
  end

  # Initializes a row with the given function.
  @spec initialize_row(non_neg_integer, function) :: [float]
  defp initialize_row(f_out, func) when is_function(func) do
    for _ <- 1..f_out, do: func.()
  end
end
