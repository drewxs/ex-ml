defmodule MlTest do
  use ExUnit.Case
  doctest Ml

  test "greets the world" do
    assert Ml.hello() == :world
  end
end
