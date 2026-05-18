defmodule Betlang.Fixture do
  defstruct [:name, count: 0]

  def new(name) when is_binary(name) do
    %__MODULE__{name: String.trim(name), count: 1}
  end

  def greet(%__MODULE__{name: name, count: count}) do
    Enum.map_join(1..count, "\n", fn _ -> "hello #{name}" end)
  end
end

IO.puts(Betlang.Fixture.new("world") |> Betlang.Fixture.greet())
