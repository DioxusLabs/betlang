class Greeter
  attr_reader :name

  def initialize(name)
    @name = name.strip
  end

  def message
    "hello #{@name}"
  end
end

names = %w[Ada Grace Linus]
names
  .map { |name| Greeter.new(name) }
  .each { |greeter| puts greeter.message }
