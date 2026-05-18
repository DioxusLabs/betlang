Gem::Specification.new do |spec|
  spec.name = "betlang-fixture"
  spec.version = "0.1.0"
  spec.summary = "Fixture gemspec for language detection"
  spec.authors = ["Betlang"]
  spec.files = Dir["lib/**/*.rb"]
  spec.required_ruby_version = ">= 3.1"

  spec.add_dependency "json", "~> 2.7"
  spec.add_development_dependency "rake", "~> 13.2"
end
