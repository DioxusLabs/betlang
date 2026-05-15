module BetlangFixture

export summarize

struct LanguageScore
    name::String
    probability::Float64
end

function summarize(scores::Vector{LanguageScore})
    total = sum(score.probability for score in scores)
    Dict(score.name => score.probability / total for score in scores)
end

scores = [
    LanguageScore("rust", 0.7),
    LanguageScore("python", 0.3),
]

println(summarize(scores))

end
