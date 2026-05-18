class LanguageScore {
    String slug
    BigDecimal probability

    String format() {
        "${slug}=${probability.setScale(2, BigDecimal.ROUND_HALF_UP)}"
    }
}

def scores = [
    new LanguageScore(slug: 'rust', probability: 0.75G),
    new LanguageScore(slug: 'python', probability: 0.25G),
]

scores
    .collect { it.format() }
    .each { println it }
