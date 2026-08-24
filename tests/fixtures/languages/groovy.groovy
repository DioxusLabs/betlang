package betlang.samples

import groovy.transform.Canonical

@Canonical
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

def bySlug = scores.collectEntries { [(it.slug): it.probability] }
assert bySlug['rust'] > bySlug['python']

scores
    .findAll { it.probability > 0.1G }
    .collect { it.format() }
    .each { println it }

def total = scores*.probability.sum()
println "total=${total}"
