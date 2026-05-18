class LanguageScore {
  const LanguageScore({required this.slug, required this.probability});

  final String slug;
  final double probability;
}

List<LanguageScore> normalizeScores(List<LanguageScore> scores) {
  final total = scores.fold<double>(
    0,
    (sum, score) => sum + score.probability,
  );

  return [
    for (final score in scores)
      LanguageScore(
        slug: score.slug,
        probability: score.probability / total,
      ),
  ];
}

void main() {
  final scores = normalizeScores([
    const LanguageScore(slug: 'rust', probability: 0.75),
    const LanguageScore(slug: 'python', probability: 0.25),
  ]);

  for (final score in scores) {
    print('${score.slug}=${score.probability.toStringAsFixed(2)}');
  }
}
