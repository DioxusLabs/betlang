type LanguageScore = {
  readonly slug: string;
  readonly probability: number;
};

function normalize(scores: LanguageScore[]): LanguageScore[] {
  const total = scores.reduce((sum, score) => sum + score.probability, 0);
  return scores.map((score) => ({
    slug: score.slug,
    probability: score.probability / total,
  }));
}

const scores: LanguageScore[] = [
  { slug: "rust", probability: 0.75 },
  { slug: "python", probability: 0.25 },
];

console.log(normalize(scores));
