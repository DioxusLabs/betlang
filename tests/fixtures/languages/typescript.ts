export interface LanguageScore {
  readonly slug: string;
  readonly probability: number;
}

export type ScoreMap = Map<string, LanguageScore>;

export function normalize(scores: LanguageScore[]): LanguageScore[] {
  const total: number = scores.reduce(
    (sum: number, score: LanguageScore) => sum + score.probability,
    0,
  );
  return scores.map((score: LanguageScore): LanguageScore => ({
    slug: score.slug,
    probability: score.probability / total,
  }));
}

const scores: LanguageScore[] = [
  { slug: "rust", probability: 0.75 },
  { slug: "python", probability: 0.25 },
];

console.log(normalize(scores));
