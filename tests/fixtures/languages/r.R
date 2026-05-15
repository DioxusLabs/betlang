library(stats)

values <- data.frame(
  language = c("rust", "python", "rust", "javascript"),
  score = c(0.75, 0.20, 0.80, 0.05)
)

summary <- aggregate(score ~ language, data = values, FUN = mean)
summary$rank <- rank(-summary$score, ties.method = "first")

model <- lm(score ~ rank, data = summary)
print(summary)
print(coef(model))
