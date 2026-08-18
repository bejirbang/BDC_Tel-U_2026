import pandas as pd

df_trigrams = pd.read_csv("analysis/distinctive_patterns/distinctive_trigrams.csv")

top = (
    df_trigrams.sort_values(["emotion", "distinctiveness"], ascending=[True, False])
    .groupby("emotion")
    .head(15)
)

print(top.to_string(index=False))


df_bigrams = pd.read_csv("analysis/distinctive_patterns/distinctive_bigrams.csv")

top = (
    df_bigrams.sort_values(["emotion", "distinctiveness"], ascending=[True, False])
    .groupby("emotion")
    .head(15)
)

print(top.to_string(index=False))


df_words = pd.read_csv("analysis/distinctive_patterns/distinctive_words.csv")

top = (
    df_words.sort_values(["emotion", "distinctiveness"], ascending=[True, False])
    .groupby("emotion")
    .head(20)
)

print(top.to_string(index=False))