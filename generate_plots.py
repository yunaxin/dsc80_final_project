import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance


# -----------------------------
# Setup
# -----------------------------
os.makedirs("assets", exist_ok=True)
np.random.seed(42)

tracks = pd.read_csv("data/music_tracks.csv")


# -----------------------------
# Data cleaning
# -----------------------------
cleaned = tracks.copy()

relevant_cols = [
    "track_id",
    "popularity",
    "track_genre",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "explicit",
    "artists",
    "duration_ms",
    "release_date"
]

cleaned = cleaned[relevant_cols]

selected_genres = [
    "classical",
    "hip-hop",
    "country",
    "electronic",
    "metal",
    "pop"
]

cleaned = cleaned[cleaned["track_genre"].isin(selected_genres)].copy()

cleaned["release_date"] = pd.to_datetime(
    cleaned["release_date"].astype(str),
    format="mixed",
    errors="coerce"
)

cleaned["release_year"] = cleaned["release_date"].dt.year
cleaned["duration_min"] = cleaned["duration_ms"] / 60000

popularity_cutoff = 55
cleaned["is_popular"] = cleaned["popularity"] >= popularity_cutoff

cleaned["tempo_missing"] = cleaned["tempo"].isna()

cleaned["num_artists"] = (
    cleaned["artists"]
    .fillna("")
    .str.split(";")
    .str.len()
)

cleaned["tempo_filled"] = cleaned.groupby("track_genre")["tempo"].transform(
    lambda x: x.fillna(x.median())
)

cleaned["loudness_energy_ratio"] = cleaned["loudness"] / (cleaned["energy"] + 1e-6)


# -----------------------------
# Helper functions
# -----------------------------
def save_plot(fig, filename):
    fig.write_html(f"assets/{filename}.html")


def tvd(dist1, dist2):
    all_cats = dist1.index.union(dist2.index)
    dist1 = dist1.reindex(all_cats, fill_value=0)
    dist2 = dist2.reindex(all_cats, fill_value=0)
    return 0.5 * np.abs(dist1 - dist2).sum()


# -----------------------------
# Univariate plots
# -----------------------------
fig = px.histogram(
    cleaned,
    x="popularity",
    nbins=20,
    title="Popularity Distribution"
)
save_plot(fig, "popularity_distribution")

fig = px.histogram(
    cleaned,
    x="danceability",
    nbins=20,
    title="Danceability Distribution"
)
save_plot(fig, "danceability_distribution")

fig = px.histogram(
    cleaned,
    x="energy",
    nbins=30,
    title="Distribution of Spotify Track Energy"
)
save_plot(fig, "energy_distribution")

fig = px.histogram(
    cleaned,
    x="explicit",
    title="Explicit vs Non-Explicit Tracks"
)
save_plot(fig, "explicit_distribution")

fig = px.box(
    cleaned,
    y="duration_min",
    title="Distribution of Track Duration"
)
save_plot(fig, "duration_distribution")


# -----------------------------
# Bivariate plots
# -----------------------------
fig = px.scatter(
    cleaned,
    x="danceability",
    y="popularity",
    title="Danceability vs Popularity",
    opacity=0.2
)
save_plot(fig, "danceability_popularity")

fig = px.box(
    cleaned,
    x="explicit",
    y="popularity",
    title="Popularity by Explicit Content"
)
save_plot(fig, "explicit_popularity")

fig = px.box(
    cleaned,
    x="track_genre",
    y="popularity",
    title="Popularity Distribution by Genre"
)
save_plot(fig, "genre_popularity")

fig = px.box(
    cleaned,
    x="track_genre",
    y="danceability",
    title="Danceability Distribution by Genre"
)
save_plot(fig, "danceability_by_genre")

fig = px.scatter(
    cleaned,
    x="energy",
    y="popularity",
    color="track_genre",
    opacity=0.5,
    title="Energy vs. Popularity by Genre"
)
save_plot(fig, "energy_popularity")


# -----------------------------
# Missingness permutation test 1:
# tempo_missing vs track_genre
# -----------------------------
missing_dist = cleaned[cleaned["tempo_missing"]]["track_genre"].value_counts(normalize=True)
not_missing_dist = cleaned[~cleaned["tempo_missing"]]["track_genre"].value_counts(normalize=True)

observed_tvd = tvd(missing_dist, not_missing_dist)

simulated_tvds = []

for _ in range(1000):
    shuffled = np.random.permutation(cleaned["tempo_missing"])
    temp = cleaned.copy()
    temp["shuffled_missing"] = shuffled

    dist1 = temp[temp["shuffled_missing"]]["track_genre"].value_counts(normalize=True)
    dist2 = temp[~temp["shuffled_missing"]]["track_genre"].value_counts(normalize=True)

    simulated_tvds.append(tvd(dist1, dist2))

fig = px.histogram(
    x=simulated_tvds,
    nbins=30,
    title="Permutation Test: Tempo Missingness and Track Genre",
    labels={"x": "Simulated TVD", "y": "Count"}
)
fig.add_vline(x=observed_tvd, line_dash="dash")
save_plot(fig, "permutation_test_1")


# -----------------------------
# Missingness permutation test 2:
# tempo_missing vs release_year
# -----------------------------
observed_diff = abs(
    cleaned[cleaned["tempo_missing"]]["release_year"].mean()
    - cleaned[~cleaned["tempo_missing"]]["release_year"].mean()
)

simulated_diffs = []

for _ in range(1000):
    shuffled = np.random.permutation(cleaned["tempo_missing"])
    temp = cleaned.copy()
    temp["shuffled_missing"] = shuffled

    diff = abs(
        temp[temp["shuffled_missing"]]["release_year"].mean()
        - temp[~temp["shuffled_missing"]]["release_year"].mean()
    )

    simulated_diffs.append(diff)

fig = px.histogram(
    x=simulated_diffs,
    nbins=30,
    title="Permutation Test: Tempo Missingness and Release Year",
    labels={"x": "Absolute Difference in Mean Release Year", "y": "Count"}
)
fig.add_vline(x=observed_diff, line_dash="dash")
save_plot(fig, "permutation_test_2")


# -----------------------------
# Hypothesis test:
# track_genre vs is_popular
# -----------------------------
popular_dist = cleaned[cleaned["is_popular"]]["track_genre"].value_counts(normalize=True)
not_popular_dist = cleaned[~cleaned["is_popular"]]["track_genre"].value_counts(normalize=True)

observed_genre_tvd = tvd(popular_dist, not_popular_dist)

simulated_genre_tvds = []

for _ in range(1000):
    shuffled = np.random.permutation(cleaned["is_popular"])
    temp = cleaned.copy()
    temp["shuffled_popular"] = shuffled

    dist1 = temp[temp["shuffled_popular"]]["track_genre"].value_counts(normalize=True)
    dist2 = temp[~temp["shuffled_popular"]]["track_genre"].value_counts(normalize=True)

    simulated_genre_tvds.append(tvd(dist1, dist2))

fig = px.histogram(
    x=simulated_genre_tvds,
    nbins=30,
    title="Hypothesis Test: Genre Distribution and Popularity",
    labels={"x": "Simulated TVD", "y": "Count"}
)
fig.add_vline(x=observed_genre_tvd, line_dash="dash")
save_plot(fig, "hypothesis_test")


# -----------------------------
# Final model
# -----------------------------
final_features = [
    "danceability",
    "acousticness",
    "energy",
    "loudness",
    "instrumentalness",
    "speechiness",
    "valence",
    "duration_min",
    "explicit",
    "track_genre",
    "loudness_energy_ratio",
    "tempo_filled",
    "release_year",
    "num_artists"
]

model_df = cleaned[final_features + ["is_popular"]].dropna()

X_final = model_df[final_features]
y = model_df["is_popular"]

X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_final,
    y,
    test_size=0.2,
    random_state=42
)

categorical = X_final.select_dtypes(include=["object", "category"]).columns.tolist()
numerical = X_final.select_dtypes(include=["number", "bool"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ]
)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [5, 10, 20]
}

grid_search = GridSearchCV(
    final_pipeline,
    param_grid,
    cv=5,
    scoring="f1"
)

grid_search.fit(X_train_final, y_train_final)
y_pred_final = grid_search.predict(X_test_final)

print("Best params:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test_final, y_pred_final))
print("Precision:", precision_score(y_test_final, y_pred_final))
print("Recall:", recall_score(y_test_final, y_pred_final))
print("F1:", f1_score(y_test_final, y_pred_final))


# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_test_final, y_pred_final)

labels = ["Not Popular", "Popular"]

fig = ff.create_annotated_heatmap(
    cm,
    x=labels,
    y=labels,
    colorscale="Blues",
    showscale=True
)

fig.update_layout(
    title="Confusion Matrix — Final Model",
    xaxis_title="Predicted",
    yaxis_title="Actual"
)

save_plot(fig, "confusion_matrix")


# -----------------------------
# Feature correlation
# -----------------------------
corr_features = [
    "danceability",
    "acousticness",
    "energy",
    "loudness",
    "instrumentalness",
    "speechiness",
    "valence",
    "duration_min",
    "loudness_energy_ratio",
    "tempo_filled",
    "release_year",
    "num_artists",
    "popularity"
]

corr_df = (
    cleaned[corr_features]
    .corr(numeric_only=True)["popularity"]
    .drop("popularity")
    .sort_values()
    .reset_index()
)

corr_df.columns = ["feature", "correlation"]

fig = px.bar(
    corr_df,
    x="correlation",
    y="feature",
    orientation="h",
    title="Feature Correlation with Popularity",
    labels={"correlation": "Correlation with Popularity", "feature": "Feature"}
)

save_plot(fig, "feature_correlation")


# -----------------------------
# Permutation importance
# -----------------------------
result = permutation_importance(
    grid_search.best_estimator_,
    X_test_final,
    y_test_final,
    n_repeats=10,
    scoring="f1",
    random_state=42
)

perm_df = pd.DataFrame({
    "feature": X_test_final.columns,
    "importance": result.importances_mean
}).sort_values("importance", ascending=False)

fig = px.bar(
    perm_df.sort_values("importance"),
    x="importance",
    y="feature",
    orientation="h",
    title="Permutation Importance — Final Model",
    labels={
        "importance": "Mean F1 Drop When Shuffled",
        "feature": "Feature"
    },
    color="importance",
    color_continuous_scale="RdYlGn"
)

save_plot(fig, "permutation_importance")


# -----------------------------
# Fairness test 1:
# lower-popularity genres vs higher-popularity genres
# -----------------------------
lower_genres = ["classical", "country"]
higher_genres = ["electronic", "hip-hop", "metal", "pop"]

test_df = X_test_final.copy()
test_df["actual"] = y_test_final.values
test_df["predicted"] = y_pred_final

lower_mask = test_df["track_genre"].isin(lower_genres)
higher_mask = test_df["track_genre"].isin(higher_genres)

f1_lower = f1_score(
    test_df.loc[lower_mask, "actual"],
    test_df.loc[lower_mask, "predicted"],
    pos_label=True,
    zero_division=0
)

f1_higher = f1_score(
    test_df.loc[higher_mask, "actual"],
    test_df.loc[higher_mask, "predicted"],
    pos_label=True,
    zero_division=0
)

observed_genre_diff = f1_lower - f1_higher

simulated_genre_diffs = []

for _ in range(1000):
    shuffled_genres = np.random.permutation(test_df["track_genre"])
    temp = test_df.copy()
    temp["track_genre"] = shuffled_genres

    lower_mask = temp["track_genre"].isin(lower_genres)
    higher_mask = temp["track_genre"].isin(higher_genres)

    f1_lower_sim = f1_score(
        temp.loc[lower_mask, "actual"],
        temp.loc[lower_mask, "predicted"],
        pos_label=True,
        zero_division=0
    )

    f1_higher_sim = f1_score(
        temp.loc[higher_mask, "actual"],
        temp.loc[higher_mask, "predicted"],
        pos_label=True,
        zero_division=0
    )

    simulated_genre_diffs.append(f1_lower_sim - f1_higher_sim)

fig = px.histogram(
    x=simulated_genre_diffs,
    nbins=30,
    title="Fairness Permutation Test by Genre Group",
    labels={"x": "Difference in F1 Score", "y": "Count"}
)

fig.add_vline(x=observed_genre_diff, line_dash="dash")

save_plot(fig, "fairness_permutation")


# -----------------------------
# Fairness test 2:
# explicit vs non-explicit
# -----------------------------
explicit_mask = X_test_final["explicit"] == True
non_explicit_mask = X_test_final["explicit"] == False

f1_explicit = f1_score(
    y_test_final[explicit_mask],
    y_pred_final[explicit_mask],
    pos_label=True,
    zero_division=0
)

f1_non_explicit = f1_score(
    y_test_final[non_explicit_mask],
    y_pred_final[non_explicit_mask],
    pos_label=True,
    zero_division=0
)

observed_explicit_diff = f1_explicit - f1_non_explicit

simulated_explicit_diffs = []

for _ in range(1000):
    shuffled_explicit = np.random.permutation(explicit_mask)

    f1_shuffled_explicit = f1_score(
        y_test_final[shuffled_explicit],
        y_pred_final[shuffled_explicit],
        pos_label=True,
        zero_division=0
    )

    f1_shuffled_non_explicit = f1_score(
        y_test_final[~shuffled_explicit],
        y_pred_final[~shuffled_explicit],
        pos_label=True,
        zero_division=0
    )

    simulated_explicit_diffs.append(
        f1_shuffled_explicit - f1_shuffled_non_explicit
    )

fig = px.histogram(
    x=simulated_explicit_diffs,
    nbins=30,
    title="Fairness Permutation Test by Explicit Status",
    labels={"x": "Difference in F1 Score", "y": "Count"}
)

fig.add_vline(x=observed_explicit_diff, line_dash="dash")

save_plot(fig, "fairness_explicit")


print("All plots saved successfully in the assets folder.")

