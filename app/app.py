import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# LOAD MODEL AND DATA

best_model = joblib.load(
    "best_movie_model.pkl"
)

model_columns = joblib.load(
    "model_columns.pkl"
)

actor_star_table = pd.read_csv(
    "actor_star_table.csv"
)

director_star_table = pd.read_csv(
    "director_star_table.csv"
)

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Movie Success Simulator",
    layout="wide"
)

# APP TITLE

st.title("Movie Success Simulator")

st.markdown(
    """
This application simulates hypothetical movie projects using
historical actor/director performance data and machine learning predictions.

The system evaluates how casting choices, directing history,
budget, runtime, and genre combinations influence predicted
commercial success probability.
"""
)

# GENRE LIST

all_genres = [

    "Drama",
    "Comedy",
    "Action",
    "Adventure",
    "Crime",
    "Thriller",
    "Romance",
    "Horror",
    "Mystery",
    "Fantasy",
    "Biography",
    "Animation",
    "Family",
    "Sci-Fi",
    "History",
    "Music",
    "Sport",
    "War",
    "Musical"
]

# ACTOR / DIRECTOR OPTIONS

actor_options = sorted(
    actor_star_table[
        "actor_name"
    ].dropna().unique()
)

director_options = sorted(
    director_star_table[
        "director_name"
    ].dropna().unique()
)

# DEFAULT DIRECTOR

default_director_index = 0

if "Christopher Nolan" in director_options:

    default_director_index = (
        director_options.index(
            "Christopher Nolan"
        )
    )

# HELPER FUNCTION


def build_movie_features(
    budget,
    runtime,
    genres,
    actors,
    director_name,
    year=2026
):

    # ACTOR LOOKUP

    selected_actor_rows = actor_star_table[
        actor_star_table[
            "actor_name"
        ].isin(actors)
    ]

    avg_actor_star_power = (
        selected_actor_rows[
            "actor_star_power"
        ].mean()
    )

    max_actor_star_power = (
        selected_actor_rows[
            "actor_star_power"
        ].max()
    )

    cast_size = len(
        selected_actor_rows
    )

    # Missing protection

    if np.isnan(avg_actor_star_power):
        avg_actor_star_power = 0

    if np.isnan(max_actor_star_power):
        max_actor_star_power = 0

    # DIRECTOR LOOKUP

    selected_director = director_star_table[
        director_star_table[
            "director_name"
        ] == director_name
    ]

    if not selected_director.empty:

        director_avg_revenue = (
            selected_director[
                "director_avg_revenue"
            ].iloc[0]
        )

        director_success_rate = (
            selected_director[
                "director_success_rate"
            ].iloc[0]
        )

        director_movies = (
            selected_director[
                "director_movies"
            ].iloc[0]
        )

    else:

        director_avg_revenue = 0
        director_success_rate = 0
        director_movies = 0

    # BUILD FEATURE VECTOR

    movie_features = {

        "year": year,

        "runtime": runtime,

        "log_budget": np.log1p(budget),

        "avg_actor_star_power":
            avg_actor_star_power,

        "max_actor_star_power":
            max_actor_star_power,

        "cast_size":
            cast_size,

        "director_avg_revenue":
            director_avg_revenue,

        "director_success_rate":
            director_success_rate,

        "director_movies":
            director_movies,

        "missing_actor_history": 0,

        "missing_director_history": 0
    }

    # ADD GENRES

    for genre in all_genres:

        movie_features[genre] = (
            1 if genre in genres
            else 0
        )

    # CONVERT TO DATAFRAME

    scenario_movie = pd.DataFrame([
        movie_features
    ])

    scenario_movie = scenario_movie.reindex(
        columns=model_columns,
        fill_value=0
    )

    return (
        scenario_movie,
        selected_actor_rows,
        selected_director
    )

# SIDEBAR INPUTS


st.sidebar.header("Movie Configuration")

budget = st.sidebar.slider(
    "Budget ($)",
    min_value=1_000_000,
    max_value=300_000_000,
    value=100_000_000,
    step=1_000_000
)

runtime = st.sidebar.slider(
    "Runtime (minutes)",
    min_value=70,
    max_value=240,
    value=130
)

year = st.sidebar.slider(
    "Release Year",
    min_value=2025,
    max_value=2035,
    value=2026
)

selected_genres = st.sidebar.multiselect(
    "Select Genres",
    all_genres,
    default=[
        "Action",
        "Drama",
        "Thriller"
    ]
)

# ACTOR SELECTION

st.sidebar.header("Casting")

selected_actors = st.sidebar.multiselect(
    "Select Actors",
    actor_options,
    default=[
        "Leonardo DiCaprio",
        "Al Pacino"
    ]
)

# DIRECTOR SELECTION

st.sidebar.header("Director")

selected_director_name = st.sidebar.selectbox(
    "Select Director",
    director_options,
    index=default_director_index
)

# BUILD MAIN SCENARIO

(
    scenario_movie,
    selected_actor_rows,
    selected_director
) = build_movie_features(

    budget,
    runtime,
    selected_genres,
    selected_actors,
    selected_director_name,
    year
)

# GENERATE PREDICTION

success_probability = (
    best_model.predict_proba(
        scenario_movie
    )[0][1]
)

# DISPLAY MAIN RESULTS

st.header("Prediction Results")

st.metric(
    "Predicted Success Probability",
    f"{success_probability:.1%}"
)

# COMMERCIAL POTENTIAL

if success_probability >= 0.70:

    st.success(
        "High Commercial Potential"
    )

elif success_probability >= 0.50:

    st.warning(
        "Moderate Commercial Potential"
    )

else:

    st.error(
        "Lower Commercial Potential"
    )

# KEY FACTORS

st.subheader("Key Production Factors")

key_factors = []

avg_actor_power = (
    selected_actor_rows[
        "actor_star_power"
    ].mean()
)

director_success = (
    selected_director[
        "director_success_rate"
    ].iloc[0]
)

if avg_actor_power >= 3:

    key_factors.append(
        "Strong actor star power"
    )

if director_success >= 0.60:

    key_factors.append(
        "Strong director success history"
    )

if budget >= 150_000_000:

    key_factors.append(
        "Large production budget"
    )

if "Action" in selected_genres:

    key_factors.append(
        "Action genre selected"
    )

if len(key_factors) == 0:

    key_factors.append(
        "Moderate overall production profile"
    )

for factor in key_factors:

    st.write(f"- {factor}")

# ACTOR PROFILES

st.subheader("Selected Actor Profiles")

if not selected_actor_rows.empty:

    st.dataframe(

        selected_actor_rows[
            [
                "actor_name",
                "actor_star_power",
                "actor_success_rate",
                "actor_avg_revenue",
                "actor_movies"
            ]
        ].round(3)

    )

# DIRECTOR PROFILE

st.subheader("Selected Director Profile")

if not selected_director.empty:

    st.dataframe(

        selected_director[
            [
                "director_name",
                "director_star_power",
                "director_success_rate",
                "director_avg_revenue",
                "director_movies"
            ]
        ].round(3)

    )

# VISUALIZATION

st.subheader(
    "Predicted Success Probability"
)

fig, ax = plt.subplots(
    figsize=(8, 2)
)

ax.barh(
    ["Success Probability"],
    [success_probability]
)

ax.set_xlim(0, 1)

ax.set_xlabel("Probability")

st.pyplot(fig)

# FEATURE VECTOR

with st.expander(
    "View Final Model Feature Vector"
):

    st.dataframe(
        scenario_movie.round(3)
    )

# SCENARIO COMPARISON SECTION

st.header("Scenario Comparison")

st.markdown(
    """
Compare two hypothetical movie production scenarios side-by-side.
"""
)

# CREATE COLUMNS

col1, col2 = st.columns(2)

# SCENARIO A

with col1:

    st.subheader("Scenario A")

    budget_a = st.slider(
        "Budget A ($)",
        1_000_000,
        300_000_000,
        120_000_000,
        1_000_000
    )

    runtime_a = st.slider(
        "Runtime A",
        70,
        240,
        130
    )

    genres_a = st.multiselect(
        "Genres A",
        all_genres,
        default=["Action", "Drama"],
        key="genres_a"
    )

    actors_a = st.multiselect(
        "Actors A",
        actor_options,
        default=[
            "Leonardo DiCaprio",
            "Al Pacino"
        ],
        key="actors_a"
    )

    director_a = st.selectbox(
        "Director A",
        director_options,
        index=default_director_index,
        key="director_a"
    )

# SCENARIO B

with col2:

    st.subheader("Scenario B")

    budget_b = st.slider(
        "Budget B ($)",
        1_000_000,
        300_000_000,
        60_000_000,
        1_000_000
    )

    runtime_b = st.slider(
        "Runtime B",
        70,
        240,
        120
    )

    genres_b = st.multiselect(
        "Genres B",
        all_genres,
        default=["Drama"],
        key="genres_b"
    )

    default_actor_b = []

    if "Tom Hanks" in actor_options:
        default_actor_b = ["Tom Hanks"]

    actors_b = st.multiselect(
        "Actors B",
        actor_options,
        default=default_actor_b,
        key="actors_b"
    )

    director_b = st.selectbox(
        "Director B",
        director_options,
        key="director_b"
    )

# BUILD SCENARIOS

scenario_a, _, _ = build_movie_features(
    budget_a,
    runtime_a,
    genres_a,
    actors_a,
    director_a
)

scenario_b, _, _ = build_movie_features(
    budget_b,
    runtime_b,
    genres_b,
    actors_b,
    director_b
)

# PREDICTIONS

prob_a = (
    best_model.predict_proba(
        scenario_a
    )[0][1]
)

prob_b = (
    best_model.predict_proba(
        scenario_b
    )[0][1]
)

# COMPARISON TABLE

comparison_df = pd.DataFrame({

    "Scenario": [
        "Scenario A",
        "Scenario B"
    ],

    "Predicted Success Probability": [
        prob_a,
        prob_b
    ]
})

st.subheader(
    "Scenario Comparison Results"
)

st.dataframe(
    comparison_df.round(3)
)

# COMPARISON VISUALIZATION

fig, ax = plt.subplots(
    figsize=(8, 4)
)

ax.bar(
    comparison_df["Scenario"],
    comparison_df[
        "Predicted Success Probability"
    ]
)

ax.set_ylim(0, 1)

ax.set_ylabel(
    "Success Probability"
)

ax.set_title(
    "Scenario Comparison"
)

st.pyplot(fig)

# FINAL MESSAGE

if prob_a > prob_b:

    st.success(
        "Scenario A is predicted to have "
        "higher commercial potential."
    )

elif prob_b > prob_a:

    st.success(
        "Scenario B is predicted to have "
        "higher commercial potential."
    )

else:

    st.info(
        "Both scenarios produced "
        "similar predictions."
    )
