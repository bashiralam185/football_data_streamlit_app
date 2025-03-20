import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle

# Load Data 
@st.cache_data
def load_data():
    df = pd.read_csv("matches.csv")  
    return df

df = load_data()

# Preprocess the data to add match outcomes
def preprocess_data(df):
    df["match_outcome"] = df.apply(lambda row: 1 if row["score1"] > row["score2"] 
                                         else (2 if row["score1"] < row["score2"] else 0), axis=1)
    return df

df = preprocess_data(df)

# Sidebar for Top Teams Selection
st.sidebar.title("Filters")
top_n = st.sidebar.slider("Select Top N Teams", 5, 20, 10)

# Create a mapping for teams
teams = list(set(df['team1']).union(set(df['team2'])))  # Get unique team names
team_mapping = {team: idx for idx, team in enumerate(teams)}

# Create the graphs for analysis
graphs = [
    ("Top Teams by Goals", px.bar(df.groupby('team1')['score1'].sum().nlargest(top_n).reset_index(), x='team1', y='score1', title="Top Teams by Goals")),
    ("Most Appearances in Knockout", px.bar(df[df['round'] == 'k']['team1'].value_counts().nlargest(top_n).reset_index(), x='team1', y='count', title="Most Appearances in Knockout")),
    ("Teams with Most Losses in Group Stage", px.bar(df[df['round'] == 'g'].groupby('team1')['score1'].count().nsmallest(top_n).reset_index(), x='team1', y='score1', title="Most Losses in Group Stage")),
    ("Teams Wins Most in Knockout", px.bar(df[(df['round'] == 'k') & (df['score1'] > df['score2'])]['team1'].value_counts().nlargest(top_n).reset_index(), x='team1', y='count', title="Most Wins in Knockout")),
    ("Teams with Most Losses in Knockout", px.bar(df[(df['round'] == 'k') & (df['score1'] < df['score2'])]['team1'].value_counts().nlargest(top_n).reset_index(), x='team1', y='count', title="Most Losses in Knockout")),
    ("Teams Creating Most Chances", px.bar(df.groupby('team1')['chances1'].sum().nlargest(top_n).reset_index(), x='team1', y='chances1', title="Most Chances Created")),
    ("Teams Conceding Most Goals", px.bar(df.groupby('team2')['score1'].sum().nlargest(top_n).reset_index(), x='team2', y='score1', title="Most Goals Conceded"))
]

# Load the model and team mapping
@st.cache_data
def load_model():
    with open("model.pkl", "rb") as f:  
        model = pickle.load(f)
    return model

model = load_model()

# Function to predict the match outcome
def predict_match(team1, team2, stage, model, team_mapping):
    if team1 not in team_mapping or team2 not in team_mapping:
        return "Invalid team name!"

    team1_encoded = team_mapping[team1]
    team2_encoded = team_mapping[team2]
    stage_encoded = stage  
    input_data = [[team1_encoded, team2_encoded, stage_encoded, 2025]]  
    probabilities = model.predict_proba(input_data)[0]
    return probabilities

# Sidebar for Match Prediction
st.sidebar.subheader("Match Prediction - UEFA Champions League")
team1 = st.sidebar.selectbox("Select Team 1 (Home Game):", df['team1'].unique())
team2 = st.sidebar.selectbox("Select Team 2 (Away Game):", df['team1'].unique())
round = st.sidebar.selectbox("Select Round:", ["Group Stage", "Semifinals", "Quarter"])
round_encoded = {"Group Stage": 1, "Knockout": 2, "Quarter": 3, "Semifinals": 4}.get(round, 1)

if st.sidebar.button("Predict Match Outcome"):
    prediction = predict_match(team1, team2, round_encoded, model, team_mapping)
    win_prob, lose_prob = prediction[0], prediction[1]
    fig = go.Figure(data=[go.Pie(labels=[team1, team2], values=[win_prob, lose_prob], hole=0.3)])
    fig.update_layout(title=f"Win Probability for {team1} vs {team2} ({round})")
    st.sidebar.plotly_chart(fig, use_container_width=True)

# Main Page Content
st.title("UEFA Champions League Dashboard")
selected_graph = st.selectbox("Select Graph to Display", [g[0] for g in graphs])
graph = next(g[1] for g in graphs if g[0] == selected_graph)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(graph, use_container_width=True)
with col2:
    remaining_graphs = [g for g in graphs if g[0] != selected_graph]
    selected_graph_2 = st.selectbox("Select Second Graph", [g[0] for g in remaining_graphs])
    graph2 = next(g[1] for g in remaining_graphs if g[0] == selected_graph_2)
    st.plotly_chart(graph2, use_container_width=True)

st.sidebar.info("⚽ Data Source: UEFA Champions League 2016/17 - 2021/22")
