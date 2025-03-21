import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Load Data 
@st.cache_data
def load_data():
    df = pd.read_csv("matches.csv")  
    return df

df = load_data()

# Preprocess the data to add match outcomes
def preprocess_data(df):
    # Calculate match outcome based on scores
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


st.sidebar.markdown("---")


#################################### Main Page Content ###################################

st.title("UEFA Champions League Dashboard")

# Interactive Graph Selection
graph_options = [g[0] for g in graphs]
selected_graph = st.selectbox("Select Graph to Display", graph_options)

graph = next(g[1] for g in graphs if g[0] == selected_graph)

# Add another graph for interactivity
remaining_graphs = [g for g in graphs if g[0] != selected_graph]
selected_graph_2 = st.selectbox("Select Second Graph", [g[0] for g in remaining_graphs])

graph2 = next(g[1] for g in remaining_graphs if g[0] == selected_graph_2)

# Columns for displaying the graphs
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(graph, use_container_width=True)

with col2:
    st.plotly_chart(graph2, use_container_width=True)

# Sidebar for Team Selection
st.title("Team Statistics")
team_name = st.selectbox("Select Team:", df['team1'].unique())

# Filter data for the selected team
team_data = df[(df['team1'] == team_name) | (df['team2'] == team_name)]

# Calculate the relevant stats for the selected team
group_stage_data = team_data[team_data['round'] == 'g']
knockout_stage_data = team_data[team_data['round'] == 'k']

# Stats
group_stage_appearances = len(group_stage_data)
knockout_stage_appearances = len(knockout_stage_data)
goals_scored = group_stage_data['score1'].sum() + knockout_stage_data[knockout_stage_data['team1'] == team_name]['score1'].sum() + \
               knockout_stage_data[knockout_stage_data['team2'] == team_name]['score2'].sum()
goals_conceded = group_stage_data['score2'].sum() + knockout_stage_data[knockout_stage_data['team1'] == team_name]['score2'].sum() + \
                 knockout_stage_data[knockout_stage_data['team2'] == team_name]['score1'].sum()
total_matches = len(team_data)
wins = len(team_data[(team_data['team1'] == team_name) & (team_data['score1'] > team_data['score2'])]) + \
       len(team_data[(team_data['team2'] == team_name) & (team_data['score2'] > team_data['score1'])])
losses = len(team_data) - wins
win_loss_ratio = wins / losses if losses != 0 else wins
goals_per_match = goals_scored / total_matches if total_matches != 0 else 0

# Create a DataFrame for stats
team_stats_df = pd.DataFrame({
    "Statistic": [
        "Group Stage Appearances",
        "Knockout Stage Appearances",
        "Total Goals Scored",
        "Total Goals Conceded",
        "Total Matches Played",
        "Total Wins",
        "Total Losses",
        "Win/Loss Ratio",
        "Goals per Match"
    ],
    "Value": [
        group_stage_appearances,
        knockout_stage_appearances,
        goals_scored,
        goals_conceded,
        total_matches,
        wins,
        losses,
        f"{win_loss_ratio:.2f}",
        f"{goals_per_match:.2f}"
    ]
})

# Display team stats as a table
st.header(f"Statistics for {team_name}")
st.table(team_stats_df)

# Create two columns to display visualizations
col3, col4 = st.columns(2)

with col3:
    # Pie Chart for Win/Loss Ratio
    win_loss_data = [wins, losses]
    win_loss_labels = ["Wins", "Losses"]
    fig_pie = go.Figure(data=[go.Pie(labels=win_loss_labels, values=win_loss_data, hole=0.3)])
    fig_pie.update_layout(title=f"Win/Loss Ratio for {team_name}")
    st.plotly_chart(fig_pie, use_container_width=True)

with col4:
    # Bar Chart for Goals Scored vs Goals Conceded
    fig_goals = go.Figure(data=[
        go.Bar(name='Goals Scored', x=['Goals'], y=[goals_scored], marker_color='green'),
        go.Bar(name='Goals Conceded', x=['Goals'], y=[goals_conceded], marker_color='red')
    ])
    fig_goals.update_layout(title=f"Goals Scored vs Goals Conceded by {team_name}", barmode='group')
    st.plotly_chart(fig_goals, use_container_width=True)

with col3:
    # Radar Chart (Spider Plot) for Match Stats
    categories = ['Goals Scored', 'Goals Conceded', 'Wins', 'Losses', 'Goals per Match']
    values = [goals_scored, goals_conceded, wins, losses, goals_per_match]

    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=f"{team_name} Stats"
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(values) + 5])
        ),
        title=f"Radar Chart for {team_name} Stats"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col4:
    # Stacked Bar Chart for Group vs Knockout Goals
    group_goals = group_stage_data['score1'].sum() + group_stage_data['score2'].sum()
    knockout_goals = knockout_stage_data['score1'].sum() + knockout_stage_data['score2'].sum()

    fig_stacked = go.Figure(data=[
        go.Bar(name='Group Stage Goals', x=['Goals'], y=[group_goals], marker_color='blue'),
        go.Bar(name='Knockout Stage Goals', x=['Goals'], y=[knockout_goals], marker_color='purple')
    ])

    fig_stacked.update_layout(title=f"Group vs Knockout Goals for {team_name}", barmode='stack')
    st.plotly_chart(fig_stacked, use_container_width=True)

# Information about the dataset and source
st.sidebar.info("⚽ Data Source: UEFA Champions League 2016/17 - 2021/22")
