import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Career Intelligence Dashboard",
    layout="wide"
)

st.title("Career Intelligence Dashboard")

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

cip_df = pd.read_csv("Education_CIP_to_ONET_SOC.csv")

bls_df = pd.read_csv("Employment Projections.csv")

abilities_df = pd.read_csv(
    "Abilities.csv"
)

skills_df = pd.read_csv("Essential_Skills.csv")

activities_df = pd.read_csv("Work_Activities.csv")

anthropic_df = pd.read_csv("aei_claude_ai_2026-06-26.csv")

tasks_df = pd.read_csv("task_statements.csv")

bls_df["Occupation Code"] = (
    bls_df["Occupation Code"]
    .astype(str)
    .str.replace('="', '', regex=False)
    .str.replace('"', '', regex=False)
    .str.strip()
)

bls_df["Median Annual Wage 2024"] = (
    bls_df["Median Annual Wage 2024"]
    .replace("N/A", None)
    .str.replace(",", "", regex=False)
)

bls_df["Median Annual Wage 2024"] = pd.to_numeric(
    bls_df["Median Annual Wage 2024"],
    errors="coerce"
)

numeric_cols = [
    "Employment 2024",
    "Employment 2034",
    "Employment Change, 2024-2034",
    "Occupational Openings, 2024-2034 Annual Average",
    "Median Annual Wage 2024"
]

for col in numeric_cols:
    bls_df[col] = (
        bls_df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    bls_df[col] = pd.to_numeric(bls_df[col], errors="coerce")

cip_df = cip_df[
    [
        "2020 CIP Code",
        "2020 CIP Title",
        "O*NET-SOC 2019 Code",
        "O*NET-SOC 2019 Title"
    ]
].dropna()

cip_df["Occupation Code"] = (
    cip_df["O*NET-SOC 2019 Code"]
        .astype(str)
        .str.split(".")
        .str[0]
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.header("Career Selection")

selected_cip = st.sidebar.selectbox(
    "Select CIP Major",
    sorted(cip_df["2020 CIP Title"].unique())
)

# ---------------------------------------------------
# Occupation Selection
# ---------------------------------------------------

occupation_df = (
    cip_df[
        cip_df["2020 CIP Title"] == selected_cip
    ][
        ["O*NET-SOC 2019 Title", "Occupation Code"]
    ]
    .drop_duplicates()
    .sort_values("O*NET-SOC 2019 Title")
)

st.sidebar.subheader("Occupations")

selected_occupation = st.sidebar.radio(
    "Select Occupation",
    occupation_df["O*NET-SOC 2019 Title"]
)

selected_code = occupation_df.loc[
    occupation_df["O*NET-SOC 2019 Title"] == selected_occupation,
    "Occupation Code"
].iloc[0]

# ---------------------------------------------------
# Find BLS Match
# ---------------------------------------------------

bls_match = bls_df[
    bls_df["Occupation Code"].astype(str).str.strip()
    == selected_code
]

# ---------------------------------------------------
# Main Display
# ---------------------------------------------------

st.header(selected_occupation)

if not bls_match.empty:

    row = bls_match.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Employment (2024)",
            f"{row['Employment 2024']:,.1f}"
        )

    with col2:

        wage = row["Median Annual Wage 2024"]

        if pd.isna(wage):
            st.metric("Median Wage (2024)", "N/A")
        else:
            st.metric(
                "Median Wage (2024)",
                f"${row['Median Annual Wage 2024']:,.0f}"
            )

    with col3:

        st.metric(
            "Education Level",
            row["Typical Entry-Level Education"]
        )

else:

    st.warning("No matching BLS data found.")

# ---------------------------------------------------
# Abilities Visualization
# ---------------------------------------------------

st.subheader("Occupation Ability Profile")

# Get O*NET code for selected occupation
selected_onet = cip_df[
    cip_df["O*NET-SOC 2019 Title"] == selected_occupation
][
    ["O*NET-SOC 2019 Code", "O*NET-SOC 2019 Title"]
]


# Match abilities
occupation_abilities = abilities_df.merge(
    selected_onet,
    left_on="O*NET-SOC Code",
    right_on="O*NET-SOC 2019 Code",
    how="inner"
)

if occupation_abilities.empty:
    st.warning(
        f"No ability data available for {selected_occupation}."
    )

else:

    # Convert IM and LV into separate columns
    ability_profile = occupation_abilities.pivot_table(
        index=[
            "O*NET-SOC 2019 Title",
            "Element Name"
        ],
        columns="Scale ID",
        values="Data Value"
    ).reset_index()


    # Rename columns
    ability_profile = ability_profile.rename(
        columns={
            "O*NET-SOC 2019 Title": "Occupation",
            "Element Name": "Ability",
            "IM": "Importance",
            "LV": "Level"
        }
    )

    # Sort abilities by importance
    ability_profile = (
        ability_profile
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )

    plot_df = (
        ability_profile
        .groupby(["Importance", "Level"])
        .agg(
            Ability=("Ability", ", ".join)
        )
        .reset_index()
    )

    # Scatter plot
    fig = px.scatter(
        plot_df,
        x="Importance",
        y="Level",
        color="Ability",
        hover_name="Ability",
        hover_data={
            "Ability": False,
            "Importance": ":.2f",
            "Level": ":.2f"
        },
        title=f"Ability Profile: {selected_occupation}"
    )

    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Required Level",
        legend_title="Ability"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# Skills Visualization
# ---------------------------------------------------

# Get skills for selected occupation
occupation_skills = skills_df.merge(
    selected_onet,
    left_on="O*NET-SOC Code",
    right_on="O*NET-SOC 2019 Code",
    how="inner"
)

if occupation_skills.empty:
    st.warning(
        f"No skill data available for {selected_occupation}."
    )

else:

    # Convert IM and LV into separate columns
    skill_profile = occupation_skills.pivot_table(
        index=[
            "O*NET-SOC 2019 Title",
            "Element Name"
        ],
        columns="Scale ID",
        values="Data Value"
    ).reset_index()


    # Rename columns
    skill_profile = skill_profile.rename(
        columns={
            "O*NET-SOC 2019 Title": "Occupation",
            "Element Name": "Skill",
            "IM": "Importance",
            "LV": "Level"
        }
    )

    skill_profile = (
        skill_profile
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )

    plot_df = (
        skill_profile
        .groupby(["Importance", "Level"])
        .agg(
            Skill=("Skill", ", ".join)
        )
        .reset_index()
    )

    # Scatter plot

    fig = px.scatter(
        plot_df,
        x="Importance",
        y="Level",
        color="Skill",
        hover_name="Skill",
        hover_data={
            "Skill": False,
            "Importance": ":.2f",
            "Level": ":.2f"
        },
        title=f"Skill Profile: {selected_occupation}"
    )

    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Required Level",
        legend_title="Skill"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ---------------------------------------------------
# Activities Visualization
# ---------------------------------------------------

# Get activities for selected occupation
occupation_activities = activities_df.merge(
    selected_onet,
    left_on="O*NET-SOC Code",
    right_on="O*NET-SOC 2019 Code",
    how="inner"
)

if occupation_activities.empty:
    st.warning(
        f"No activity data available for {selected_occupation}."
    )

else:

    # Convert IM and LV into separate columns
    activity_profile = occupation_activities.pivot_table(
        index=[
            "O*NET-SOC 2019 Title",
            "Element ID",
            "Element Name"
        ],
        columns="Scale ID",
        values="Data Value"
    ).reset_index()


    # Rename columns
    activity_profile = activity_profile.rename(
        columns={
            "O*NET-SOC 2019 Title": "Occupation",
            "Element Name": "Activity",
            "IM": "Importance",
            "LV": "Level"
        }
    )

    activity_profile = (
        activity_profile
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )

    plot_df = (
        activity_profile
        .groupby(["Importance", "Level"])
        .agg(
            Activity=("Activity", ", ".join)
        )
        .reset_index()
    )

    # Scatter plot

    fig = px.scatter(
        plot_df,
        x="Importance",
        y="Level",
        color="Activity",
        hover_name="Activity",
        hover_data={
            "Activity": False,
            "Importance": ":.2f",
            "Level": ":.2f"
        },
        title=f"Activity Profile: {selected_occupation}"
    )

    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Required Level",
        legend_title="Activity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ---------------------------------------------------
# AI Tasks Metrics
# ---------------------------------------------------

# Get tasks for selected occupation
occupation_tasks = tasks_df.merge(
    selected_onet,
    left_on="O*NET-SOC Code",
    right_on="O*NET-SOC 2019 Code",
    how="inner"
)

if occupation_tasks.empty:
    st.warning(
        f"No task data available for {selected_occupation}."
    )

else:

    # st.write(tasks_df["Task ID"].dtype)
    # st.write(anthropic_df["node_external_id"].dtype)
    occupation_tasks["Task ID"] = occupation_tasks["Task ID"].astype(str)

    # Match tasks to Anthropic AI metrics
    task_ai = occupation_tasks.merge(
        anthropic_df,
        left_on="Task ID",
        right_on="node_external_id",
        how="inner"
    )

    # Keep only AI autonomy metric
    # AI Impact = pct * collaboration_bucket_automation_pct
    # AI Impact = usage * auto_share?
    task_ai = task_ai[
        task_ai["metric_id"] == "collaboration_bucket_automation_pct"
    ]

    # Keep the most recent value for each task
    task_ai = (
        task_ai
        .sort_values("date_start")
        .drop_duplicates(
            subset=["Task ID"],
            keep="last"
        )
    )

    if task_ai.empty:

        st.warning(
            f"No AI activity data available for {selected_occupation}."
        )

    else:

        # ---------------------------------------------------
        # Display AI activity metrics table
        # ---------------------------------------------------

        st.subheader(
            f"AI Activity Metrics: {selected_occupation}"
        )

        ai_table = (
            task_ai[
                ["Task", "value"]
            ]
            .sort_values("value", ascending=False)
        )


        st.dataframe(
            ai_table,
            hide_index=True,
            use_container_width=True
        )

        average_autonomy = task_ai["value"].mean()
        
        st.metric(
            "Occupation AI Autonomy",
            f"{average_autonomy:.2f}"
        )