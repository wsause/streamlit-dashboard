import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from streamlit_plotly_events import plotly_events


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

@st.cache_data
def load_data():

    cip_df = pd.read_csv(
        "Education_CIP_to_ONET_SOC.csv"
    )

    bls_df = pd.read_csv(
        "Employment Projections.csv"
    )

    abilities_df = pd.read_csv(
        "Abilities.csv"
    )

    skills_df = pd.read_csv(
        "Essential_Skills.csv"
    )

    activities_df = pd.read_csv(
        "Work_Activities.csv"
    )

    tasks_df = pd.read_csv(
        "task_statements.csv"
    )

    openai_df = pd.read_csv(
        "occ_level.csv"
    )

    return (
        cip_df,
        bls_df,
        abilities_df,
        skills_df,
        activities_df,
        tasks_df,
        openai_df
    )


(
    cip_df,
    bls_df,
    abilities_df,
    skills_df,
    activities_df,
    tasks_df,
    openai_df

) = load_data()



# ---------------------------------------------------
# Clean BLS Data
# ---------------------------------------------------

bls_df["Occupation Code"] = (
    bls_df["Occupation Code"]
    .astype(str)
    .str.replace('="', '', regex=False)
    .str.replace('"', '', regex=False)
    .str.strip()
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

    bls_df[col] = pd.to_numeric(
        bls_df[col],
        errors="coerce"
    )


# Clean wages

bls_df["Median Annual Wage 2024"] = (
    bls_df["Median Annual Wage 2024"]
    .replace("N/A", None)
)



# ---------------------------------------------------
# Prepare CIP / O*NET Data
# ---------------------------------------------------

cip_df = cip_df[
    [
        "2020 CIP Code",
        "2020 CIP Title",
        "O*NET-SOC 2019 Code",
        "O*NET-SOC 2019 Title"
    ]
].dropna()


# Full O*NET code
cip_df["O*NET Code"] = (
    cip_df["O*NET-SOC 2019 Code"]
    .astype(str)
    .str.strip()
)


# SOC code for BLS matching
cip_df["Occupation Code"] = (
    cip_df["O*NET Code"]
    .str.split(".")
    .str[0]
)



# ---------------------------------------------------
# Sidebar - Major Selection
# ---------------------------------------------------

st.subheader("Select a Major")

selected_cip = st.selectbox(
    "Select Major",
    sorted(cip_df["2020 CIP Title"].unique())
)



# ---------------------------------------------------
# Get Occupations for Selected Major
# ---------------------------------------------------

occupation_df = (
    cip_df[
        cip_df["2020 CIP Title"]
        == selected_cip
    ]
    [
        [
            "O*NET-SOC 2019 Title",
            "O*NET Code",
            "Occupation Code"
        ]
    ]
    .drop_duplicates()
)



# ---------------------------------------------------
# Merge BLS Data
# ---------------------------------------------------

career_df = occupation_df.merge(
    bls_df,
    on="Occupation Code",
    how="left"
)


career_df = career_df.dropna(
    subset=[
        "Employment 2024"
    ]
)



# ---------------------------------------------------
# Career Comparison Charts
# ---------------------------------------------------

st.header(
    "Career Options"
)

st.write(
    "Click an occupation in any chart to view the O*NET career profile."
)

col1, col2, col3 = st.columns(3)



# ---------------------------------------------------
# Employment Chart
# ---------------------------------------------------

with col1:

    employment_df = career_df[
        [
            "O*NET-SOC 2019 Title",
            "Employment 2024"
        ]
    ].copy()


    employment_fig = px.bar(
        employment_df,
        x="O*NET-SOC 2019 Title",
        y="Employment 2024",
        title="Employment (2024)"
    )


    employment_fig.update_layout(
        xaxis_tickangle=-60,
        height=450
    )


    employment_event = st.plotly_chart(
        employment_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points"
    )


# ---------------------------------------------------
# Wage Chart
# ---------------------------------------------------

with col2:

    wage_df = career_df[
        [
            "O*NET-SOC 2019 Title",
            "Median Annual Wage 2024"
        ]
    ].copy()
    
    
    wage_fig = px.bar(
        wage_df,
        x="O*NET-SOC 2019 Title",
        y="Median Annual Wage 2024",
        title="Median Annual Wage (2024)"
    )


    wage_fig.update_layout(
        xaxis_tickangle=-60,
        height=450
    )


    wage_event = st.plotly_chart(
        wage_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points"
    )



# ---------------------------------------------------
# Education Chart
# ---------------------------------------------------

education_levels = {
    "No formal educational credential": 1,
    "High school diploma or equivalent": 2,
    "Some college, no degree": 2.5,
    "Postsecondary nondegree award": 3,
    "Associate's degree": 4,
    "Bachelor's degree": 5,
    "Master's degree": 6,
    "Doctoral or professional degree": 7
}


education_df = career_df.copy()


education_df["Education Score"] = (
    education_df[
        "Typical Entry-Level Education"
    ]
    .map(education_levels)
)

with col3:

    education_df = education_df[
        [
            "O*NET-SOC 2019 Title",
            "Education Score"
        ]
    ].copy()
        
        
    education_fig = px.bar(
        education_df,
        x="O*NET-SOC 2019 Title",
        y="Education Score",
        title="Education Level"
    )

    education_fig.update_yaxes(
        tickmode="array",
        tickvals=list(education_levels.values()),
        ticktext=list(education_levels.keys())
    )


    education_fig.update_layout(
        xaxis_tickangle=-60,
        height=450
    )


    education_event = st.plotly_chart(
        education_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points"
    )
    

# ---------------------------------------------------
# Initialize
# ---------------------------------------------------

if "selected_occupation" not in st.session_state:
    st.session_state.selected_occupation = None


# Employment selection
if employment_event.selection and employment_event.selection.points:
    st.session_state.selected_occupation = (
        employment_event.selection.points[0]["x"]
    )

# Wage selection
if wage_event.selection and wage_event.selection.points:
    st.session_state.selected_occupation = (
        wage_event.selection.points[0]["x"]
    )

# Education selection
if education_event.selection and education_event.selection.points:
    st.session_state.selected_occupation = (
        education_event.selection.points[0]["x"]
    )

# Get the current selection
selected_occupation = st.session_state.selected_occupation


if selected_occupation is None:
    st.info("Click a bar in one of the charts above to view the occupation profile.")
    st.stop()

# ---------------------------------------------------
# Selected Occupation Information
# ---------------------------------------------------

st.divider()

st.header(
    selected_occupation
)


selected_onet = cip_df[
    cip_df["O*NET-SOC 2019 Title"]
    == selected_occupation
][
    [
        "O*NET-SOC 2019 Code",
        "O*NET-SOC 2019 Title",
        "O*NET Code",
        "Occupation Code"
    ]
]


selected_soc_code = (
    selected_onet["Occupation Code"]
    .iloc[0]
)



# BLS Metrics

bls_match = bls_df[
    bls_df["Occupation Code"]
    == selected_soc_code
]


if not bls_match.empty:

    row = bls_match.iloc[0]

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Employment (2024)",
            f"{row['Employment 2024']:,.1f}"
        )


    with col2:

        wage = row[
            "Median Annual Wage 2024"
        ]

        if pd.isna(wage):

            st.metric(
                "Median Wage (2024)",
                "N/A"
            )

        else:

            st.metric(
                "Median Wage (2024)",
                f"${wage:,.0f}"
            )


    with col3:

        st.metric(
            "Education Level",
            row[
                "Typical Entry-Level Education"
            ]
        )

else:

    st.warning(
        "No matching BLS data found."
    )

# ---------------------------------------------------
# Abilities Visualization
# ---------------------------------------------------

st.subheader(
    "Occupation Ability Profile"
)


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

    ability_profile = occupation_abilities.pivot_table(
        index=[
            "O*NET-SOC 2019 Title",
            "Element Name"
        ],
        columns="Scale ID",
        values="Data Value"
    ).reset_index()


    ability_profile = ability_profile.rename(
        columns={
            "O*NET-SOC 2019 Title": "Occupation",
            "Element Name": "Ability",
            "IM": "Importance",
            "LV": "Level"
        }
    )


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
        .groupby(
            [
                "Importance",
                "Level"
            ]
        )
        .agg(
            Ability=("Ability", ", ".join)
        )
        .reset_index()
    )


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

st.subheader(
    "Occupation Skill Profile"
)


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

    skill_profile = occupation_skills.pivot_table(
        index=[
            "O*NET-SOC 2019 Title",
            "Element Name"
        ],
        columns="Scale ID",
        values="Data Value"
    ).reset_index()


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
        .groupby(
            [
                "Importance",
                "Level"
            ]
        )
        .agg(
            Skill=("Skill", ", ".join)
        )
        .reset_index()
    )


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

st.subheader(
    "Occupation Work Activities"
)


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

    activity_profile = occupation_activities.pivot_table(
        index=[
            "O*NET-SOC 2019 Title",
            "Element ID",
            "Element Name"
        ],
        columns="Scale ID",
        values="Data Value"
    ).reset_index()


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
        .groupby(
            [
                "Importance",
                "Level"
            ]
        )
        .agg(
            Activity=("Activity", ", ".join)
        )
        .reset_index()
    )


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
# AI Exposure Metrics
# ---------------------------------------------------

st.subheader(
    "AI Exposure Rating"
)


occupation_aiexposure = openai_df.merge(
    selected_onet,
    left_on="O*NET-SOC Code",
    right_on="O*NET-SOC 2019 Code",
    how="inner"
)


if occupation_aiexposure.empty:

    st.warning(
        f"No AI exposure data available for {selected_occupation}."
    )

else:

    row = occupation_aiexposure.iloc[0]


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "AI Exposure Rating",
            f"{row['dv_rating_beta']:.0%}"
        )


    with col2:

        st.metric(
            "Occupation",
            selected_occupation
        )