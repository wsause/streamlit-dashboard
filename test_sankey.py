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

    openai_df = pd.read_csv(
        "occ_level.csv"
    )

    return (
        cip_df,
        bls_df,
        abilities_df,
        skills_df,
        activities_df,
        openai_df
    )


(
    cip_df,
    bls_df,
    abilities_df,
    skills_df,
    activities_df,
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

# Reset selected occupation when the major changes
if "last_major" not in st.session_state:
    st.session_state.last_major = selected_cip

if st.session_state.last_major != selected_cip:
    st.session_state.selected_occupation = None
    st.session_state.last_major = selected_cip


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


# ---------------------------------------------------------
# Sankey Diagram: Majors → Occupations
# ---------------------------------------------------------
import plotly.graph_objects as go

# Get occupations associated with the selected major
selected_occupation_df = (
    cip_df[
        cip_df["2020 CIP Title"] == selected_cip
    ]
    [["2020 CIP Title", "O*NET-SOC 2019 Title"]]
    .drop_duplicates()
)

selected_occupations = (
    selected_occupation_df["O*NET-SOC 2019 Title"]
    .tolist()
)


# Find other majors that share those occupations
related_majors = (
    cip_df[
        cip_df["O*NET-SOC 2019 Title"].isin(selected_occupations)
        & (cip_df["2020 CIP Title"] != selected_cip)
    ]
    ["2020 CIP Title"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)


# Put the selected major first, followed by related majors
all_majors = [selected_cip] + related_majors


# Get all major → occupation relationships
sankey_df = (
    cip_df[
        cip_df["2020 CIP Title"].isin(all_majors)
    ]
    [["2020 CIP Title", "O*NET-SOC 2019 Title"]]
    .drop_duplicates()
)


# ---------------------------------------------------------
# Create unique nodes
# ---------------------------------------------------------

major_nodes = all_majors

occupation_nodes = (
    sankey_df["O*NET-SOC 2019 Title"]
    .drop_duplicates()
    .tolist()
)

# Major nodes are on the left; occupation nodes are on the right
nodes = major_nodes + occupation_nodes

node_index = {
    node: i
    for i, node in enumerate(nodes)
}


# ---------------------------------------------------------
# Create links
# ---------------------------------------------------------

sources = (
    sankey_df["2020 CIP Title"]
    .map(node_index)
    .tolist()
)

targets = (
    sankey_df["O*NET-SOC 2019 Title"]
    .map(node_index)
    .tolist()
)

# Each major → occupation relationship has a value of 1
values = [1] * len(sankey_df)


# ---------------------------------------------------------
# Position nodes
# ---------------------------------------------------------

def evenly_spaced(n, start=0.01, end=0.99):
    if n == 1:
        return [0.5]

    return np.linspace(start, end, n).tolist()


x_positions = (
    [0] * len(major_nodes)
    + [1] * len(occupation_nodes)
)

y_positions = (
    evenly_spaced(len(major_nodes))
    + evenly_spaced(len(occupation_nodes))
)


# ---------------------------------------------------------
# Build Sankey
# ---------------------------------------------------------

fig = go.Figure(
    go.Sankey(
        arrangement="fixed",

        node=dict(
            pad=15,
            thickness=20,
            label=nodes,
            x=x_positions,
            y=y_positions,
        ),

        link=dict(
            source=sources,
            target=targets,
            value=values,
        )
    )
)


fig.update_layout(
    title="Majors and Related Occupations",
    height=700
)


st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# Career Comparison Charts
# ---------------------------------------------------

st.header(
    "Career Options"
)

st.write(
    "Click a career bubble to view details."
)


# ---------------------------------------------------
# Employment Chart
# ---------------------------------------------------

bubble_df = career_df[
    [
        "O*NET-SOC 2019 Title",
        "Employment 2024",
        "Median Annual Wage 2024",
        "Occupational Openings, 2024-2034 Annual Average",
        "Typical Entry-Level Education"
    ]
].copy()

bubble_df = bubble_df.dropna(
    subset=[
        "Employment 2024",
        "Median Annual Wage 2024"
    ]
)

education_order = [
    "No formal educational credential",
    "High school diploma or equivalent",
    "Some college, no degree",
    "Postsecondary nondegree award",
    "Associate's degree",
    "Bachelor's degree",
    "Master's degree",
    "Doctoral or professional degree"
]

bubble_df["Typical Entry-Level Education"] = pd.Categorical(
    bubble_df["Typical Entry-Level Education"],
    categories=education_order,
    ordered=True
)

education_colors = {
    "No formal educational credential": "gray",
    "High school diploma or equivalent": "red",
    "Some college, no degree": "orange",
    "Postsecondary nondegree award": "gold",
    "Associate's degree": "green",
    "Bachelor's degree": "blue",
    "Master's degree": "purple",
    "Doctoral or professional degree": "black"
}

bubble_fig = px.scatter(
    bubble_df,
    x="Median Annual Wage 2024",
    y="Employment 2024",
    size="Occupational Openings, 2024-2034 Annual Average",
    size_max=40,  
    color="Typical Entry-Level Education",
    color_discrete_map=education_colors,
    hover_name="O*NET-SOC 2019 Title",
    custom_data=["O*NET-SOC 2019 Title"],
    title="Career Landscape"
)

bubble_fig.update_layout(
    height=400,
    xaxis_title="Median Annual Wage ($)",
    yaxis_title="Employment (2024)",
    legend_title="Education",
    hovermode="closest",
    clickmode="event+select"
)

bubble_fig.update_traces(
    hovertemplate="%{hovertext}<extra></extra>",
    mode="markers",
    selected=dict(
        marker=dict(opacity=1)
    ),
    unselected=dict(
        marker=dict(opacity=0.2)
    )
)

chart_col, info_col = st.columns([3, 1])

with chart_col:

    bubble_event = st.plotly_chart(
        bubble_fig,
        use_container_width=True,
        key="career_bubble_chart",
        on_select="rerun",
        selection_mode="points"
    )

    if bubble_event.selection.points:

        st.session_state.selected_occupation = (
            # bubble_event.selection.points[0]["hovertext"]
            bubble_event.selection.points[0]["customdata"][0]
        )
    

# ---------------------------------------------------
# Initialize
# ---------------------------------------------------

if "selected_occupation" not in st.session_state:
    st.session_state.selected_occupation = None


# Get the current selection
selected_occupation = st.session_state.selected_occupation


if selected_occupation is None:
    # st.info("Click a career bubble to view details.")
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

# AI Exposure Metrics

occupation_aiexposure = openai_df.merge(
    selected_onet,
    left_on="O*NET-SOC Code",
    right_on="O*NET-SOC 2019 Code",
    how="inner"
)


# BLS Metrics

bls_match = bls_df[
    bls_df["Occupation Code"]
    == selected_soc_code
]


if not bls_match.empty:

    with info_col:

        if selected_occupation is None:
            st.info("Click a career bubble to view details.")

        else:

            row = career_df[
                career_df["O*NET-SOC 2019 Title"] == selected_occupation
            ]

            if not row.empty:

                row = row.iloc[0]

                st.subheader(selected_occupation)

                st.metric(
                    "Employment",
                    f"{row['Employment 2024']:,.1f}"
                )

                wage = row["Median Annual Wage 2024"]

                if pd.isna(wage):
                    st.metric("Median Wage", "N/A")
                else:
                    st.metric(
                        "Median Wage",
                        f"${wage:,.0f}"
                    )

                st.metric(
                    "Education",
                    row["Typical Entry-Level Education"]
                )

                if occupation_aiexposure.empty:
                    st.warning(
                        f"No AI exposure data available for {selected_occupation}."
                    )
                else:
                    row = occupation_aiexposure.iloc[0]
                    st.metric(
                        "AI Exposure Rating",
                        f"{row['dv_rating_beta']:.0%}"
                    )

else:

    st.warning(
        "No matching BLS data found."
    )


abilities_col, skills_col, activities_col = st.columns(3)

# ---------------------------------------------------
# Abilities Visualization
# ---------------------------------------------------

with abilities_col:
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
            legend_title="Ability",
            height=550, 
            margin=dict(b=150),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=9)
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ---------------------------------------------------
# Skills Visualization
# ---------------------------------------------------

with skills_col:

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
            legend_title="Skill",   
            height=550,    
            margin=dict(b=150),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=9)
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ---------------------------------------------------
# Activities Visualization
# ---------------------------------------------------

with activities_col:

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
            legend_title="Activity",
            height=550,
            margin=dict(b=150),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=9)
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )
