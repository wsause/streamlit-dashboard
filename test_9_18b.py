import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

    occupation_data_df = pd.read_csv(
        "Occupation_Data.csv"
    )

    openai_df = pd.read_csv(
        "occ_level.csv"
    )

    return (
        cip_df,
        bls_df,
        occupation_data_df,
        openai_df
    )


(
    cip_df,
    bls_df,
    occupation_data_df,
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
# AI exposure lookup
# ---------------------------------------------------

# Collapse the AI-exposure table down to a 6-digit SOC code so it can
# be joined against bls_df / cip_df's "Occupation Code" (which drops
# the O*NET-SOC ".00" style suffix).
openai_df["Occupation Code"] = (
    openai_df["O*NET-SOC Code"]
    .astype(str)
    .str.split(".")
    .str[0]
)

occ_beta_df = (
    openai_df
    .groupby("Occupation Code")["dv_rating_beta"]
    .mean()
    .reset_index()
)

# O*NET occupation descriptions, keyed on the full O*NET-SOC code
# (e.g. "15-1252.00") to match career_df / fan_df's "O*NET Code".
occupation_data_df["O*NET-SOC Code"] = (
    occupation_data_df["O*NET-SOC Code"]
    .astype(str)
    .str.strip()
)

occupation_description_lookup = dict(
    zip(occupation_data_df["O*NET-SOC Code"], occupation_data_df["Description"])
)

# Light styling for the alternative-major cards further down.
st.markdown(
    """
    <style>
    .alt-badge {
        display: inline-block;
        border-radius: 6px;
        padding: 2px 9px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .alt-shared {
        color: #8A8A8A;
        font-size: 0.85rem;
        margin-top: 6px;
    }
    /* Full-width colored banner for the profile panel's AI exposure
       readout -- background color is set inline per-tier so it
       actually reflects Low/Moderate/High/Very High instead of
       always being red (st.error's fixed color). */
    .exposure-banner {
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0 14px 0;
        font-weight: 700;
        color: white;
    }
    /* Tighten the native bordered container used for each
       alternative-major card so the link button sits snugly with
       the stats beneath it. */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 2px 4px;
    }
    /* Style the alternative-major buttons to look like text links
       rather than boxed buttons -- these are the only buttons in
       the app, so a global rule is safe. */
    div[data-testid="stButton"] > button {
        background: none;
        border: none;
        padding: 0;
        margin: 0 0 2px 0;
        color: #1A56DB;
        font-weight: 700;
        font-size: 1rem;
        text-align: left;
        box-shadow: none;
    }
    div[data-testid="stButton"] > button:hover {
        text-decoration: underline;
        color: #123E9E;
        background: none;
        border: none;
    }
    div[data-testid="stButton"] > button:focus {
        box-shadow: none;
        outline: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _exposure_tier(beta):
    if pd.isna(beta):
        return "Unknown", "#B0B0B0"
    if beta < 0.35:
        return "Low", "#2CA02C"
    if beta < 0.60:
        return "Moderate", "#F2C744"
    if beta < 0.80:
        return "High", "#E67E22"
    return "Very High", "#B22222"


@st.cache_data
def compute_major_exposure(cip_df, occ_beta_df):
    """Average AI exposure + SOC code set for every major (CIP title).

    Used to build the "related majors" panel: for any given major we
    can look up other majors that lead to at least one of the same
    occupations, and compare their average exposure scores.

    Matching is done on the 6-digit SOC code ("Occupation Code"),
    not the more granular O*NET-SOC code -- O*NET splits many broad
    occupations into multiple detailed specializations with
    different decimal suffixes, so matching on the full O*NET-SOC
    code can miss real overlaps between related majors (e.g. two
    majors both leading to "Software Developers" via different
    O*NET specializations would look unrelated under an exact
    O*NET-SOC match, but do share the same SOC code).
    """

    merged = cip_df.merge(
        occ_beta_df,
        on="Occupation Code",
        how="left"
    )

    grouped = (
        merged
        .groupby("2020 CIP Title")
        .agg(
            avg_beta=("dv_rating_beta", "mean"),
            soc_codes=("Occupation Code", lambda s: frozenset(s))
        )
        .reset_index()
    )

    return grouped


major_exposure_df = compute_major_exposure(cip_df, occ_beta_df)


@st.cache_data
def compute_occupation_majors(cip_df):
    """Reverse lookup: which majors (CIP titles) lead to each SOC code.

    Used to show "majors related to this job" in the occupation
    profile panel and on saved-job cards.
    """

    return (
        cip_df
        .groupby("Occupation Code")["2020 CIP Title"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )


occupation_majors_lookup = compute_occupation_majors(cip_df)


# ---------------------------------------------------
# Sidebar - Major Selection
# ---------------------------------------------------
#
# NOTE: the selectbox has key="major_select". Elsewhere in the app
# (e.g. the "Explore" buttons on the lower-exposure alternative
# cards), we can't write to st.session_state.major_select directly
# once this widget has been instantiated this run -- Streamlit
# forbids that. So those buttons instead set a "pending_major" flag
# and call st.rerun(); this block applies that pending value BEFORE
# the selectbox is created, which is allowed.

if "pending_major" in st.session_state:
    st.session_state.major_select = st.session_state.pop("pending_major")

if "saved_jobs" not in st.session_state:
    st.session_state.saved_jobs = {}

st.subheader("Select a Major")

all_major_titles = sorted(cip_df["2020 CIP Title"].unique())

if "major_select" not in st.session_state:
    st.session_state.major_select = all_major_titles[0]

selected_cip = st.selectbox(
    "Select Major",
    all_major_titles,
    key="major_select"
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

career_df = career_df.merge(
    occ_beta_df,
    on="Occupation Code",
    how="left"
)


# ---------------------------------------------------------
# Major -> Occupations fan chart
# ---------------------------------------------------------
#
# Left panel: the selected major plus a couple of lower-AI-exposure
# alternative majors (with how many occupations they share).
# Middle: the major fans directly out to its own occupations (styled
# with soft S-curves, a light background, and a selection ring on the
# clicked node to get closer to the reference mockup).
# Right panel: a compact profile card for whatever occupation was
# last clicked.

MAX_FAN_OCCUPATIONS = 50

st.markdown("## Choose an Occupation")
st.caption(
    "Click a dot to expand its full profile on the right · "
    "line color = AI exposure tier"
)
st.caption(
    "AI exposure (β) estimates how much of a job's tasks could be done at "
    "least twice as fast with AI tools like ChatGPT. Projected job growth "
    "should also be considered when assessing an occupation's AI impact, "
    "since it reflects how employers are actually expected to use AI in "
    "that occupation."
)

if career_df.empty:

    st.info(f"No occupation data found for {selected_cip}.")

else:

    selected_avg_beta = career_df["dv_rating_beta"].mean()
    selected_median_wage = career_df["Median Annual Wage 2024"].median()
    selected_codes = major_exposure_df.loc[
        major_exposure_df["2020 CIP Title"] == selected_cip,
        "soc_codes"
    ].iloc[0]

    fan_df = (
        career_df
        .dropna(subset=["Employment 2024"])
        .sort_values("Employment 2024", ascending=False)
        .head(MAX_FAN_OCCUPATIONS)
        .reset_index(drop=True)
    )

    n_occ = len(fan_df)

    left_col, chart_col, profile_col = st.columns([1, 3.8, 1.1])

    # -----------------------------------------------------
    # Left: your major + lower-exposure alternatives
    # -----------------------------------------------------

    with left_col:

        st.markdown("**YOUR MAJOR**")
        wage_bit = f" · median wage ${selected_median_wage:,.0f}" if pd.notna(selected_median_wage) else ""
        st.info(f"{selected_cip}\n\navg β {selected_avg_beta:.0%}{wage_bit}")

        st.markdown("**RELATED MAJORS**")

        MAX_RELATED_MAJORS = 4

        related = major_exposure_df[
            major_exposure_df["2020 CIP Title"] != selected_cip
        ].copy()

        related["shared_count"] = related["soc_codes"].apply(
            lambda s: len(s & selected_codes)
        )

        # Only surface majors that actually lead to at least one of
        # the same occupations -- shared occupations is what makes
        # a major "related" to this one, regardless of whether its
        # exposure score is higher or lower.
        related = related[related["shared_count"] > 0]

        if related.empty:
            st.caption("No related majors share occupations with this one.")
        else:
            related["pts_diff"] = (related["avg_beta"] - selected_avg_beta) * 100

            # Rank by relevance first (most shared occupations), then
            # by exposure (lowest first) as the tiebreaker.
            related = related.sort_values(
                ["shared_count", "avg_beta"],
                ascending=[False, True]
            )

            featured = related.head(MAX_RELATED_MAJORS)
            remaining = related.iloc[MAX_RELATED_MAJORS:]

            for _, rel in featured.iterrows():
                _, badge_color = _exposure_tier(rel["avg_beta"])

                is_lower = rel["pts_diff"] < 0
                diff_color = "#2CA02C" if is_lower else "#B22222"
                arrow = "↓" if is_lower else "↑"
                direction_word = "lower" if is_lower else "higher"

                with st.container(border=True):

                    if st.button(
                        rel["2020 CIP Title"],
                        key=f"alt_select_{rel['2020 CIP Title']}",
                    ):
                        st.session_state.pending_major = rel["2020 CIP Title"]
                        st.rerun()

                    st.markdown(
                        f"""
                        <span class="alt-badge" style="background:{badge_color}; color:white;">
                            avg β {rel['avg_beta']:.0%}
                        </span>
                        <span style="color:{diff_color}; font-weight:600; margin-left:6px; font-size:0.85rem;">
                            {arrow} {abs(rel['pts_diff']):.0f} pts {direction_word}
                        </span>
                        <div class="alt-shared">{rel['shared_count']} shared occupations</div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Anything beyond the top MAX_RELATED_MAJORS is still a
            # genuine match (shares at least one occupation) -- just
            # not one of the closest ones. Surface it instead of
            # silently dropping it, since "related" is a mutual
            # relationship even when one side has many more matches
            # competing for the featured slots than the other.
            if not remaining.empty:
                with st.expander(f"+{len(remaining)} more related majors"):
                    for _, rel in remaining.iterrows():
                        if st.button(
                            f"{rel['2020 CIP Title']} · {rel['shared_count']} shared · avg β {rel['avg_beta']:.0%}",
                            key=f"alt_select_more_{rel['2020 CIP Title']}",
                        ):
                            st.session_state.pending_major = rel["2020 CIP Title"]
                            st.rerun()

    # -----------------------------------------------------
    # Middle: the fan chart itself
    # -----------------------------------------------------

    with chart_col:

        occ_y = [(n_occ - 1) / 2 - i for i in range(n_occ)]

        OCC_NODE_SIZE = 20

        major_x, occ_x = 0, 4
        current_selection = st.session_state.get("selected_occupation")

        fig = go.Figure()

        # Smoothstep S-curve fan lines: leave the major node flat,
        # ease into a curve, then arrive at the occupation flat --
        # this reads much closer to the reference image than a
        # simple spline through 4 control points.
        t = np.linspace(0, 1, 24)
        ease = 3 * t**2 - 2 * t**3  # smoothstep easing

        for i in range(n_occ):
            row = fan_df.loc[i]
            _, color = _exposure_tier(row["dv_rating_beta"])
            y1 = occ_y[i]
            xs = major_x + t * (occ_x - major_x)
            ys = ease * y1
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines",
                line=dict(width=2, color=color),
                opacity=0.55,
                hoverinfo="none",
                showlegend=False,
            ))

        # Major node
        fig.add_trace(go.Scatter(
            x=[major_x], y=[0],
            mode="markers+text",
            marker=dict(size=34, color="#333333", line=dict(width=0)),
            text=[f"<b>{selected_cip}</b>"],
            textposition="middle left",
            textfont=dict(size=14, color="#222222"),
            hovertext=[
                f"{selected_cip}<br>avg AI exposure {selected_avg_beta:.0%}"
                + (f"<br>median wage ${selected_median_wage:,.0f}" if pd.notna(selected_median_wage) else "")
            ],
            hoverinfo="text",
            showlegend=False,
        ))

        # Occupation nodes
        occ_colors, occ_sizes, occ_labels, occ_hover = [], [], [], []
        occ_line_widths, occ_line_colors = [], []

        for i in range(n_occ):
            row = fan_df.loc[i]
            tier, color = _exposure_tier(row["dv_rating_beta"])
            beta = row["dv_rating_beta"]
            title = row["O*NET-SOC 2019 Title"]
            occ_colors.append(color)
            occ_sizes.append(OCC_NODE_SIZE)

            is_selected = title == current_selection
            occ_line_widths.append(3 if is_selected else 1)
            occ_line_colors.append("#111111" if is_selected else "white")

            beta_text = f" ({beta:.0%})" if pd.notna(beta) else ""
            occ_labels.append(f"{title}{beta_text}")

            wage = row["Median Annual Wage 2024"]
            wage_text = f"${wage:,.0f}" if pd.notna(wage) else "N/A"

            growth_text = "N/A"
            if pd.notna(row.get("Employment 2034")) and row["Employment 2024"]:
                growth_pct = (
                    (row["Employment 2034"] - row["Employment 2024"])
                    / row["Employment 2024"] * 100
                )
                growth_text = f"{growth_pct:+.1f}% by 2034"

            occ_hover.append(
                f"<b>{title}</b><br>"
                f"AI exposure: {tier}" + (f" ({beta:.0%})" if pd.notna(beta) else "")
                + f"<br>Employment 2024: {row['Employment 2024']:,.0f}"
                + f"<br>Median wage: {wage_text}"
                + f"<br>Projected growth: {growth_text}"
                + "<br><i>Click to see full profile</i>"
            )

        fig.add_trace(go.Scatter(
            x=[occ_x] * n_occ,
            y=occ_y,
            mode="markers+text",
            marker=dict(
                size=occ_sizes,
                color=occ_colors,
                line=dict(width=occ_line_widths, color=occ_line_colors),
            ),
            text=occ_labels,
            textposition="middle right",
            textfont=dict(size=12, color="#222222"),
            hovertext=occ_hover,
            hoverinfo="text",
            customdata=fan_df["O*NET-SOC 2019 Title"],
            showlegend=False,
        ))

        # Legend
        for tier_name, tier_color in [
            ("Low", "#2CA02C"),
            ("Moderate", "#F2C744"),
            ("High", "#E67E22"),
            ("Very High", "#B22222"),
        ]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=tier_color),
                name=tier_name,
                showlegend=True,
            ))

        if n_occ < len(career_df.dropna(subset=["Employment 2024"])):
            st.caption(f"Showing the top {n_occ} occupations for {selected_cip} by 2024 employment.")

        fig.update_layout(
            height=max(480, 42 * n_occ),
            margin=dict(l=10, r=40, t=50, b=10),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="left", x=0,
                title="AI exposure",
                font=dict(size=12),
            ),
            xaxis=dict(visible=False, range=[-2.2, occ_x + 4.5]),
            yaxis=dict(visible=False),
            plot_bgcolor="#FAFAF8",
            paper_bgcolor="#FAFAF8",
            font=dict(family="Helvetica, Arial, sans-serif"),
        )

        fan_event = st.plotly_chart(
            fig,
            use_container_width=True,
            key="major_fan_chart",
            on_select="rerun",
            selection_mode="points",
        )

        if fan_event.selection.points:
            clicked_title = fan_event.selection.points[0].get("customdata")
            if clicked_title:
                st.session_state.selected_occupation = clicked_title
                st.rerun()

    # -----------------------------------------------------
    # Right: occupation profile card
    # -----------------------------------------------------

    with profile_col:

        st.markdown("**OCCUPATION PROFILE**")

        current_selection = st.session_state.get("selected_occupation")

        if not current_selection:
            st.caption("Click an occupation to see its full profile.")
        else:
            prof_row = fan_df[fan_df["O*NET-SOC 2019 Title"] == current_selection]

            if prof_row.empty:
                st.caption("No profile data available for this occupation yet.")
            else:
                prof_row = prof_row.iloc[0]
                beta = prof_row.get("dv_rating_beta")
                tier, tier_color = _exposure_tier(beta)

                st.markdown(f"**{current_selection}**")
                st.caption(f"SOC {prof_row.get('Occupation Code', 'N/A')}")

                occ_description = occupation_description_lookup.get(prof_row.get("O*NET Code"))
                if occ_description:
                    st.markdown(
                        f"<p style='font-size:0.85rem; color:#4A4A4A; "
                        f"line-height:1.4; margin-bottom:10px;'>{occ_description}</p>",
                        unsafe_allow_html=True,
                    )

                if pd.notna(beta):
                    st.markdown(
                        f"""
                        <div class="exposure-banner" style="background:{tier_color};">
                            🤖 AI exposure (β): {beta:.0%} · {tier}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                wage = prof_row.get("Median Annual Wage 2024")
                wage_display = f"${wage:,.0f}" if pd.notna(wage) else "N/A"
                st.metric("Median wage", wage_display)

                emp = prof_row.get("Employment 2024")
                employment_display = f"{emp:,.1f}k" if pd.notna(emp) else "N/A"
                st.metric("Employment 2024", employment_display)

                growth_display = "N/A"
                growth_pct = None
                if pd.notna(prof_row.get("Employment 2034")) and emp:
                    growth_pct = (prof_row["Employment 2034"] - emp) / emp * 100
                    growth_display = f"{growth_pct:+.1f}% by 2034"
                    # delta_color="normal" colors the delta green for
                    # positive growth and red for negative, so the
                    # sign is visible at a glance rather than always
                    # rendering in the same neutral color.
                    st.metric(
                        "Projected growth",
                        growth_display,
                        delta=f"{growth_pct:+.1f}%",
                        delta_color="normal",
                    )

                openings = prof_row.get("Occupational Openings, 2024-2034 Annual Average")
                openings_display = f"{openings:,.1f}k" if pd.notna(openings) else "N/A"
                if pd.notna(openings):
                    st.metric("Annual openings", openings_display)

                education = prof_row.get("Typical Entry-Level Education", "N/A")
                st.metric("Education", education)

                soc_code = prof_row.get("Occupation Code")
                related_majors_for_job = [
                    m for m in occupation_majors_lookup.get(soc_code, [])
                    if m != selected_cip
                ]

                st.markdown("**Related majors**")
                if related_majors_for_job:
                    MAX_JOB_RELATED_MAJORS = 5
                    for m in related_majors_for_job[:MAX_JOB_RELATED_MAJORS]:
                        if st.button(
                            m,
                            key=f"occ_related_major_{current_selection}_{m}",
                        ):
                            st.session_state.pending_major = m
                            st.rerun()
                    if len(related_majors_for_job) > MAX_JOB_RELATED_MAJORS:
                        st.caption(f"+{len(related_majors_for_job) - MAX_JOB_RELATED_MAJORS} more")
                else:
                    st.caption("No other majors in the crosswalk lead to this occupation.")

                already_saved = current_selection in st.session_state.saved_jobs

                if st.button(
                    "✅ Saved" if already_saved else "💾 Save Job",
                    key=f"save_job_{current_selection}",
                    use_container_width=True,
                    disabled=already_saved,
                ):
                    st.session_state.saved_jobs[current_selection] = {
                        "title": current_selection,
                        "major": selected_cip,
                        "soc_code": prof_row.get("Occupation Code", "N/A"),
                        "description": occ_description,
                        "beta_display": f"{beta:.0%} · {tier}" if pd.notna(beta) else "N/A",
                        "tier_color": tier_color,
                        "wage_display": wage_display,
                        "employment_display": employment_display,
                        "growth_display": growth_display,
                        "growth_pct": growth_pct,
                        "openings_display": openings_display,
                        "education": education,
                        "related_majors": related_majors_for_job,
                    }
                    st.rerun()


# ---------------------------------------------------
# Saved Jobs
# ---------------------------------------------------
#
# A simple side-by-side comparison board: every occupation the
# student has saved from the profile panel above, shown as a card
# with the same info (wage, employment, growth, openings, education,
# AI exposure), with a way to remove it from the list.

st.divider()
st.header("Saved Jobs")

if not st.session_state.saved_jobs:

    st.caption("Save an occupation from the profile panel above to start comparing jobs here.")

else:

    CARDS_PER_ROW = 3
    saved_list = list(st.session_state.saved_jobs.values())

    for row_start in range(0, len(saved_list), CARDS_PER_ROW):

        row_jobs = saved_list[row_start:row_start + CARDS_PER_ROW]
        row_cols = st.columns(CARDS_PER_ROW)

        for col, job in zip(row_cols, row_jobs):

            with col:
                with st.container(border=True):

                    st.markdown(f"**{job['title']}**")
                    st.caption(f"SOC {job['soc_code']} · via {job['major']}")

                    if job.get("description"):
                        st.caption(job["description"])

                    st.markdown(
                        f"""
                        <span class="alt-badge" style="background:{job['tier_color']}; color:white;">
                            AI exposure {job['beta_display']}
                        </span>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.write(f"**Median wage:** {job['wage_display']}")
                    st.write(f"**Employment 2024:** {job['employment_display']}")

                    growth_pct_saved = job.get("growth_pct")
                    if growth_pct_saved is not None:
                        growth_color_saved = "#2CA02C" if growth_pct_saved >= 0 else "#B22222"
                        st.markdown(
                            f"**Projected growth:** "
                            f"<span style='color:{growth_color_saved}; font-weight:700;'>"
                            f"{job['growth_display']}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write(f"**Projected growth:** {job['growth_display']}")

                    st.write(f"**Annual openings:** {job['openings_display']}")
                    st.write(f"**Education:** {job['education']}")

                    related_majors_saved = job.get("related_majors", [])
                    if related_majors_saved:
                        st.write(f"**Related majors:** {', '.join(related_majors_saved)}")
                    else:
                        st.write("**Related majors:** None")

                    if st.button(
                        "🗑 Remove",
                        key=f"remove_saved_{job['title']}",
                        use_container_width=True,
                    ):
                        del st.session_state.saved_jobs[job["title"]]
                        st.rerun()