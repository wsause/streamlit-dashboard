import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import html


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
        "Employment_Projections.csv"
    )

    occupation_data_df = pd.read_csv(
        "Occupation_Data.csv"
    )

    openai_df = pd.read_csv(
        "occ_level.csv"
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

    return (
        cip_df,
        bls_df,
        occupation_data_df,
        openai_df,
        abilities_df,
        skills_df,
        activities_df
    )


(
    cip_df,
    bls_df,
    occupation_data_df,
    openai_df,
    abilities_df,
    skills_df,
    activities_df

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
    .alt-wage {
        display: inline-block;
        color: #374151;
        font-weight: 600;
        font-size: 0.85rem;
        margin-left: 6px;
    }
    .alt-shared {
        color: #8A8A8A;
        font-size: 0.85rem;
        margin-top: 6px;
    }
    /* Dotted underline signals that "N shared occupations" is
       hoverable -- hovering it shows the actual occupation names
       via the same fast CSS tooltip used for job descriptions. */
    .alt-shared-hoverable {
        cursor: help;
        text-decoration: underline dotted;
        text-underline-offset: 3px;
        width: fit-content;
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
    /* Shrink st.metric's value text (only used in the occupation
       profile panel) and let it wrap instead of truncating with an
       ellipsis, so longer values like "Bachelor's degree" display
       in full instead of cutting off mid-word. */
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.3 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    /* Small info icon next to the occupation title, with a custom
       CSS tooltip (rather than the browser's native `title`
       attribute) so the occupation description appears almost
       instantly on hover instead of after the browser's built-in
       ~600ms-1s delay. */
    .title-info-wrapper {
        position: relative;
        display: inline-block;
    }
    .title-info-icon {
        cursor: help;
        margin-left: 6px;
        font-size: 0.85rem;
    }
    .title-info-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        top: 125%;
        z-index: 1000;
        background: #1F2937;
        color: white;
        padding: 8px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 400;
        line-height: 1.4;
        width: 260px;
        max-width: 85vw;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        transition: opacity 0.05s ease-in;
    }
    /* Which edge the tooltip grows from depends on where its icon
       sits on the page -- "anchor-right" (grows leftward) for
       icons near the right edge (profile panel, saved-job cards),
       "anchor-left" (grows rightward) for icons near the left edge
       (the related-majors panel). Without this, a single fixed
       direction runs off whichever edge the icon is closest to. */
    .title-info-tooltip.anchor-right {
        right: 0;
    }
    .title-info-tooltip.anchor-left {
        left: 0;
    }
    .title-info-wrapper:hover .title-info-tooltip {
        visibility: visible;
        opacity: 1;
    }
    /* Tighten the native bordered container used for each
       alternative-major card (and the "your major" card) so the
       link button / title sits snugly with the stats beneath it. */
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
    /* HTML/CSS "meter list" used for the Skills/Abilities/Work
       Activities section: one row per element, a label, a filled
       horizontal bar sized to its Importance score, and the value. */
    .meter-list {
        margin-top: 4px;
    }
    .meter-row {
        margin-bottom: 10px;
    }
    .meter-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 3px;
    }
    .meter-track-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .meter-track {
        flex: 1;
        height: 8px;
        border-radius: 4px;
        background: #E5E7EB;
        overflow: hidden;
    }
    .meter-fill {
        height: 100%;
        border-radius: 4px;
    }
    .meter-value {
        font-size: 0.78rem;
        font-weight: 700;
        color: #374151;
        width: 24px;
        text-align: right;
        flex-shrink: 0;
    }
    .meter-level {
        font-size: 0.72rem;
        color: #9CA3AF;
        margin-top: 1px;
    }
    /* AI-exposure tier legend shown above the occupation bar chart --
       a plain HTML row (rather than Plotly's own legend) so it
       always spans the full chart column and sits flush above the
       whole chart, y-axis labels included. */
    .tier-legend {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 14px;
        margin: 4px 0 6px 0;
    }
    .tier-legend-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #374151;
    }
    .tier-legend-item {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.82rem;
        color: #374151;
    }
    .tier-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _exposure_tier(beta):
    """Five equal-width (20-point) bands across the full 0-100% range
    of AI exposure (beta), instead of the earlier four uneven bands
    (35/60/80 cutoffs). The old "Moderate" band spanned 25 points
    (35-60%) and visually swallowed most occupations into one color;
    splitting into five even 20-point bands spreads occupations
    across more colors instead of clustering into one.
    """
    if pd.isna(beta):
        return "Unknown", "#B0B0B0"
    if beta < 0.20:
        return "Very Low", "#2CA02C"
    if beta < 0.40:
        return "Low", "#8BC34A"
    if beta < 0.60:
        return "Moderate", "#F2C744"
    if beta < 0.80:
        return "High", "#E67E22"
    return "Very High", "#B22222"


def _metric_html(label, value, color="#0F1116"):
    """A label+value block styled to match st.metric's (shrunk) size,
    but with a caller-supplied text color -- used for the projected
    growth figure so it can be colored green/red without st.metric's
    separate, differently-sized delta chip duplicating the value.
    """
    return (
        '<div style="margin-bottom:1rem;">'
        f'<div style="font-size:0.8rem; color:#6B7280;">{label}</div>'
        f'<div style="font-size:1.35rem; font-weight:600; color:{color}; line-height:1.3;">{value}</div>'
        '</div>'
    )


def _top_elements_for_occupation(df, onet_code, top_n=10):
    """Pivot an O*NET content-model file (Abilities / Essential Skills /
    Work Activities) down to the top N elements for a single occupation.

    These three files share the same shape: one row per
    (occupation, element, scale), where Scale ID "IM" = Importance
    (1-5) and "LV" = Level -- O*NET's own definition of Level is
    "the amount needed on the job" (0-7). Ranked by Importance,
    since that's the more intuitive "how much does this matter for
    this job" number -- Level is shown as secondary detail under
    each row.
    """

    if onet_code is None:
        return pd.DataFrame(columns=["Element Name", "Importance", "Level"])

    subset = df[
        df["O*NET-SOC Code"].astype(str).str.strip() == str(onet_code).strip()
    ]

    if subset.empty:
        return pd.DataFrame(columns=["Element Name", "Importance", "Level"])

    pivoted = subset.pivot_table(
        index="Element Name",
        columns="Scale ID",
        values="Data Value",
        aggfunc="mean",
    ).reset_index()

    pivoted = pivoted.rename(columns={"IM": "Importance", "LV": "Level"})

    if "Importance" not in pivoted.columns:
        pivoted["Importance"] = np.nan
    if "Level" not in pivoted.columns:
        pivoted["Level"] = np.nan

    pivoted = (
        pivoted
        .dropna(subset=["Importance"])
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    return pivoted


def _meter_list_html(df, accent_color):
    """A compact HTML/CSS "meter list": one row per element, each a
    label above a horizontal filled bar sized to its Importance score
    (0-5) with the numeric value at the end. Level (0-7) is shown as
    small muted text under the label rather than a second visual
    channel, so the list stays skimmable at a glance -- no legend,
    no axis, no chart to interpret, just a ranked list of bars.
    """

    d = df.sort_values("Importance", ascending=False).reset_index(drop=True)

    rows_html = []
    for _, row in d.iterrows():
        pct = max(0.0, min(100.0, (row["Importance"] / 5) * 100))
        level_html = ""
        if pd.notna(row.get("Level")):
            level_html = f'<div class="meter-level">Amount needed on job: {row["Level"]:.1f}/7</div>'

        rows_html.append(
            '<div class="meter-row">'
            f'<div class="meter-label">{html.escape(str(row["Element Name"]))}</div>'
            '<div class="meter-track-row">'
            '<div class="meter-track">'
            f'<div class="meter-fill" style="width:{pct:.0f}%; background:{accent_color};"></div>'
            '</div>'
            f'<div class="meter-value">{row["Importance"]:.1f}</div>'
            '</div>'
            f'{level_html}'
            '</div>'
        )

    return '<div class="meter-list">' + "".join(rows_html) + '</div>'


@st.cache_data
def compute_major_exposure(cip_df, occ_beta_df, bls_df):
    """Average AI exposure, median wage, and SOC code set for every
    major (CIP title).

    Used to build the "related majors" panel: for any given major we
    can look up other majors that lead to at least one of the same
    occupations, and compare their average exposure scores and wages.

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

    merged = merged.merge(
        bls_df[["Occupation Code", "Median Annual Wage 2024"]],
        on="Occupation Code",
        how="left"
    )

    grouped = (
        merged
        .groupby("2020 CIP Title")
        .agg(
            avg_beta=("dv_rating_beta", "mean"),
            median_wage=("Median Annual Wage 2024", "median"),
            soc_codes=("Occupation Code", lambda s: frozenset(s))
        )
        .reset_index()
    )

    return grouped


major_exposure_df = compute_major_exposure(cip_df, occ_beta_df, bls_df)


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
    key="major_select",
    label_visibility="collapsed",
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

st.caption(
    "Note: AI exposure (β) estimates how much of a job's tasks could be done at "
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
        .reset_index(drop=True)
    )

    # Projected growth %, computed once and reused by both the bar
    # chart (when growth is the chosen sort metric) and every hover
    # tooltip -- avoids recomputing this per-row in multiple places.
    fan_df["Growth Pct"] = np.where(
        fan_df["Employment 2024"].notna()
        & fan_df["Employment 2034"].notna()
        & (fan_df["Employment 2024"] != 0),
        (fan_df["Employment 2034"] - fan_df["Employment 2024"]) / fan_df["Employment 2024"] * 100,
        np.nan,
    )

    n_occ = len(fan_df)

    left_col, chart_col, profile_col = st.columns([1, 3.8, 1.1])

    # -----------------------------------------------------
    # Left: your major + lower-exposure alternatives
    # -----------------------------------------------------

    with left_col:

        st.markdown("**YOUR MAJOR**")

        your_tier, your_badge_color = _exposure_tier(selected_avg_beta)
        your_wage_display = (
            f"${selected_median_wage:,.0f}"
            if pd.notna(selected_median_wage)
            else "N/A"
        )

        # Styled the same way as the RELATED MAJORS cards below (a
        # bordered container with a bold title and an exposure badge
        # + median wage line) instead of a plain st.info box, so the
        # two sections read as one consistent card style.
        with st.container(border=True):
            st.markdown(
                f'<span style="font-weight:700; font-size:1rem;">{selected_cip}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <span class="alt-badge" style="background:{your_badge_color}; color:white;">
                    avg β {selected_avg_beta:.0%}
                </span>
                <span class="alt-wage">median wage {your_wage_display}</span>
                """,
                unsafe_allow_html=True,
            )

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

                rel_wage_display = (
                    f"${rel['median_wage']:,.0f}"
                    if pd.notna(rel["median_wage"])
                    else "N/A"
                )

                # The occupations (as shown in this major's own bar
                # chart) that overlap with the related major -- shown
                # in a hover tooltip on "N shared occupations" so a
                # student can see exactly which ones without needing
                # to cross-reference the chart themselves.
                shared_codes = rel["soc_codes"] & selected_codes
                shared_titles = sorted(
                    fan_df.loc[
                        fan_df["Occupation Code"].isin(shared_codes),
                        "O*NET-SOC 2019 Title"
                    ].unique()
                )
                shared_titles_html = html.escape("; ".join(shared_titles))

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
                        <span style="color:{diff_color}; font-weight:600; font-size:0.85rem; margin-left:6px;">
                            {arrow} {abs(rel['pts_diff']):.0f} pts {direction_word}
                        </span>
                        <span class="alt-wage">median wage {rel_wage_display}</span>
                        <br>
                        <span class="title-info-wrapper">
                            <div class="alt-shared alt-shared-hoverable">{rel['shared_count']} shared occupations</div>
                            <span class="title-info-tooltip anchor-left">{shared_titles_html}</span>
                        </span>
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

        current_selection = st.session_state.get("selected_occupation")

        # Which metric sets each bar's length. All four sort options
        # share the same color encoding (exposure tier), so switching
        # the sort never changes what the colors mean -- only what
        # order the list is in and what the bar length represents.
        SORT_OPTIONS = {
            "Employment": ("Employment 2024", "employment"),
            "Median wage": ("Median Annual Wage 2024", "wage"),
            "Projected growth": ("Growth Pct", "growth"),
            "AI exposure": ("dv_rating_beta", "exposure"),
        }

        sort_label = st.radio(
            "Sort occupations by",
            list(SORT_OPTIONS.keys()),
            horizontal=True,
            label_visibility="collapsed",
            key="occ_sort_metric",
        )
        sort_col, metric_kind = SORT_OPTIONS[sort_label]

        bar_df = (
            fan_df
            .dropna(subset=[sort_col])
            .sort_values(sort_col, ascending=False)
            .head(MAX_FAN_OCCUPATIONS)
            .reset_index(drop=True)
        )

        n_bars = len(bar_df)

        if n_bars == 0:

            st.info("No occupations have data for this sort option.")

        else:

            bar_colors, bar_values, bar_hover = [], [], []
            bar_line_widths, bar_line_colors = [], []

            for i in range(n_bars):
                row = bar_df.loc[i]
                tier, color = _exposure_tier(row["dv_rating_beta"])
                beta = row["dv_rating_beta"]
                title = row["O*NET-SOC 2019 Title"]

                bar_colors.append(color)

                is_selected = title == current_selection
                bar_line_widths.append(2.5 if is_selected else 0)
                bar_line_colors.append("#111111" if is_selected else color)

                if metric_kind == "exposure":
                    bar_values.append(beta * 100 if pd.notna(beta) else None)
                else:
                    bar_values.append(row[sort_col])

                wage = row["Median Annual Wage 2024"]
                wage_text = f"${wage:,.0f}" if pd.notna(wage) else "N/A"
                growth_val = row["Growth Pct"]
                growth_text = f"{growth_val:+.1f}% by 2034" if pd.notna(growth_val) else "N/A"

                bar_hover.append(
                    f"<b>{title}</b><br>"
                    f"AI exposure: {tier}" + (f" ({beta:.0%})" if pd.notna(beta) else "")
                    + f"<br>Employment 2024: {row['Employment 2024']:,.0f}"
                    + f"<br>Median wage: {wage_text}"
                    + f"<br>Projected growth: {growth_text}"
                    + "<br><i>Click to see full profile</i>"
                )

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=bar_values,
                y=bar_df["O*NET-SOC 2019 Title"],
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    line=dict(width=bar_line_widths, color=bar_line_colors),
                ),
                hovertext=bar_hover,
                hoverinfo="text",
                customdata=bar_df["O*NET-SOC 2019 Title"],
                showlegend=False,
            ))

            # AI-exposure tier legend, rendered as a plain HTML/CSS row
            # above the chart rather than a Plotly legend -- Plotly
            # positions its legend relative to the *plot area*, which
            # sits well to the right of the (often long) occupation
            # labels on the y-axis; getting it to visually sit flush
            # above those labels meant fighting Plotly's paper/
            # container coordinate systems. A plain HTML row placed
            # here, above st.plotly_chart, is laid out by Streamlit
            # itself and always spans the full column, directly above
            # the whole chart -- labels included.
            TIER_LEGEND = [
                ("Very Low", "#2CA02C"),
                ("Low", "#8BC34A"),
                ("Moderate", "#F2C744"),
                ("High", "#E67E22"),
                ("Very High", "#B22222"),
            ]
            legend_items_html = "".join(
                f'<span class="tier-legend-item">'
                f'<span class="tier-dot" style="background:{tier_color};"></span>{tier_name}'
                f'</span>'
                for tier_name, tier_color in TIER_LEGEND
            )
            st.markdown(
                f'<div class="tier-legend">'
                f'<span class="tier-legend-title">AI exposure:</span>'
                f'{legend_items_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

            if n_bars < n_occ:
                st.caption(
                    f"Showing {n_bars} of {n_occ} occupations for {selected_cip} "
                    f"with {sort_label.lower()} data, sorted highest to lowest."
                )
            else:
                st.caption(f"All {n_occ} occupations for {selected_cip}, sorted highest to lowest.")

            axis_titles = {
                "employment": "Employment 2024 (thousands)",
                "wage": "Median annual wage ($)",
                "growth": "Projected growth by 2034 (%)",
                "exposure": "AI exposure (β, %)",
            }
            tick_formats = {
                "employment": ",.0f",
                "wage": "$,.0f",
                "growth": "+.1f",
                "exposure": ".0f",
            }
            tick_suffixes = {
                "employment": "",
                "wage": "",
                "growth": "%",
                "exposure": "%",
            }

            fig.update_layout(
                height=max(380, 24 * n_bars + 140),
                margin=dict(l=10, r=20, t=10, b=10),
                showlegend=False,
                xaxis=dict(
                    title=axis_titles[metric_kind],
                    tickformat=tick_formats[metric_kind],
                    ticksuffix=tick_suffixes[metric_kind],
                ),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="#FAFAF8",
                paper_bgcolor="#FAFAF8",
                font=dict(family="Helvetica, Arial, sans-serif", size=12),
            )

            bar_event = st.plotly_chart(
                fig,
                use_container_width=True,
                key="occupation_bar_chart",
                on_select="rerun",
                selection_mode="points",
            )

            if bar_event.selection.points:
                clicked_title = bar_event.selection.points[0].get("customdata")
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

                occ_description = occupation_description_lookup.get(prof_row.get("O*NET Code"))
                description_html = html.escape(occ_description) if occ_description else ""

                st.markdown(
                    f'<span style="font-weight:700; font-size:1rem;">{current_selection}</span>'
                    + (
                        '<span class="title-info-wrapper">'
                        '<span class="title-info-icon">ℹ️</span>'
                        f'<span class="title-info-tooltip anchor-right">{description_html}</span>'
                        '</span>'
                        if occ_description else ""
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(f"SOC {prof_row.get('Occupation Code', 'N/A')}")

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
                growth_color = "#6B7280"
                if pd.notna(prof_row.get("Employment 2034")) and emp:
                    growth_pct = (prof_row["Employment 2034"] - emp) / emp * 100
                    growth_display = f"{growth_pct:+.1f}%"
                    growth_color = "#2CA02C" if growth_pct >= 0 else "#B22222"

                # A single colored value, sized to match the other
                # (shrunk) st.metric values -- st.metric's built-in
                # delta would otherwise show the same number twice,
                # once uncolored and once colored, at two different
                # sizes.
                st.markdown(
                    _metric_html("Projected Growth by 2034", growth_display, growth_color),
                    unsafe_allow_html=True,
                )

                openings = prof_row.get("Occupational Openings, 2024-2034 Annual Average")
                openings_display = f"{openings:,.1f}k" if pd.notna(openings) else "N/A"
                if pd.notna(openings):
                    st.metric("Annual openings", openings_display)

                education = prof_row.get("Typical Entry-Level Education", "N/A")
                st.metric("Education", education)

                # BLS leaves this blank for occupations that don't
                # require prior work experience, rather than writing
                # "None" -- fill that in explicitly so the metric
                # doesn't render empty.
                work_experience = prof_row.get("Work Experience in a Related Occupation")
                work_experience_display = work_experience if pd.notna(work_experience) else "None"
                st.metric("Work Experience", work_experience_display)

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
                        "work_experience": work_experience_display,
                        "related_majors": related_majors_for_job,
                    }
                    st.rerun()

    # -----------------------------------------------------
    # Skills / Abilities / Work Activities detail
    # -----------------------------------------------------
    #
    # Shown once an occupation is selected, underneath the bar chart.
    # All three sections use the same HTML/CSS "meter list" (a label,
    # a filled bar sized to Importance, and the value) so they stay
    # easy to compare against each other -- only the accent color
    # changes per section. Level (0-7) is shown as small muted text
    # under each label rather than a second bar or axis, keeping the
    # list skimmable at a glance.

    current_selection = st.session_state.get("selected_occupation")

    if current_selection:

        detail_row = fan_df[fan_df["O*NET-SOC 2019 Title"] == current_selection]

        if not detail_row.empty:

            onet_code = detail_row.iloc[0].get("O*NET Code")

            st.divider()
            st.markdown(f"**Skills, Abilities & Work Activities — {current_selection}**")
            st.caption(
                "How important each is to this occupation, on a 1 (not important) to "
                "5 (extremely important) scale."
            )

            skl_col, abl_col, act_col = st.columns(3)

            sections = [
                (skl_col, "Top Skills", skills_df, "#1A56DB"),
                (abl_col, "Top Abilities", abilities_df, "#7C3AED"),
                (act_col, "Top Work Activities", activities_df, "#0E9F6E"),
            ]

            for section_col, section_title, source_df, accent_color in sections:
                with section_col:
                    st.markdown(f"*{section_title}*")
                    top_elements = _top_elements_for_occupation(source_df, onet_code)
                    if top_elements.empty:
                        st.caption("No data available for this occupation.")
                    else:
                        st.markdown(
                            _meter_list_html(top_elements, accent_color),
                            unsafe_allow_html=True,
                        )


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

        for col_idx, (col, job) in enumerate(zip(row_cols, row_jobs)):

            # The leftmost card's icon sits near the left edge of the
            # page, so its tooltip should grow rightward; every other
            # card is closer to (or at) the right edge, so theirs
            # grow leftward -- same edge-aware logic as the profile
            # panel and related-majors tooltips.
            tooltip_anchor = "anchor-left" if col_idx == 0 else "anchor-right"

            with col:
                with st.container(border=True):

                    job_description_html = html.escape(job["description"]) if job.get("description") else ""
                    st.markdown(
                        f'<span style="font-weight:700;">{job["title"]}</span>'
                        + (
                            '<span class="title-info-wrapper">'
                            '<span class="title-info-icon">ℹ️</span>'
                            f'<span class="title-info-tooltip {tooltip_anchor}">{job_description_html}</span>'
                            '</span>'
                            if job.get("description") else ""
                        ),
                        unsafe_allow_html=True,
                    )
                    st.caption(f"SOC {job['soc_code']} · via {job['major']}")

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
                            f"**Projected Growth by 2034:** "
                            f"<span style='color:{growth_color_saved}; font-weight:700;'>"
                            f"{job['growth_display']}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write(f"**Projected Growth by 2034:** {job['growth_display']}")

                    st.write(f"**Annual openings:** {job['openings_display']}")
                    st.write(f"**Education:** {job['education']}")
                    st.write(f"**Work Experience:** {job.get('work_experience', 'None')}")

                    related_majors_saved = job.get("related_majors", [])
                    if related_majors_saved:
                        st.write(f"**Related majors:** {'; '.join(related_majors_saved)}")
                    else:
                        st.write("**Related majors:** None")

                    if st.button(
                        "🗑 Remove",
                        key=f"remove_saved_{job['title']}",
                        use_container_width=True,
                    ):
                        del st.session_state.saved_jobs[job["title"]]
                        st.rerun()