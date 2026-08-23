
import streamlit as st
import pandas as pd
import plotly.express as px

from rdkit import Chem
from rdkit.Chem import Draw


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI-Driven Drug Discovery",
    page_icon="🧬",
    layout="wide"
)


# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.4rem;
}

.subtitle {
    font-size: 1rem;
    color: #9CA3AF;
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 1.7rem;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.pipeline-box {
    padding: 16px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.12);
    text-align: center;
    font-weight: 650;
    min-height: 72px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.arrow {
    text-align: center;
    font-size: 1.8rem;
    padding-top: 14px;
}

.highlight-card {
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 1rem;
}

.small-note {
    font-size: 0.9rem;
    color: #9CA3AF;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# LOAD DATA
# -----------------------------

@st.cache_data
def load_data():

    ranked = pd.read_csv("final_ranked_compounds.csv")
    models = pd.read_csv("model_comparison.csv")
    shap_df = pd.read_csv("final_lead_SHAP.csv")
    summary = pd.read_csv("project_summary.csv")

    return ranked, models, shap_df, summary


ranked, models, shap_df, summary = load_data()


# -----------------------------
# HEADER
# -----------------------------

st.markdown(
    '<div class="main-title">🧬 AI-Driven Predictive Modeling for Drug Discovery</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Integrating Disease–Gene Associations, Drug Target Identification,
    Molecular Activity Prediction, ADMET-Oriented Profiling,
    Lead Prioritization and Explainable AI
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Project Overview",
        "AI Model Performance",
        "Lead Prioritization",
        "Top Lead Analysis",
        "Explainable AI"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Computational drug-discovery prototype for Alzheimer's disease"
)


# =========================================================
# PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.markdown(
        '<div class="section-title">🔬 Drug Discovery Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Disease",
        "Alzheimer's"
    )

    c2.metric(
        "Selected Target",
        "APP"
    )

    c3.metric(
        "Compounds Screened",
        f"{len(ranked):,}"
    )

    c4.metric(
        "Best AI Model",
        "XGBoost"
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Integrated Pipeline</div>',
        unsafe_allow_html=True
    )

    steps = [
        "Disease Selection",
        "Disease–Gene Association",
        "Drug Target Identification",
        "ChEMBL Compound Retrieval",
        "RDKit Molecular Features",
        "Random Forest & XGBoost",
        "ADMET-Oriented Screening",
        "Lead Prioritization",
        "SHAP Explainability"
    ]

    for i in range(0, len(steps), 3):

        cols = st.columns(5)

        chunk = steps[i:i+3]

        for j, step in enumerate(chunk):

            cols[j*2].markdown(
                f'<div class="pipeline-box">{step}</div>',
                unsafe_allow_html=True
            )

            if j < len(chunk) - 1:

                cols[j*2 + 1].markdown(
                    '<div class="arrow">➜</div>',
                    unsafe_allow_html=True
                )

        if i + 3 < len(steps):
            st.markdown(
                "<div style='text-align:center;font-size:1.8rem;'>⬇</div>",
                unsafe_allow_html=True
            )

    st.divider()

    st.markdown(
        '<div class="section-title">Project Highlights</div>',
        unsafe_allow_html=True
    )

    h1, h2, h3 = st.columns(3)

    h1.info(
        "Disease-gene associations obtained using Open Targets."
    )

    h2.info(
        "1,416 APP-related compounds processed from ChEMBL."
    )

    h3.info(
        "Morgan fingerprints and RDKit descriptors used for AI prediction."
    )

    st.warning(
        "The platform performs computational prioritization only. "
        "It does not establish clinical efficacy or safety."
    )


# =========================================================
# AI MODEL PERFORMANCE
# =========================================================

elif page == "AI Model Performance":

    st.markdown(
        '<div class="section-title">🤖 AI Model Performance</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Best Random-Split R²",
        "0.8121"
    )

    c2.metric(
        "Scaffold-Split R²",
        "0.6284"
    )

    c3.metric(
        "Best Model",
        "XGBoost"
    )

    st.divider()

    st.subheader("Model Comparison")

    st.dataframe(
        models,
        use_container_width=True,
        hide_index=True
    )

    r2_col = None

    if "R²" in models.columns:
        r2_col = "R²"

    elif "R2" in models.columns:
        r2_col = "R2"

    if r2_col:

        fig = px.bar(
            models,
            x="Model",
            y=r2_col,
            text_auto=".3f",
            title="R² Comparison Across Machine Learning Models"
        )

        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="R² Score"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Validation Interpretation")

    st.success(
        "XGBoost achieved the best held-out random-split performance "
        "with R² = 0.8121."
    )

    st.info(
        "Scaffold-based validation was additionally performed to test "
        "generalization to structurally different chemical families."
    )

    st.caption(
        "The scaffold-split performance is lower because it represents "
        "a more difficult and more realistic generalization setting."
    )


# =========================================================
# LEAD PRIORITIZATION
# =========================================================

elif page == "Lead Prioritization":

    st.markdown(
        '<div class="section-title">🏆 Lead Compound Prioritization</div>',
        unsafe_allow_html=True
    )

    top10 = ranked.head(10).copy()

    display_columns = [
        "Rank",
        "molecule_chembl_id",
        "Predicted_pIC50",
        "Advanced_ADMET_Score",
        "BBB_Profile",
        "Drug_Likeness",
        "Final_Lead_Score_Percent"
    ]

    display_columns = [
        col for col in display_columns
        if col in top10.columns
    ]

    st.dataframe(
        top10[display_columns],
        use_container_width=True,
        hide_index=True
    )

    if "Final_Lead_Score_Percent" in top10.columns:

        fig = px.bar(
            top10,
            x="Final_Lead_Score_Percent",
            y="molecule_chembl_id",
            orientation="h",
            text_auto=".1f",
            title="Top 10 Prioritized Candidate Compounds"
        )

        fig.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            },
            xaxis_title="Lead Score (%)",
            yaxis_title="Compound"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info(
        "Lead ranking integrates predicted activity, "
        "ADMET-oriented physicochemical suitability and drug-likeness."
    )


# =========================================================
# TOP LEAD ANALYSIS
# =========================================================

elif page == "Top Lead Analysis":

    best = ranked.iloc[0]

    st.markdown(
        '<div class="section-title">🥇 Final Prioritized Lead</div>',
        unsafe_allow_html=True
    )

    st.success(
        f"Top Candidate: {best['molecule_chembl_id']}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Predicted pIC50",
        f"{best['Predicted_pIC50']:.4f}"
    )

    c2.metric(
        "ADMET Score",
        f"{best['Advanced_ADMET_Score']:.2f}"
    )

    c3.metric(
        "BBB Profile",
        best["BBB_Profile"]
    )

    c4.metric(
        "Lead Score",
        f"{best['Final_Lead_Score_Percent']:.2f}%"
    )

    st.divider()

    left, right = st.columns([1, 1.25])

    with left:

        st.subheader("Molecular Structure")
        st.subheader("Molecular Representation")

        mol = Chem.MolFromSmiles(
            best["canonical_smiles"]
        )

        if mol:

           img = Draw.MolToImage(
               mol,
               size=(550, 450)
           )

           st.image(
              img,
              use_container_width=True
           )
        

    with right:

        st.subheader("Molecular Profile")

        profile = pd.DataFrame({
            "Property": [
                "Compound ID",
                "Predicted pIC50",
                "Molecular Weight",
                "LogP",
                "TPSA",
                "BBB Suitability",
                "Drug-Likeness",
                "Advanced ADMET Score",
                "Final Lead Score"
            ],

            "Value": [
                best["molecule_chembl_id"],
                round(best["Predicted_pIC50"], 4),
                round(best["Molecular_Weight"], 2),
                round(best["LogP"], 2),
                round(best["TPSA"], 2),
                best["BBB_Profile"],
                best["Drug_Likeness"],
                round(best["Advanced_ADMET_Score"], 2),
                f"{best['Final_Lead_Score_Percent']:.2f}%"
            ]
        })

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            """
            <div class="highlight-card">
            <b>Why was this molecule prioritized?</b><br><br>
            It achieved a strong predicted target activity while also
            maintaining favourable physicochemical characteristics,
            CNS/BBB suitability and drug-likeness.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.warning(
        f"{best['molecule_chembl_id']} is a computationally prioritized "
        "candidate for further validation. It is not presented as a "
        "clinically validated Alzheimer's treatment."
    )


# =========================================================
# EXPLAINABLE AI
# =========================================================

elif page == "Explainable AI":

    best = ranked.iloc[0]

    st.markdown(
        '<div class="section-title">🧠 Explainable AI</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"SHAP analysis was used to interpret the XGBoost prediction "
        f"for the final lead compound **{best['molecule_chembl_id']}**."
    )

    shap_plot = shap_df.head(15).copy()

    value_col = None

    if "SHAP_Value" in shap_plot.columns:
        value_col = "SHAP_Value"

    elif "SHAP Value" in shap_plot.columns:
        value_col = "SHAP Value"

    if value_col:

        fig = px.bar(
            shap_plot,
            x=value_col,
            y="Feature",
            orientation="h",
            title=f"SHAP Feature Contributions for {best['molecule_chembl_id']}"
        )

        fig.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            },
            xaxis_title="Contribution to Predicted pIC50"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.success(
        "Positive SHAP values increase the predicted pIC50 relative "
        "to the model baseline, while negative values reduce it."
    )

    st.caption(
        "SHAP explains model behaviour. It does not establish biological causality."
    )
