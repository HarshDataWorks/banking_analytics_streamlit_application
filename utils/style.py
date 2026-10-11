"""
Shared visual style for the whole app — colors reverse-engineered from the
project's own final PPTX deliverable (PRP_Project2_Final_Deliverable_Presentation.pptx),
so the Streamlit app, the dashboards, and the slide deck all look like one project
rather than three unrelated artifacts.
"""

import streamlit as st

# Color system (from the final PPTX build)
NAVY = "#002060"
NAVY2 = "#1E2761"
PURPLE = "#7030A0"
LPURPLE = "#E0C1FF"
TEAL = "#348E90"
BERRY = "#993366"
BODY = "#23262F"
GRAY = "#5F5F5F"
LBG = "#E3E7F0"
LBG2 = "#F4F6FB"

# Risk / fraud tier colors, used consistently on every dashboard page
LOW_COLOR = "#2E8B57"      # seagreen
MEDIUM_COLOR = "#D4A017"   # goldenrod
HIGH_COLOR = "#B22222"     # firebrick

PLOTLY_SEQUENCE = [NAVY, TEAL, PURPLE, BERRY, "#5F8FD1", "#8FBF8F"]


def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #FFFFFF;
        }}
        h1, h2, h3 {{
            color: {NAVY2};
            font-family: 'Georgia', 'Cambria', serif;
        }}
        div[data-testid="stMetric"] {{
            background-color: {LBG2};
            border: 1px solid {LBG};
            border-radius: 10px;
            padding: 12px 14px 6px 14px;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {GRAY};
        }}
        .prp-banner {{
            background: linear-gradient(90deg, {NAVY} 0%, {NAVY2} 60%, {PURPLE} 100%);
            padding: 22px 28px;
            border-radius: 12px;
            color: white;
            margin-bottom: 18px;
        }}
        .prp-banner h1 {{
            color: white !important;
            margin-bottom: 4px;
        }}
        .prp-banner p {{
            color: #E0C1FF;
            margin: 0;
            font-size: 0.95rem;
        }}
        .prp-tag {{
            display: inline-block;
            background-color: {TEAL};
            color: white;
            border-radius: 6px;
            padding: 2px 10px;
            font-size: 0.78rem;
            margin-right: 6px;
        }}
        .prp-callout {{
            background-color: {LBG2};
            border-left: 5px solid {TEAL};
            padding: 12px 16px;
            border-radius: 6px;
            margin: 10px 0;
        }}
        .prp-callout-warn {{
            background-color: #FFF4E5;
            border-left: 5px solid {BERRY};
            padding: 12px 16px;
            border-radius: 6px;
            margin: 10px 0;
        }}
        .prp-finding {{
            background-color: #EAF6F0;
            border-left: 5px solid {LOW_COLOR};
            padding: 12px 16px;
            border-radius: 6px;
            margin: 10px 0;
        }}
        footer {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, tag: str = ""):
    tag_html = f'<span class="prp-tag">{tag}</span>' if tag else ""
    st.markdown(
        f"""
        <div class="prp-banner">
            {tag_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "info"):
    cls = "prp-callout" if kind == "info" else ("prp-finding" if kind == "finding" else "prp-callout-warn")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)
