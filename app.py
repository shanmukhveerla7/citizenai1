import streamlit as st
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# Page config
st.set_page_config(page_title="🌆 Citizen Dashboard", layout="wide", page_icon="🌆")

# Language support
LANGUAGES = {
    "en": {
        "title": "🌆 Citizen Dashboard",
        "subtitle": "Explore insights about your city.",
        "insights": "🧠 City Insights",
        "services": "🚑 Public Services",
        "footer": "© 2025 Citizen Dashboard | Built with ❤️ using Streamlit & Watsonx"
    }
}

# Initialize session state
if "current_section" not in st.session_state:
    st.session_state.current_section = "insights"

# Custom CSS - Dual Theme UI
st.markdown("""
    <style>
        body { background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; }
        .main { background-color: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        .navbar { display: flex; justify-content: center; gap: 20px; padding: 15px 0; margin-bottom: 25px; }
        .nav-button {
            background-color: #ffffff;
            color: #3498db;
            border: 2px solid #3498db;
            width: 150px;
            height: 50px;
            font-size: 16px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .nav-button:hover {
            background-color: #def8ff;
            transform: scale(1.05);
        }
        h1, h2, h3 { color: #2c3e50; }
        label { font-weight: bold; color: #34495e; }
        .card-blue {
            background-color: #ebf5fb;
            padding: 20px;
            border-left: 6px solid #2980b9;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin: 20px 0;
        }
        .card-green {
            background-color: #ecf5eb;
            padding: 20px;
            border-left: 6px solid #27ae60;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin: 20px 0;
        }
        .button-blue {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 14px;
            border-radius: 8px;
            cursor: pointer;
        }
        .button-blue:hover {
            background-color: #2980b9;
        }
        .button-green {
            background-color: #2ecc71;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 14px;
            border-radius: 8px;
            cursor: pointer;
        }
        .button-green:hover {
            background-color: #27ae60;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 40px;
            color: #7f8c8d;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation Bar
def render_navbar():
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧠 City Insights", key="btn_insights", use_container_width=True):
            st.session_state.current_section = "insights"
    with col2:
        if st.button("🚑 Public Services", key="btn_services", use_container_width=True):
            st.session_state.current_section = "services"
    st.markdown('</div>', unsafe_allow_html=True)

# Header
lang = "en"
st.markdown(f'<h1 style="text-align:center;">{LANGUAGES[lang]["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center; font-size:16px;">{LANGUAGES[lang]["subtitle"]}</p>', unsafe_allow_html=True)
render_navbar()

# Load Watsonx credentials
try:
    credentials = {
        "url": st.secrets["WATSONX_URL"],
        "apikey": st.secrets["WATSONX_APIKEY"]
    }
    project_id = st.secrets["WATSONX_PROJECT_ID"]
except KeyError as e:
    st.warning(f"⚠️ Missing Watsonx credential: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error initializing LLM: {str(e)}")
    st.stop()

model_map = {
    "insights": "ibm/granite-13b-instruct-v2",
    "services": "ibm/granite-13b-instruct-v2"
}

def get_llm(model_name):
    return WatsonxLLM(
        model_id=model_map[model_name],
        url=credentials.get("url"),
        apikey=credentials.get("apikey"),
        project_id=project_id,
        params={
            GenParams.DECODING_METHOD: "greedy",
            GenParams.TEMPERATURE: 0.7,
            GenParams.MIN_NEW_TOKENS: 5,
            GenParams.MAX_NEW_TOKENS: 300,
            GenParams.STOP_SEQUENCES: ["Human:", "Observation"],
        },
    )

# ------------------------------ DASHBOARD 1: CITY INSIGHTS ------------------------------
if st.session_state.current_section == "insights":
    st.markdown('<div class="card-blue">', unsafe_allow_html=True)
    st.markdown('<h2>🧠 City Insights</h2>', unsafe_allow_html=True)
    query = st.text_area("Ask something about the city:", placeholder="e.g., What's the traffic like today?")
    if st.button("🔍 Get Insight", key="insight_button", help="Get AI-powered insight"):
        if query.strip():
            llm = get_llm("insights")
            res = llm.invoke(query)
            st.markdown(f'<div class="ai-analysis">{res}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a question.")
    st.markdown('</div>')

# ------------------------------ DASHBOARD 2: PUBLIC SERVICES ------------------------------
elif st.session_state.current_section == "services":
    st.markdown('<div class="card-green">', unsafe_allow_html=True)
    st.markdown('<h2>🚑 Public Services</h2>', unsafe_allow_html=True)
    service_query = st.text_area("What would you like to know about public services?", placeholder="e.g., Where is the nearest hospital?")
    if st.button("🏥 Find Service", key="service_button", help="Find location or info"):
        if service_query.strip():
            llm = get_llm("services")
            res = llm.invoke(service_query)
            st.markdown(f'<div class="ai-analysis">{res}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please describe your request.")
    st.markdown('</div>')

# Footer
st.markdown(f'<p class="footer">{LANGUAGES[lang]["footer"]}</p>', unsafe_allow_html=True)
