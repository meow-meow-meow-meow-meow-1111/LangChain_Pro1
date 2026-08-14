import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(
    page_title="Wanderlust AI - Next-Gen Trip Architect",
    page_icon="✈️",
    layout="wide"
)

# 2. Custom UI Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8533 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model & Chain Setup (Cached for Performance)
@st.cache_resource
def load_trip_chain():
    # Retrieve the token safely from Streamlit Secrets or Environment Variables
    hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not hf_token:
        st.error("⚠️ Hugging Face API Token not found! Please add `HUGGINGFACEHUB_API_TOKEN` to Streamlit Secrets.")
        st.stop()

    # Hosted serverless model with verified text-generation pipeline support
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        max_new_tokens=1200,
        temperature=0.7,
        huggingfacehub_api_token=hf_token
    )
    
    template = PromptTemplate.from_template(
        """You are an elite travel architect and local insider.
Create a structured, inspiring {duration_days}-day travel itinerary for {destination} during {season}.
Budget Level: {budget_style}
Traveler Preferences / Interests: {interests}

Strict Output Format:
## 📍 Destination Brief & Vibe
(A 2-sentence summary of the vibe, culture, and budgeting reality)

## 🗓️ Day-by-Day Masterplan
### Day 1: [Theme / Focus Area]
- **Morning:** [Specific landmark/activity + local actionable tip]
- **Afternoon:** [Cultural spot, walk, or activity matching budget]
- **Evening:** [Scenic spot, twilight view, or night vibe]

## 🍽️ Local Culinary Highlights
- **Must-Try 1:** [Dish Name] — [Where to get it & why it fits {budget_style}]
- **Must-Try 2:** [Dish Name] — [Where to get it & why it fits {budget_style}]

## 💡 Local Insider Pro-Tip
[One practical cultural, transit, or money-saving secret]
"""
    )
    
    return template | llm | StrOutputParser()

# 4. Sidebar Controls
with st.sidebar:
    st.title("🗺️ Trip Parameters")
    st.caption("Customize your bespoke itinerary")
    
    destination = st.text_input("Destination", value="Kyoto, Japan")
    
    col1, col2 = st.columns(2)
    with col1:
        duration_days = st.slider("Days", min_value=1, max_value=5, value=3)
    with col2:
        season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter", "Monsoon"])
        
    budget_style = st.select_slider(
        "Budget Style",
        options=["Backpacker Budget", "Smart Casual / Mid-Range", "Luxury / High-End"],
        value="Smart Casual / Mid-Range"
    )
    
    interests = st.multiselect(
        "Interests / Focus",
        ["Hidden Gems", "Street Food", "Art & Architecture", "Nature & Scenery", "Nightlife", "History & Heritage"],
        default=["Hidden Gems", "Street Food"]
    )
    
    generate_btn = st.button("✨ Architect My Itinerary")

# 5. Main Content Area
st.title("✈️ Wanderlust AI Itinerary Architect")
st.caption("Powered by LangChain LCEL & Hugging Face Serverless Inference")

if generate_btn:
    if not destination.strip():
        st.error("⚠️ Please specify a destination in the sidebar.")
    else:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Destination: <b>{destination}</b> &nbsp;|&nbsp; ⏱️ <b>{duration_days} Days</b> &nbsp;|&nbsp; 🍂 <b>{season}</b> &nbsp;|&nbsp; 💳 <b>{budget_style}</b></h4>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Crafting your bespoke travel itinerary..."):
            try:
                chain = load_trip_chain()
                itinerary = chain.invoke({
                    "destination": destination.strip(),
                    "duration_days": str(duration_days),
                    "season": season,
                    "budget_style": budget_style,
                    "interests": ", ".join(interests) if interests else "General Exploration"
                })
                
                st.markdown(itinerary)
                st.divider()
                
                # Download Button
                st.download_button(
                    label="📥 Download Itinerary (.md)",
                    data=itinerary,
                    file_name=f"{destination.lower().replace(' ', '_')}_itinerary.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"Execution Error: {e}")
else:
    st.info("👈 Set your parameters in the sidebar and click **Architect My Itinerary** to generate your plan.")
