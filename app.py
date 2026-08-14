import streamlit as st
import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(
    page_title="Wanderlust AI - Next-Gen Trip Architect",
    page_icon="✈️",
    layout="wide"
)

# 2. Custom CSS
st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8533 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load LangChain Chain using Free Hugging Face Endpoint
@st.cache_resource
def load_trip_chain():
    # Uses Hugging Face's free cloud server instead of local RAM
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        max_new_tokens=1200,
        temperature=0.7,
        huggingfacehub_api_token=st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
    )
    
    template = PromptTemplate.from_template(
        """You are a luxury travel architect and local insider.
Design a curated, {duration_days}-day travel itinerary for {destination} during {season}.
Budget Level: {budget_style}
Traveler Preferences / Interests: {interests}

Strict Output Format:
## 📍 Destination Brief & Vibe
(A 2-sentence summary)

## 🗓️ Day-by-Day Masterplan
### Day 1: [Theme]
- **Morning:** [Activity]
- **Afternoon:** [Activity]
- **Evening:** [Activity]

## 🍽️ Local Culinary Highlights
- **Must-Try 1:** [Dish] - [Why]
- **Must-Try 2:** [Dish] - [Why]

## 💡 Local Insider Pro-Tip
[Practical tip]
"""
    )
    
    return template | llm | StrOutputParser()

# 4. Sidebar Controls
with st.sidebar:
    st.title("🗺️ Trip Parameters")
    destination = st.text_input("Destination", value="Kyoto, Japan")
    
    col1, col2 = st.columns(2)
    with col1:
        duration_days = st.slider("Days", 1, 5, 3)
    with col2:
        season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter", "Monsoon"])
        
    budget_style = st.select_slider(
        "Budget Style",
        options=["Backpacker Budget", "Smart Casual / Mid-Range", "Luxury / High-End"],
        value="Smart Casual / Mid-Range"
    )
    
    interests = st.multiselect(
        "Interests",
        ["Hidden Gems", "Street Food", "Art & Architecture", "Nature & Scenery", "Nightlife"],
        default=["Hidden Gems", "Street Food"]
    )
    
    generate_btn = st.button("✨ Architect My Itinerary")

# 5. Main UI Logic
st.title("✈️ Wanderlust AI Itinerary Architect")
st.caption("Powered by LangChain LCEL & Hugging Face Cloud Inference")

if generate_btn:
    if not destination.strip():
        st.error("Please enter a destination.")
    else:
        with st.spinner("Crafting itinerary..."):
            try:
                chain = load_trip_chain()
                itinerary = chain.invoke({
                    "destination": destination.strip(),
                    "duration_days": str(duration_days),
                    "season": season,
                    "budget_style": budget_style,
                    "interests": ", ".join(interests) if interests else "General"
                })
                
                st.markdown(itinerary)
                st.download_button("📥 Download (.md)", itinerary, "itinerary.md")
            except Exception as e:
                st.error(f"Error: {e}")
