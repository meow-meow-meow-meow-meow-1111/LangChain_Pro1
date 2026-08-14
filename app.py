import streamlit as st
import torch
from transformers import pipeline
from transformers.utils import logging
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration & Custom Theme
st.set_page_config(
    page_title="Wanderlust AI - Next-Gen Trip Architect",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
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

# 2. Cached LangChain Model Loading (Prevents OOM Crashes)
@st.cache_resource(show_spinner="Booting AI Engine on GPU...")
def load_trip_chain():
    logging.set_verbosity_error()
    
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens=1500,
        return_full_text=False,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)
    llm = ChatHuggingFace(llm=hf_pipeline)
    
    template = PromptTemplate.from_template(
        """You are a luxury travel architect and local cultural insider.
Design a highly curated, {duration_days}-day travel itinerary for {destination} during {season}.
Budget Level: {budget_style}
Traveler Preferences / Interests: {interests}

Strict Output Format:
## 📍 Destination Brief & Vibe
A sharp, 2-sentence summary capturing the mood, best transit strategy, and budget reality.

## 🗓️ Day-by-Day Masterplan
Generate an entry for each day (Day 1 to Day {duration_days}):
### Day X: [Curated Title / Theme]
- **Morning:** [Specific landmark or neighborhood + actionable local tip]
- **Afternoon:** [Cultural spot, walk, or activity matching the budget]
- **Evening:** [Scenic view, twilight activity, or night vibe]

## 🍽️ Local Culinary Highlights
- **Must-Try 1:** [Dish name] — [Where to get it & why it fits {budget_style}]
- **Must-Try 2:** [Dish name] — [Where to get it & why it fits {budget_style}]

## 💡 Local Insider Pro-Tip
[One practical safety, ticketing, or cultural secret that saves money/time]
"""
    )
    
    return template | llm | StrOutputParser()

# Initialize the chain
chain = load_trip_chain()

# 3. Sidebar Controls
with st.sidebar:
    st.title("🗺️ Trip Parameters")
    st.caption("Customize your bespoke itinerary")
    
    destination = st.text_input("Destination", value="Kyoto, Japan", placeholder="e.g. Rome, Bali, Paris")
    
    col_dur, col_season = st.columns(2)
    with col_dur:
        duration_days = st.slider("Days", min_value=1, max_value=5, value=3)
    with col_season:
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

# 4. Main Display Area
st.title("✈️ Wanderlust AI Itinerary Architect")
st.markdown("Automated travel curation powered by **LangChain LCEL & Qwen 2.5 3B**.")

if generate_btn:
    if not destination.strip():
        st.error("⚠️ Please specify a destination in the sidebar.")
    else:
        # Quick summary banner
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Trip Target: <b>{destination}</b> &nbsp;|&nbsp; ⏱️ <b>{duration_days} Days</b> &nbsp;|&nbsp; 🍂 <b>{season}</b> &nbsp;|&nbsp; 💳 <b>{budget_style}</b></h4>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Generating itinerary..."):
            try:
                itinerary = chain.invoke({
                    "destination": destination.strip(),
                    "duration_days": str(duration_days),
                    "season": season,
                    "budget_style": budget_style,
                    "interests": ", ".join(interests) if interests else "General Exploration"
                })
                
                st.markdown(itinerary)
                st.divider()
                
                # Export options
                col_dl, _ = st.columns([1, 4])
                with col_dl:
                    st.download_button(
                        label="📥 Download Itinerary (.md)",
                        data=itinerary,
                        file_name=f"{destination.lower().replace(' ', '_')}_itinerary.md",
                        mime="text/markdown"
                    )
            except Exception as e:
                st.error(f"Execution Error: {e}")
else:
    st.info("👈 Set your destination and trip parameters in the sidebar, then click **Architect My Itinerary** to generate your plan.")
