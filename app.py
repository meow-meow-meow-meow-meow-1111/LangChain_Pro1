import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
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

# 3. Model & Chain Setup
@st.cache_resource
def load_trip_chain():
    hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not hf_token:
        st.error("⚠️ Hugging Face API Token not found in Secrets! Please check your Streamlit settings.")
        st.stop()

    # Hosted serverless model
    endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=1800,
        temperature=0.6,
        huggingfacehub_api_token=hf_token
    )
    
    llm = ChatHuggingFace(llm=endpoint)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an elite travel architect. You must generate a complete, day-by-day plan covering EVERY single day requested (from Day 1 up to Day {duration_days}). Never skip or summarize days into one."
        ),
        (
            "user",
            """Create a comprehensive {duration_days}-day travel itinerary for {destination} during {season}.
Budget Level: {budget_style}
Traveler Preferences / Interests: {interests}

CRITICAL RULES:
1. You MUST generate separate entries for ALL {duration_days} days. (e.g. If {duration_days} is 3, include Day 1, Day 2, and Day 3).
2. Keep each time block (Morning, Afternoon, Evening) to 1-2 punchy, actionable sentences so the complete itinerary fits within the response.

Strict Output Format:
## 📍 Destination Brief & Vibe
(A 2-sentence summary of the vibe and budgeting approach)

## 🗓️ Day-by-Day Masterplan
### Day 1: [Focus / Theme]
- **Morning:** [Landmark or neighborhood + local tip]
- **Afternoon:** [Activity matching budget]
- **Evening:** [Scenic view, dining, or night vibe]

### Day 2: [Focus / Theme]
- **Morning:** [Landmark or neighborhood + local tip]
- **Afternoon:** [Activity matching budget]
- **Evening:** [Scenic view, dining, or night vibe]

(Continue this exact pattern for all {duration_days} days)

## 🍽️ Local Culinary Highlights
- **Must-Try 1:** [Dish Name] — [Where to find it & why it matches {budget_style}]
- **Must-Try 2:** [Dish Name] — [Where to find it & why it matches {budget_style}]

## 💡 Local Insider Pro-Tip
[One practical transit, ticketing, or money-saving advice]
"""
        )
    ])
    
    return prompt | llm | StrOutputParser()

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
st.caption("Powered by LangChain LCEL & Hugging Face Serverless Chat")

if generate_btn:
    if not destination.strip():
        st.error("⚠️ Please specify a destination in the sidebar.")
    else:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Destination: <b>{destination}</b> &nbsp;|&nbsp; ⏱️ <b>{duration_days} Days</b> &nbsp;|&nbsp; 🍂 <b>{season}</b> &nbsp;|&nbsp; 💳 <b>{budget_style}</b></h4>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner(f"Crafting your complete {duration_days}-day itinerary..."):
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
