import streamlit as st
import google.generativeai as genai
from PIL import Image
from duckduckgo_search import DDGS
import io
import json

# Setup Page Configuration
st.set_page_config(page_title="CeylonAI OmniAgent", page_icon="🦁", layout="wide")

# Custom Styling and Title Headers
st.markdown("# 🦁🧠 CeylonAI OmniAgent: Fully Optimized Master System")
st.markdown("### Created by Mr. Sanija Munasinghe")
st.markdown("Powered entirely by **Google Gemini 3.7 (with Live Web Search)** and **Imagen 3 (Visual Rendering Engine)**.")
st.markdown("---")

# System configuration validation
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing API configuration secrets. Please add GEMINI_API_KEY to Advanced Settings.")
    st.stop()

# Initialize Client Configurations
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- Production History Memory Matrix ---
SYSTEM_MEMORY = """
You are CeylonAI OmniAgent, a sovereign, fully optimized intelligence system.
Your core specifications are:
- Name: CeylonAI OmniAgent
- Developer: Created and engineered by Mr. Sanija Munasinghe.
- Origin Country: Based in Sri Lanka (Ceylon).
Always maintain respect for your heritage and speak proudly of your creator, Mr. Sanija Munasinghe, if asked about your origin or history.
"""

# --- Internal Web Search Function ---
def search_the_web(query):
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No web results found."
            formatted_results = ""
            for i, r in enumerate(results):
                formatted_results += f"Source [{i+1}]: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n\n"
            return formatted_results
    except Exception as e:
        return f"Web search tool error: {e}"

# User Command Input Interface
user_query = st.text_area("⚡ Enter your command (Supports complex tasks, text creation, research, or direct image generation requests):", 
                          placeholder="e.g., 'Write an essay about quantum computing and generate a cinematic header image for it'", height=100)

if st.button("🚀 Execute OmniAgent Pipeline", type="primary"):
    if not user_query.strip():
        st.warning("Please supply a functional execution command.")
    else:
        # Step 1: Smart Routing Intent Analysis
        with st.spinner("🤖 Phase 1: Analyzing request architecture..."):
            try:
                intent_prompt = f"""
                {SYSTEM_MEMORY}
                
                Analyze the user request: "{user_query}"
                Categorize it and determine exactly what tools are required.
                
                Respond ONLY with a valid JSON block containing these four keys:
                1. "mode": Set to "simple" (greetings/short talk), "complex" (coding/deep analysis), or "creative" (stories/art layout requests).
                2. "search_query": String query if online info is needed. Leave empty "" if not needed.
                3. "text_prompt": Refined prompt for generating text answers.
                4. "image_prompt": Detailed image description ONLY if explicitly requested or highly relevant to a creative task. If the user just says hello or asks a question without visual intent, leave this completely empty "".
                
                Output raw JSON only. Do not include markdown backticks or wrappers.
                """
                
                intent_model = genai.GenerativeModel(
                    "gemini-3.7-flash",
                    generation_config={"response_mime_type": "application/json"}
                )
                intent_response = intent_model.generate_content(intent_prompt)
                
                parsed_actions = json.loads(intent_response.text)
                execution_mode = parsed_actions.get("mode", "simple")
                web_search_query = parsed_actions.get("search_query", "")
                text_directive = parsed_actions.get("text_prompt", user_query)
                image_directive = parsed_actions.get("image_prompt", "")
                
            except Exception as e:
                # Fallback to safe defaults on failure to prevent crashes
                execution_mode = "simple"
                web_search_query = ""
                text_directive = user_query
                image_directive = ""

        # Display identified operational mode banner
        st.info(f"⚙️ **System Mode Auto-Selected:** {execution_mode.upper()}")

        # Step 1.5: Execute Live Web Search if required
        web_context = ""
        if web_search_query.strip():
            with st.spinner(f"🔍 Searching the web for real-time data..."):
                web_context = search_the_web(web_search_query)

        # Step 2: Content Generation Layout Split
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Engine 1: Analytical Text & Logic")
            with st.spinner("Generating response..."):
                try:
                    text_model = genai.GenerativeModel("gemini-3.7-flash")
                    
                    # Merge system memory identity directly into the text generation logic
                    prompt_with_context = f"{SYSTEM_MEMORY}\n\nDirective: {text_directive}"
                    if web_context:
                        prompt_with_context = f"Web Search Context:\n{web_context}\n\n{prompt_with_context}"
                    
                    final_text_response = text_model.generate_content(prompt_with_context).text
                    st.markdown(final_text_response)
                    st.download_button("💾 Export Text Result", final_text_response, file_name="ceylonai_output.md", mime="text/markdown")
                except Exception as e:
                    st.error(f"Logic Pipeline Interruption: {e}")

        with col2:
            st.subheader("🎨 Engine 2: Internal Visual Generation")
            # Only trigger visual engine if the router verified an actual image prompt instruction
            if image_directive.strip():
                with st.spinner("Engaging Gemini Imagen 3 rendering engines..."):
                    try:
                        imagen_model = genai.GenerativeModel("imagen-3.0-generate-002")
                        image_result = imagen_model.generate_images(
                            prompt=image_directive,
                            number_of_images=1,
                            aspect_ratio="16:9"
                        )
                        
                        img_bytes = image_result.generated_images.image.image_bytes
                        display_img = Image.open(io.BytesIO(img_bytes))
                        st.image(display_img, caption="Rendered by Gemini Imagen 3 Engine", use_column_width=True)
                        
                        buf = io.BytesIO()
                        display_img.save(buf, format="PNG")
                        st.download_button("💾 Download Rendered Artifact", buf.getvalue(), file_name="ceylonai_visual.png", mime="image/png")
                    except Exception as e:
                        st.error(f"Visual Engine Interruption: {e}")
            else:
                st.info("ℹ️ No visual requirements detected. Visual rendering engine remains idle to conserve resource limits.")
