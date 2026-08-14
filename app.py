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

# --- Internal Web Search Function ---
def search_the_web(query):
    try:
        with DDGS() as ddgs:
            # Fetch top 4 search results
            results = [r for r in ddgs.text(query, max_results=4)]
            if not results:
                return "No web results found."
            
            # Format results cleanly for Gemini to read
            formatted_results = ""
            for i, r in enumerate(results):
                formatted_results += f"Source [{i+1}]: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n\n"
            return formatted_results
    except Exception as e:
        return f"Web search tool error: {e}"

# User Command Input Interface (Preserved Exactly)
user_query = st.text_area("⚡ Enter your command (Supports complex tasks, text creation, research, or direct image generation requests):", 
                          placeholder="e.g., 'Write an essay about quantum computing and generate a cinematic header image for it'", height=100)

if st.button("🚀 Execute OmniAgent Pipeline", type="primary"):
    if not user_query.strip():
        st.warning("Please supply a functional execution command.")
    else:
        # Step 1: Structural Parsing & Intent Evaluation via Gemini 3.7
        with st.spinner("🤖 Phase 1: Gemini 3.7 parsing intent and determining search requirements..."):
            try:
                intent_prompt = f"""
                Analyze this user request: "{user_query}"
                Break it down into technical components:
                1. search_query: If the user is asking about current events, news, or real-time info, create an optimized web search query string. If no web search is needed, leave it empty.
                2. text_prompt: Instructions for generating summaries, analysis, code, or answers.
                3. image_prompt: Explicit descriptions for generating associated graphics/images. Leave empty if no visuals are needed.
                
                Respond ONLY with a valid JSON block containing exactly these three keys: "search_query", "text_prompt", and "image_prompt".
                Do not include markdown wrappers, backticks, or any additional text. Just raw JSON.
                """
                
                intent_model = genai.GenerativeModel(
                    "gemini-3.7-flash",
                    generation_config={"response_mime_type": "application/json"}
                )
                intent_response = intent_model.generate_content(intent_prompt)
                
                parsed_actions = json.loads(intent_response.text)
                web_search_query = parsed_actions.get("search_query", "")
                text_directive = parsed_actions.get("text_prompt", user_query)
                image_directive = parsed_actions.get("image_prompt", "")
                
            except Exception as e:
                st.error(f"Intent Mapping Failure: {e}")
                web_search_query = ""
                text_directive = user_query
                image_directive = ""

        # Step 1.5: Execute Live Web Search if required
        web_context = ""
        if web_search_query:
            with st.spinner(f"🔍 Searching the web for: '{web_search_query}'..."):
                web_context = search_the_web(web_search_query)
                with st.expander("🌐 View Raw Web Search Results"):
                    st.text(web_context)

        # Step 2: Content Generation Layout Split
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Engine 1: Analytical Text & Logic")
            with st.spinner("Processing deep analysis with Gemini 3.7..."):
                try:
                    text_model = genai.GenerativeModel("gemini-3.7-flash")
                    
                    # Feed web context directly into Gemini if it exists
                    prompt_with_context = f"Directive: {text_directive}"
                    if web_context:
                        prompt_with_context = f"Use these real-time web search results to answer the user accurately:\n\n{web_context}\n\n{prompt_with_context}"
                    
                    text_system_instruction = "Generate complete, precise, functional text content answering the directive. Use the provided web search context if available to ensure up-to-date factual accuracy. Format beautifully using markdown."
                    
                    final_text_response = text_model.generate_content(
                        f"{text_system_instruction}\n\n{prompt_with_context}"
                    ).text
                    
                    st.markdown(final_text_response)
                    st.download_button("💾 Export Text Result", final_text_response, file_name="omni_output.md", mime="text/markdown")
                except Exception as e:
                    st.error(f"Logic Pipeline Interruption: {e}")

        with col2:
            st.subheader("🎨 Engine 2: Internal Visual Generation")
            if image_directive:
                with st.spinner("Engaging Gemini Imagen 3 rendering engines..."):
                    try:
                        prompt_model = genai.GenerativeModel("gemini-3.7-flash")
                        optimized_visual_prompt = prompt_model.generate_content(
                            f"Transform the input into a hyper-detailed, photorealistic, cinematic prompt for a high-end image generator. Input: {image_directive}"
                        ).text
                        
                        imagen_model = genai.GenerativeModel("imagen-3.0-generate-002")
                        image_result = imagen_model.generate_images(
                            prompt=optimized_visual_prompt,
                            number_of_images=1,
                            aspect_ratio="16:9"
                        )
                        
                        img_bytes = image_result.generated_images.image.image_bytes
                        display_img = Image.open(io.BytesIO(img_bytes))
                        st.image(display_img, caption="Rendered by Gemini Imagen 3 Engine", use_column_width=True)
                        
                        buf = io.BytesIO()
                        display_img.save(buf, format="PNG")
                        st.download_button("💾 Download Rendered Artifact", buf.getvalue(), file_name="omni_visual.png", mime="image/png")
                    except Exception as e:
                        st.error(f"Visual Engine Interruption: {e}")
            else:
                st.info("No visual generation requirements detected within this specific task chain.")
