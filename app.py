import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import json

# Setup Page Configuration
st.set_page_config(page_title="Gemini OmniAgent", page_icon="🧠", layout="wide")

st.markdown("# 🧠 Gemini OmniAgent: Fully Optimized Master System")
st.markdown("Powered entirely by **Google Gemini (Analytical Logic & Context Engines)** and **Imagen 3 (Visual Rendering Engine)**.")

# System configuration validation
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing API configuration secrets. Please add GEMINI_API_KEY to Advanced Settings.")
    st.stop()

# Initialize Client Configurations
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# User Command Input Interface
user_query = st.text_area("⚡ Enter your command (Supports complex tasks, text creation, research, or direct image generation requests):", 
                          placeholder="e.g., 'Write an essay about quantum computing and generate a cinematic header image for it'", height=100)

if st.button("🚀 Execute OmniAgent Pipeline", type="primary"):
    if not user_query.strip():
        st.warning("Please supply a functional execution command.")
    else:
        # Step 1: Structural Parsing & Intent Evaluation via Gemini
        with st.spinner("🤖 Phase 1: Gemini parsing intent and structuring tasks..."):
            try:
                intent_prompt = f"""
                Analyze this user request: "{user_query}"
                Break it down into two technical components:
                1. Text Tasks: Summaries, code, copy, analysis, structures.
                2. Visual Tasks: Explicit descriptions for generating associated graphics/images.
                
                Respond ONLY with a valid JSON block containing exactly these two keys: "text_prompt" and "image_prompt".
                If no image generation is explicitly or implicitly required, leave "image_prompt" empty.
                Do not include markdown wrappers, backticks, or any additional text. Just raw JSON.
                """
                
                # Using Gemini Flash to handle intent routing quickly
                intent_model = genai.GenerativeModel(
                    "gemini-2.5-flash",
                    generation_config={"response_mime_type": "application/json"}
                )
                intent_response = intent_model.generate_content(intent_prompt)
                
                parsed_actions = json.loads(intent_response.text)
                text_directive = parsed_actions.get("text_prompt", user_query)
                image_directive = parsed_actions.get("image_prompt", "")
                
            except Exception as e:
                st.error(f"Intent Mapping Failure: {e}")
                text_directive = user_query
                image_directive = ""

        # Step 2: Content Generation Layout Split
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📝 Engine 1: Analytical Text & Logic")
            with st.spinner("Processing deep analysis..."):
                try:
                    # Using Gemini Flash for robust writing, formatting, and analysis
                    text_model = genai.GenerativeModel("gemini-2.5-flash")
                    text_system_instruction = "Generate complete, precise, functional text content answering the directive. Format beautifully using markdown."
                    
                    final_text_response = text_model.generate_content(
                        f"{text_system_instruction}\n\nDirective: {text_directive}"
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
                        # Optimizing the image directive using Gemini text model for top visual results
                        prompt_model = genai.GenerativeModel("gemini-2.5-flash")
                        optimized_visual_prompt = prompt_model.generate_content(
                            f"Transform the input into a hyper-detailed, photorealistic, cinematic prompt for a high-end image generator. Input: {image_directive}"
                        ).text
                        
                        # Call free Imagen 3 model
                        imagen_model = genai.GenerativeModel("imagen-3.0-generate-002")
                        image_result = imagen_model.generate_images(
                            prompt=optimized_visual_prompt,
                            number_of_images=1,
                            aspect_ratio="16:9"
                        )
                        
                        img_bytes = image_result.generated_images[0].image.image_bytes
                        display_img = Image.open(io.BytesIO(img_bytes))
                        st.image(display_img, caption="Rendered by Gemini Imagen 3 Engine", use_column_width=True)
                        
                        buf = io.BytesIO()
                        display_img.save(buf, format="PNG")
                        st.download_button("💾 Download Rendered Artifact", buf.getvalue(), file_name="omni_visual.png", mime="image/png")
                    except Exception as e:
                        st.error(f"Visual Engine Interruption: {e}")
            else:
                st.info("No visual generation requirements detected within this specific task chain.")
