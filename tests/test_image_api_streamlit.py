"""
Streamlit version of the image API test - can access Streamlit secrets.
Run with: streamlit run test_image_api_streamlit.py
"""

import streamlit as st
from openai import OpenAI

st.title("🔍 OpenAI Image Generation API Test")
st.markdown("This tool tests if your API key has image generation permissions.")

# Get API key from Streamlit secrets
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.success(f"✅ API Key found (first 10 chars: {api_key[:10]}...)")
except KeyError:
    st.error("❌ OPENAI_API_KEY not found in Streamlit secrets!")
    st.info("Add it to your `.streamlit/secrets.toml` file")
    st.stop()

client = OpenAI(api_key=api_key)

if st.button("🧪 Run Image Generation Test", type="primary"):
    st.divider()
    
    # Test 1: Try gpt-image-1
    with st.expander("1️⃣ Testing gpt-image-1 model", expanded=True):
        try:
            with st.spinner("Testing gpt-image-1..."):
                response = client.images.generate(
                    model="gpt-image-1",
                    prompt="A simple test image with text 'API Test'",
                    size="1024x1024",
                    quality="high",
                    n=1,
                )
                
                if hasattr(response, 'data') and len(response.data) > 0:
                    image_url = response.data[0].url
                    st.success("✅ SUCCESS! gpt-image-1 is available")
                    st.image(image_url, caption="Generated with gpt-image-1", use_container_width=True)
                    st.info(f"Image URL: {image_url}")
                    st.balloons()
                else:
                    st.warning("⚠️ Response received but no image data found")
                    
        except Exception as e:
            error_str = str(e)
            st.error(f"❌ FAILED: {error_str}")
            
            # Check specific error types
            if "model" in error_str.lower() and "not found" in error_str.lower():
                st.info("💡 gpt-image-1 model not available in your account")
            elif "permission" in error_str.lower() or "access" in error_str.lower():
                st.warning("💡 Possible permission issue - check API key permissions")
            elif "billing" in error_str.lower() or "credit" in error_str.lower():
                st.warning("💡 Possible billing/credit issue - check your account balance")
            
            # Try DALL-E 3 as fallback
            st.divider()
            with st.expander("2️⃣ Testing DALL-E 3 model (fallback)", expanded=True):
                try:
                    with st.spinner("Testing DALL-E 3..."):
                        response = client.images.generate(
                            model="dall-e-3",
                            prompt="A simple test image with text 'API Test'",
                            size="1024x1024",
                            quality="standard",
                            n=1,
                        )
                        
                        if hasattr(response, 'data') and len(response.data) > 0:
                            image_url = response.data[0].url
                            st.success("✅ SUCCESS! DALL-E 3 is available")
                            st.image(image_url, caption="Generated with DALL-E 3", use_container_width=True)
                            st.info(f"Image URL: {image_url}")
                            st.balloons()
                        else:
                            st.warning("⚠️ Response received but no image data found")
                            
                except Exception as e2:
                    error_str2 = str(e2)
                    st.error(f"❌ FAILED: {error_str2}")
                    
                    st.divider()
                    st.error("❌ DIAGNOSIS: Image generation not available")
                    
                    st.markdown("""
                    **Possible reasons:**
                    1. API key doesn't have image generation permissions
                    2. Insufficient API credits/balance
                    3. Image generation models not enabled for your account
                    4. API key is invalid or expired
                    
                    **How to fix:**
                    1. Check OpenAI Platform Dashboard: https://platform.openai.com/
                    2. Verify API key permissions in Settings > API Keys
                    3. Check billing/credits: https://platform.openai.com/account/billing
                    4. Contact OpenAI support if needed
                    """)
                    
                    with st.expander("🔍 Full Error Details"):
                        st.code(f"gpt-image-1 error:\n{error_str}\n\nDALL-E 3 error:\n{error_str2}")

st.divider()
st.markdown("""
### 📚 Additional Resources
- [OpenAI Platform Dashboard](https://platform.openai.com/)
- [API Usage](https://platform.openai.com/usage)
- [Billing](https://platform.openai.com/account/billing)
- [API Documentation](https://platform.openai.com/docs/guides/images)
""")

