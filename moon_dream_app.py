import streamlit as st

st.set_page_config(page_title="Handwritten Text Recognition", page_icon="📝")

from PIL import Image
import moondream as md

@st.cache_resource
def load_model():
    return md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI1YTk4NjkzNi02ZWRmLTQwYWUtODQwYS00NzU3YzBmODFmYjQiLCJvcmdfaWQiOiJLTWoxVHhTMGF1ZVN4MkVnekdZMjZnZVNYTFJWUnhyZCIsImlhdCI6MTc0NTgzMDk1NiwidmVyIjoxfQ.sT7l7x4WGAfNfy7IGyar-JaAG8JKCHYaEEyhgFCQvf4")

model = load_model()

st.title("Handwritten Text Recognition")
st.write("Upload an image with handwritten text to display it below.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown("## Recognition Result")
    with st.spinner("Processing..."):
        try:
            result = model.query(
                image,
                "Perform OCR on the handwritten text in this image. Only return exactly the text as it appears, without adding, removing, or changing anything. Do not add any commentary, explanation, punctuation, new lines, extra spaces, words, characters, or symbols. Do not correct spelling or grammar. Output only the recognized text."
            )
            answer = result["answer"]
            st.subheader("Converted Text")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
