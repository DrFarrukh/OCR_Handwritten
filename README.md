# Handwritten Text Recognition with Moondream and Streamlit

This project provides a simple web app for recognizing handwritten text from images using the Moondream API and Streamlit. It is designed for educators, students, and anyone who needs to quickly convert handwritten notes or answers into digital text.

## Features
- Upload an image containing handwritten text (JPG, JPEG, PNG)
- View the uploaded image
- Extract and display the recognized text using Moondream's OCR capabilities
- Clean, user-friendly Streamlit interface

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Add your Moondream API key
Edit `moon_dream_app.py` and replace the placeholder in `load_model()` with your valid Moondream API key:
```python
@st.cache_resource
def load_model():
    return md.vl(api_key="YOUR_API_KEY_HERE")
```

### 4. Run the app
```bash
streamlit run moon_dream_app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501).

## Example Usage
1. Open the app in your browser.
2. Upload an image with handwritten text.
3. The recognized text will be displayed below the image.

## Preprocessing (Optional)
You can enhance recognition by preprocessing images (e.g., grayscale, denoising, removing colored marks). See comments in `moon_dream_app.py` for how to add or adjust preprocessing steps.

## License
This project is for educational and research purposes. Please check the Moondream API terms for commercial use.

## Acknowledgements
- [Streamlit](https://streamlit.io/)
- [Moondream](https://moondream.dev/)
- [Pillow](https://python-pillow.org/)
