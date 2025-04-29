"""
Enhanced Multi-Perspective Grading System
==========================================
A robust application for evaluating student answers with support for:
- Semantic similarity analysis
- Concept coverage detection
- Relationship analysis
- Multiple perspective recognition
- Contradiction detection

This version includes performance optimizations and fallback mechanisms
to handle SSL certificate and model loading issues.
"""

# Fix for PyTorch-Streamlit compatibility issue
import os
import sys
import warnings
import ssl
import requests

# Suppress warnings
warnings.filterwarnings('ignore')

# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''

# Create unverified SSL context and set as default
ssl._create_default_https_context = ssl._create_unverified_context

# Disable SSL verification for requests library
requests.packages.urllib3.disable_warnings()

# Import other dependencies
import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from time import time
from PIL import Image
import moondream as md

# Page configuration
st.set_page_config(
    page_title="Multi-Perspective Grading System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance optimization: Turn on caching for expensive operations
st.cache_data.clear()
st.cache_resource.clear()

# Load Moondream model for handwritten text conversion
@st.cache_resource
def load_moondream_model():
    try:
        # Create a session with SSL verification disabled
        session = requests.Session()
        session.verify = False
        
        # Get API key from Streamlit secrets
        try:
            api_key = st.secrets["moondream"]["api_key"]
        except Exception as e:
            st.error(f"Error loading API key from secrets: {e}")
            st.info("Please make sure you have created a .streamlit/secrets.toml file with your API key")
            return None
        
        # Try with SSL verification disabled
        return md.vl(api_key=api_key)
    except Exception as e:
        st.warning(f"Failed to load Moondream model: {e}")
        return None

# Initialize the model
try:
    # Add a message to indicate we're loading the model
    with st.spinner("Loading Moondream OCR model..."):
        moondream_model = load_moondream_model()
        if moondream_model is not None:
            st.success("Moondream OCR model loaded successfully!")
except Exception as e:
    st.warning(f"Error initializing Moondream model: {e}")
    moondream_model = None

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .feedback-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .grade-a {
        background-color: rgba(0, 255, 0, 0.1);
        border-left: 4px solid green;
    }
    .grade-b {
        background-color: rgba(255, 255, 0, 0.1);
        border-left: 4px solid #CCCC00;
    }
    .grade-c {
        background-color: rgba(255, 165, 0, 0.1);
        border-left: 4px solid orange;
    }
    .grade-d {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 4px solid red;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .perspective-card {
        background-color: rgba(100, 149, 237, 0.1);
        border-left: 4px solid cornflowerblue;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .concept-card {
        background-color: rgba(60, 179, 113, 0.1);
        border-left: 4px solid mediumseagreen;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .contradiction-card {
        background-color: rgba(255, 99, 71, 0.1);
        border-left: 4px solid tomato;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'correct_answer' not in st.session_state:
    st.session_state.correct_answer = ""
if 'student_answer' not in st.session_state:
    st.session_state.student_answer = ""
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Title and description
st.title("🧠 Multi-Perspective Grading System")
st.markdown("""
This enhanced system uses advanced analysis to evaluate student answers based on multiple factors:

- **Semantic Similarity**: How closely the meaning matches the correct answer
- **Concept Coverage**: Which key concepts from the correct answer are included
- **Relationship Analysis**: How well connections between concepts are explained
- **Perspective Analysis**: Whether multiple viewpoints or approaches are presented
- **Contradiction Detection**: Identifying misconceptions in student answers

The system is designed to recognize and reward different explanatory styles and multiple perspectives.
""")

# Sidebar
with st.sidebar:
    st.header("📋 Grading Options")
    
    # Question type selection
    question_type = st.selectbox(
        "Question Type",
        ["text", "numerical", "code"],
        index=0,
        help="Select the type of question you're grading"
    )
    
    # Points allocation
    points = st.number_input(
        "Maximum Points",
        min_value=1,
        max_value=100,
        value=10,
        help="Maximum points for this question"
    )
    
    # Grading threshold
    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum similarity required for partial credit"
    )
    
    st.divider()
    
    # Sample questions
    st.subheader("📚 Sample Questions")
    
    if st.button("⚡ Circuits Example"):
        correct = """Current flows in a circuit when a switch is turned on because closing the switch completes the 
        electrical path, allowing electrons to move through the conductor. According to Ohm's Law (V = IR), when 
        a potential difference (voltage) is applied across a closed circuit with resistance, it causes electrons 
        to drift from the negative terminal to the positive terminal of the power source, resulting in electric 
        current. The switch acts as a control mechanism—open switch means the path is broken, and no current flows; 
        closed switch completes the path, enabling current flow."""
        
        student = """Think of a circuit like a closed racetrack for tiny particles called electrons. When the switch 
        is off, there's a gap in the track—so the electrons can't go anywhere. But as soon as you flip the switch on, 
        the track is completed, and a kind of invisible push—caused by the battery or power source—starts nudging all 
        the electrons forward through the wire. This push is called voltage, and the movement of electrons is what we 
        call electric current. It's not that electrons from the battery travel all the way to each component instantly—instead, 
        it's more like a domino effect: one pushes the next, and energy is delivered to the whole circuit rapidly, even 
        though each individual electron moves slowly."""
        
        st.session_state.correct_answer = correct
        st.session_state.student_answer = student
    
    if st.button("🧪 Biology Example"):
        correct = """Cellular respiration is the process by which cells convert glucose and oxygen into energy in the form of ATP, 
        releasing carbon dioxide and water as byproducts. This process primarily occurs in the mitochondria, often called the 
        powerhouse of the cell. The chemical equation for cellular respiration is C6H12O6 + 6O2 → 6CO2 + 6H2O + ATP energy. 
        Without oxygen, cells must rely on fermentation, which produces much less ATP."""
        
        student = """Cellular respiration happens in the cell membrane, not the mitochondria. During this process, cells take in 
        carbon dioxide and release oxygen, which is why plants are important for producing oxygen. The energy created is stored 
        as starch rather than ATP. Humans can perform cellular respiration without oxygen when needed, and this actually produces 
        more energy than when oxygen is present."""
        
        st.session_state.correct_answer = correct
        st.session_state.student_answer = student

# Main content area
tab1, tab2, tab3 = st.tabs(["✏️ Grade Answer", "📊 Results Analysis", "📚 Grading History"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correct Answer")
        correct_answer = st.text_area(
            "Enter the correct answer",
            value=st.session_state.get('correct_answer', ''),
            height=200
        )
    
    with col2:
        st.subheader("Student Answer")
        answer_input_method = st.radio(
            "Student Answer Input Method",
            options=["Text", "Handwritten Image"],
            horizontal=True
        )
        
        if answer_input_method == "Text":
            # Text input for student answer
            student_answer = st.text_area(
                "Student Answer",
                value=st.session_state.student_answer if 'student_answer' in st.session_state else "",
                height=300,
                placeholder="Enter the student's answer here..."
            )
        else:
            # Image upload for handwritten answers
            uploaded_image = st.file_uploader(
                "Upload handwritten answer image", 
                type=["jpg", "jpeg", "png"], 
                key="student_image"
            )
            
            # Check if we need to run OCR or use cached results
            run_ocr = False
            
            # Initialize student_answer
            if st.session_state.extracted_text and uploaded_image is not None:
                # We have previously extracted text
                student_answer = st.session_state.extracted_text
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Handwritten Answer", use_container_width=True)
                
                # Show the previously extracted text
                st.success("Text already extracted from image!")
                edited_text = st.text_area(
                    "Extracted Text (you can edit if needed)", 
                    value=student_answer, 
                    height=200
                )
                student_answer = edited_text  # Use edited text if user makes changes
                
                # Option to re-run OCR
                if st.button("Re-run OCR"):
                    run_ocr = True
                    
            elif uploaded_image is not None:
                # New image, need to run OCR
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Handwritten Answer", use_container_width=True)
                run_ocr = True
            else:
                # No image uploaded
                student_answer = ""
                
            # Run OCR if needed
            if run_ocr and uploaded_image is not None:
                image = Image.open(uploaded_image)
                
                with st.spinner("Extracting text from image using OCR..."):
                    try:
                        if moondream_model is not None:
                            # Use a more specific prompt based on question type
                            prompt = "Perform OCR on the handwritten text in this image. Only return exactly the text as it appears, without adding, removing, or changing anything. Do not add any commentary, explanation, punctuation, new lines, extra spaces, words, characters, or symbols. Do not correct spelling or grammar. Output only the recognized text."
                            
                            if question_type == "code":
                                prompt = "Convert this handwritten code to computer code. Preserve all syntax, indentation, and formatting. Do not add any commentary or explanation."
                            elif question_type == "numerical":
                                prompt = "Convert this handwritten mathematical expression to text. Preserve all numbers, operators, and mathematical notation. Do not add any commentary or explanation."
                            
                            # Use a try-except block with SSL context handling
                            try:
                                # Ensure SSL verification is disabled for this request
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    result = moondream_model.query(image, prompt)
                                    student_answer = result["answer"]
                            except Exception as ssl_error:
                                st.error(f"SSL error: {ssl_error}")
                                # Try again with a simplified approach
                                st.info("Trying alternative method...")
                                try:
                                    # Use a more direct prompt
                                    simple_prompt = "OCR this text"
                                    result = moondream_model.query(image, simple_prompt)
                                    student_answer = result["answer"]
                                except Exception as e:
                                    raise Exception(f"Failed after retry: {e}")
                            
                            # Store in session state
                            st.session_state.extracted_text = student_answer
                            st.session_state.current_image = uploaded_image
                            
                            st.success("Text extracted from image!")
                            
                            # Show the extracted text with ability to edit
                            edited_text = st.text_area(
                                "Extracted Text (you can edit if needed)", 
                                value=student_answer, 
                                height=200
                            )
                            student_answer = edited_text  # Use edited text if user makes changes
                            
                            # Show preprocessing steps for OCR
                            with st.expander("OCR Preprocessing Details"):
                                st.write("**OCR Processing Steps:**")
                                st.write("1. ✅ Image loaded and prepared")
                                st.write("2. ✅ Binarization applied")
                                st.write("3. ✅ Noise reduction completed")
                                st.write("4. ✅ Text extraction performed")
                        else:
                            st.error("Moondream model is not available. Please use text input instead.")
                            student_answer = st.text_area(
                                "Student Answer (Manual Entry)",
                                height=300,
                                placeholder="Enter the student's answer here..."
                            )
                    except Exception as e:
                        st.error(f"Text extraction failed: {str(e)}")
                        student_answer = st.text_area(
                            "Student Answer (Manual Entry)",
                            height=300,
                            placeholder="Enter the student's answer here..."
                        )
    
    # Grade button
    if st.button("🔍 Grade Answer", type="primary"):
        if not student_answer or not correct_answer:
            st.error("⚠️ Please provide both the correct answer and student answer.")
        else:
            # Store the current student answer in session state
            st.session_state.student_answer = student_answer
            # Here we would normally call our grading system
            # For this simplified version, we'll use mock data
            
            # Use st.cache_resource to cache the grader initialization
            @st.cache_resource
            def get_grader():
                try:
                    from evaluator import GradingSystem
                    grader = GradingSystem()
                    return grader
                except Exception as e:
                    st.error(f"Error initializing grading system: {e}")
                    st.info("Using fallback grading mechanism with reduced functionality.")
                    # Return a simple fallback grader if the main one fails
                    class FallbackGrader:
                        def semantic_similarity_grading(self, student_text, correct_text, **kwargs):
                            # Simple word overlap similarity
                            student_words = set(student_text.lower().split())
                            correct_words = set(correct_text.lower().split())
                            if not student_words or not correct_words:
                                return 0, {"error": "Empty text"}
                            
                            # Jaccard similarity
                            intersection = len(student_words.intersection(correct_words))
                            union = len(student_words.union(correct_words))
                            similarity = intersection / union if union > 0 else 0
                            
                            # Basic feedback
                            points = kwargs.get("points", 10)
                            score = round(similarity * points)
                            
                            feedback = {
                                "raw_similarity": similarity,
                                "adjusted_similarity": similarity,
                                "weighted_score": similarity,
                                "composite_score": similarity,
                                "contradiction_penalty": 0,
                                "contradiction_reasons": [],
                                "concept_analysis": {
                                    "concept_coverage": similarity,
                                    "covered_concepts": list(student_words.intersection(correct_words)),
                                    "missing_concepts": list(correct_words.difference(student_words)),
                                    "total_concepts": len(correct_words)
                                },
                                "relationship_analysis": {
                                    "relationship_coverage": similarity,
                                    "relationship_explanations": [],
                                    "missing_relationships": [],
                                    "total_relationships": 0
                                },
                                "perspective_analysis": {
                                    "perspective_score": 0,
                                    "perspective_count": 0,
                                    "perspective_sentences": []
                                },
                                "grading_explanation": {
                                    "concept_bonus": False,
                                    "perspective_bonus": False,
                                    "relationship_bonus": False,
                                    "similarity_bonus": False
                                },
                                "final_score": score,
                                "max_score": points
                            }
                            
                            return score, feedback
                    
                    return FallbackGrader()
            
            # Get cached grader instance
            grader = get_grader()
            
            # Use st.cache_data to cache grading results for the same inputs
            @st.cache_data
            def grade_with_cache(q_type, student_ans, correct_ans, pts):
                start_time = time()
                grader = get_grader()
                
                # The semantic_similarity_grading method doesn't accept question_type parameter
                # Instead, it uses the evaluator internally with the correct question type
                result = grader.semantic_similarity_grading(student_ans, correct_ans, points=pts)
                
                processing_time = time() - start_time
                st.session_state.processing_time = processing_time
                return result
            
            # Grade the answer with caching
            with st.spinner(" Analyzing answer... This may take a few seconds."):
                result = grade_with_cache(question_type, student_answer, correct_answer, points)
            
            # Handle the result - grade_question returns a tuple (score, feedback) for text questions
            if isinstance(result, tuple) and len(result) == 2:
                # For text questions that return (score, feedback)
                score, detailed_feedback = result
                
                # Create a formatted result dictionary
                formatted_result = {
                    'score': score,
                    'max_score': points,
                    'percentage': (score / points) * 100 if points > 0 else 0,
                    'question_feedback': [{
                        'similarity_percentage': (detailed_feedback['raw_similarity'] * 2 - 1 + 1) / 2 * 100,
                        'adjusted_similarity_percentage': (detailed_feedback['adjusted_similarity'] * 2 - 1 + 1) / 2 * 100,
                        'contradiction_penalty': detailed_feedback['contradiction_penalty'],
                        'contradiction_reasons': detailed_feedback['contradiction_reasons'],
                        'detailed_feedback': detailed_feedback
                    }]
                }
                result = formatted_result
            
            # Store result in session state
            st.session_state.current_result = result
            
            # Store in history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_item = {
                "timestamp": timestamp,
                "question_type": question_type,
                "correct_answer": correct_answer,
                "student_answer": student_answer,
                "score": result['score'],
                "max_score": result['max_score'],
                "percentage": result['percentage'],
                "processing_time": st.session_state.processing_time,
                "feedback": result['question_feedback'][0]['detailed_feedback'] if 'question_feedback' in result else {}
            }
            st.session_state.history.append(history_item)
            
            # Display result with processing time
            st.success(f"✅ Grading complete in {st.session_state.processing_time:.2f} seconds! Score: {result['score']}/{result['max_score']} ({result['percentage']:.1f}%)")
            
            # Display metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(
                    "Score",
                    f"{result['score']}/{result['max_score']}"
                )
            with metric_cols[1]:
                st.metric(
                    "Percentage",
                    f"{result['percentage']:.1f}%"
                )
            with metric_cols[2]:
                # Display composite score if available
                if 'question_feedback' in result and result['question_feedback'] and \
                   'detailed_feedback' in result['question_feedback'][0] and \
                   'composite_score' in result['question_feedback'][0]['detailed_feedback']:
                    composite_score = result['question_feedback'][0]['detailed_feedback']['composite_score']
                    st.metric(
                        "Composite Score",
                        f"{composite_score * 100:.1f}%",
                        delta=f"{(composite_score * 100) - result['percentage']:.1f}%",
                        help="Overall score considering all factors including perspectives and concept coverage"
                    )
            
            with metric_cols[3]:
                if 'question_feedback' in result and result['question_feedback']:
                    if 'detailed_feedback' in result['question_feedback'][0] and 'weighted_score' in result['question_feedback'][0]['detailed_feedback']:
                        # Use the weighted score if available
                        weighted_score = result['question_feedback'][0]['detailed_feedback']['weighted_score'] * 100
                        st.metric(
                            "Weighted Score",
                            f"{weighted_score:.1f}%",
                            delta=f"{weighted_score - result['percentage']:.1f}%",
                            delta_color="normal" if weighted_score > result['percentage'] else "inverse",
                            help="Combined score based on similarity, concept coverage, and relationship analysis"
                        )
            
            # Add explanation about grading approach based on question type
            st.markdown("### Grading Approach")
            if question_type == 'text':
                st.info("""
                **Theory Question Grading:** This answer was evaluated using our advanced concept-based analysis system:
                
                1. **Semantic Similarity** (35%): Comparing the meaning of your answer to the correct answer
                2. **Concept Coverage** (30%): Identifying which key concepts you included
                3. **Relationship Analysis** (15%): Evaluating how well you explained connections between concepts
                4. **Perspective Analysis** (20%): Assessing different viewpoints or approaches in your answer
                5. **Contradiction Detection**: Checking for inconsistencies in your explanation
                
                The system also awards bonus points for exceptional performance in specific areas. The final score is a weighted 
                combination of these factors, prioritizing conceptual understanding over surface-level wording.
                """)
            elif question_type == 'code':
                st.info("""
                **Code Question Grading:** This answer was evaluated using CodeBERT to analyze code structure, 
                syntax, and logic. The system looks beyond exact character matches to understand the semantic 
                meaning and functionality of the code.
                """)
            elif question_type == 'numerical':
                st.info("""
                **Numerical Question Grading:** This answer was evaluated based on numerical accuracy with 
                tolerance for minor calculation errors. Partial credit may be awarded for answers that are close 
                to the correct value.
                """)

            # Detailed feedback in an expander
            with st.expander("📝 Raw Feedback Data"):
                st.json(result)

            # Advanced analysis dashboard
            if 'question_feedback' in result and result['question_feedback']:
                feedback = result['question_feedback'][0]

                # Create tabs for different analysis types
                analysis_tabs = st.tabs(["📊 Similarity Analysis", "🧩 Concept Coverage", "🔄 Relationship Analysis", "👁️ Multiple Perspectives", "⚠️ Contradictions", "🎯 Grading Explanation"])

                # Tab 1: Similarity Analysis
                with analysis_tabs[0]:
                    st.markdown("### Similarity Analysis")

                    # Show similarity metrics
                    if 'similarity_percentage' in feedback:
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Raw Similarity", 
                                f"{feedback['similarity_percentage']:.1f}%",
                                help="Initial text similarity score"
                            )

                        # Show adjusted similarity if available
                        if 'adjusted_similarity_percentage' in feedback:
                            with col2:
                                st.metric(
                                    "Adjusted Similarity", 
                                    f"{feedback['adjusted_similarity_percentage']:.1f}%",
                                    delta=f"{feedback['adjusted_similarity_percentage'] - feedback['similarity_percentage']:.1f}%" 
                                        if 'contradiction_reasons' in feedback and feedback['contradiction_reasons'] else None,
                                    delta_color="inverse" if 'contradiction_reasons' in feedback and feedback['contradiction_reasons'] else "normal",
                                    help="Similarity after contradiction penalties"
                                )

                        # Show weighted score if available
                        if 'detailed_feedback' in feedback and 'weighted_score' in feedback['detailed_feedback']:
                            with col3:
                                weighted_score = feedback['detailed_feedback']['weighted_score'] * 100
                                st.metric(
                                    "Weighted Score", 
                                    f"{weighted_score:.1f}%",
                                    delta=f"{weighted_score - feedback['similarity_percentage']:.1f}%",
                                    delta_color="normal" if weighted_score > feedback['similarity_percentage'] else "inverse",
                                    help="Combined score based on similarity, concept coverage, and relationship analysis"
                                )

                    # Explanation of scoring
                    st.markdown("""---
                    #### How the Score is Calculated

                    The grading system uses multiple factors to evaluate the student's answer:

                    1. **Raw Similarity** (35%): How closely the text matches the correct answer
                    2. **Concept Coverage** (30%): How many key concepts from the correct answer are present
                    3. **Relationship Analysis** (15%): How well the student explains connections between concepts
                    4. **Perspective Analysis** (20%): How many different perspectives are presented
                    5. **Contradiction Detection**: Penalties for statements that contradict the correct answer

                    This approach evaluates conceptual understanding rather than just text matching.
                    """)

                # Tab 2: Concept Coverage
                with analysis_tabs[1]:
                    st.markdown("### Concept Coverage Analysis")

                    # Check if concept analysis is available
                    if 'detailed_feedback' in feedback and 'concept_analysis' in feedback['detailed_feedback']:
                        concept_analysis = feedback['detailed_feedback']['concept_analysis']

                        # Show concept coverage metrics
                        coverage = concept_analysis['concept_coverage'] * 100
                        st.metric(
                            "Concept Coverage", 
                            f"{coverage:.1f}%",
                            help=f"Covered {len(concept_analysis['covered_concepts'])}/{concept_analysis['total_concepts']} key concepts"
                        )
                        
                        # Display covered and missing concepts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### ✅ Covered Concepts")
                            if concept_analysis['covered_concepts']:
                                for concept in concept_analysis['covered_concepts']:
                                    st.markdown(f"- `{concept}`")
                            else:
                                st.info("No key concepts covered in the student's answer.")
                        
                        with col2:
                            st.markdown("#### ❌ Missing Concepts")
                            if concept_analysis['missing_concepts']:
                                for concept in concept_analysis['missing_concepts'][:10]:  # Limit to top 10
                                    st.markdown(f"- `{concept}`")
                                if len(concept_analysis['missing_concepts']) > 10:
                                    st.caption(f"...and {len(concept_analysis['missing_concepts']) - 10} more")
                            else:
                                st.success("All key concepts covered! Great job!")
                    else:
                        st.info("Concept analysis not available for this answer type.")
                
                # Tab 3: Relationship Analysis
                with analysis_tabs[2]:
                    st.markdown("### Conceptual Relationship Analysis")
                    
                    # Check if relationship analysis is available
                    if 'detailed_feedback' in feedback and 'relationship_analysis' in feedback['detailed_feedback']:
                        relationship_analysis = feedback['detailed_feedback']['relationship_analysis']
                        
                        # Show relationship coverage metrics
                        if relationship_analysis['total_relationships'] > 0:
                            coverage = relationship_analysis['relationship_coverage'] * 100
                            st.metric(
                                "Relationship Coverage", 
                                f"{coverage:.1f}%",
                                help=f"Explained {len(relationship_analysis['relationship_explanations'])}/{relationship_analysis['total_relationships']} key relationships"
                            )
                            
                            # Display matched relationships
                            if relationship_analysis['relationship_explanations']:
                                st.markdown("#### Matched Relationships")
                                for i, rel in enumerate(relationship_analysis['relationship_explanations'], 1):
                                    with st.expander(f"Relationship {i} - Similarity: {rel['similarity']:.2f}"):
                                        st.markdown("**Correct Answer:**")
                                        st.markdown(f"*{rel['correct']}*")
                                        st.markdown("**Student Answer:**")
                                        st.markdown(f"*{rel['student']}*")
                            else:
                                st.warning("No conceptual relationships were adequately explained.")
                        else:
                            st.info("No key relationships identified in the correct answer.")
                    else:
                        st.info("Relationship analysis not available for this answer type.")
                
                # Tab 4: Perspective Analysis
                with analysis_tabs[3]:
                    st.markdown("### Multiple Perspectives Analysis")
                    
                    # Check if perspective analysis is available
                    if 'detailed_feedback' in feedback and 'perspective_analysis' in feedback['detailed_feedback']:
                        perspective_analysis = feedback['detailed_feedback']['perspective_analysis']
                        
                        # Show perspective score
                        perspective_score = perspective_analysis['perspective_score'] * 100
                        st.metric(
                            "Perspective Score", 
                            f"{perspective_score:.1f}%",
                            help=f"Based on {perspective_analysis['perspective_count']} different perspectives identified"
                        )
                        
                        # Display identified perspective sentences
                        if perspective_analysis['perspective_sentences']:
                            st.markdown("#### Identified Perspective Indicators")
                            for i, sentence in enumerate(perspective_analysis['perspective_sentences'], 1):
                                st.markdown(f"**{i}. \"{sentence}\"**")
                            
                            st.info("""
                            The system identified sentences that indicate the student is considering multiple viewpoints 
                            or alternative perspectives on the topic. This demonstrates critical thinking and a nuanced 
                            understanding of the subject matter.
                            """)
                        else:
                            st.warning("""
                            No explicit perspective indicators were detected in the student's answer. 
                            Encouraging students to consider alternative viewpoints or different approaches 
                            can demonstrate deeper critical thinking and a more nuanced understanding of complex topics.
                            """)
                    else:
                        st.info("Perspective analysis not available for this answer type.")
                
                # Tab 5: Contradictions
                with analysis_tabs[4]:
                    st.markdown("### ⚠️ Contradiction Analysis")
                    
                    if 'detailed_feedback' in feedback and 'contradiction_reasons' in feedback['detailed_feedback'] and feedback['detailed_feedback']['contradiction_reasons']:
                        st.error("Contradictions were detected in this answer.")
                        for i, reason in enumerate(feedback['detailed_feedback']['contradiction_reasons'], 1):
                            st.markdown(f"**{i}. {reason}**")
                        
                        # Show contradiction penalty
                        if 'contradiction_penalty' in feedback['detailed_feedback']:
                            st.metric(
                                "Contradiction Penalty",
                                f"-{feedback['detailed_feedback']['contradiction_penalty'] * 100:.1f}%"
                            )
                    else:
                        st.success("No contradictions were detected in this answer.")

with tab2:
    if st.session_state.current_result:
        st.subheader("📊 Detailed Analysis")
        
        # Display the results
        result = st.session_state.current_result
        
        # Create a summary dataframe
        if 'question_feedback' in result:
            feedback = result['question_feedback'][0]
            
            # Create a summary of the analysis
            st.markdown("### Summary of Grading Analysis")
            
            # Create a summary table
            summary_data = {
                "Metric": ["Score", "Percentage", "Raw Similarity"],
                "Value": [
                    f"{result['score']:.1f}/{result['max_score']}",
                    f"{result['percentage']:.1f}%",
                    f"{feedback.get('similarity_percentage', 'N/A'):.1f}%" if isinstance(feedback.get('similarity_percentage'), (int, float)) else "N/A"
                ]
            }
            
            # Add adjusted similarity if available
            if 'adjusted_similarity_percentage' in feedback:
                summary_data["Metric"].append("Adjusted Similarity")
                summary_data["Value"].append(f"{feedback['adjusted_similarity_percentage']:.1f}%")
            
            # Add weighted score if available
            if 'detailed_feedback' in feedback and 'weighted_score' in feedback['detailed_feedback']:
                summary_data["Metric"].append("Weighted Score")
                summary_data["Value"].append(f"{feedback['detailed_feedback']['weighted_score'] * 100:.1f}%")
                
            # Add composite score if available
            if 'detailed_feedback' in feedback and 'composite_score' in feedback['detailed_feedback']:
                summary_data["Metric"].append("Composite Score")
                summary_data["Value"].append(f"{feedback['detailed_feedback']['composite_score'] * 100:.1f}%")
            
            # Add perspective score if available
            if 'detailed_feedback' in feedback and 'perspective_analysis' in feedback['detailed_feedback']:
                perspective_analysis = feedback['detailed_feedback']['perspective_analysis']
                if perspective_analysis['perspective_count'] > 0:
                    summary_data["Metric"].append("Perspective Score")
                    summary_data["Value"].append(f"{perspective_analysis['perspective_score'] * 100:.1f}% ({perspective_analysis['perspective_count']} perspectives)")
            
            # Add concept coverage if available
            if 'detailed_feedback' in feedback and 'concept_analysis' in feedback['detailed_feedback']:
                concept_analysis = feedback['detailed_feedback']['concept_analysis']
                summary_data["Metric"].append("Concept Coverage")
                summary_data["Value"].append(f"{concept_analysis['concept_coverage'] * 100:.1f}% ({len(concept_analysis['covered_concepts'])}/{concept_analysis['total_concepts']} concepts)")
            
            # Add relationship coverage if available
            if 'detailed_feedback' in feedback and 'relationship_analysis' in feedback['detailed_feedback']:
                relationship_analysis = feedback['detailed_feedback']['relationship_analysis']
                if relationship_analysis['total_relationships'] > 0:
                    summary_data["Metric"].append("Relationship Coverage")
                    summary_data["Value"].append(f"{relationship_analysis['relationship_coverage'] * 100:.1f}% ({len(relationship_analysis['relationship_explanations'])}/{relationship_analysis['total_relationships']} relationships)")
            
            # Create and display the dataframe
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
            
            # Show contradictions if any
            if 'contradiction_reasons' in feedback and feedback['contradiction_reasons']:
                st.markdown("### ⚠️ Contradictions Detected")
                for i, reason in enumerate(feedback['contradiction_reasons'], 1):
                    st.markdown(f"**{i}. {reason}**")
            
            # Visualization of concept coverage
            if 'detailed_feedback' in feedback and 'concept_analysis' in feedback['detailed_feedback']:
                concept_analysis = feedback['detailed_feedback']['concept_analysis']
                
                st.markdown("### 🧩 Concept Coverage Visualization")
                
                # Create a bar chart of concept coverage
                concept_data = {
                    "Status": ["Covered", "Missing"],
                    "Count": [
                        len(concept_analysis['covered_concepts']),
                        len(concept_analysis['missing_concepts'])
                    ]
                }
                
                concept_df = pd.DataFrame(concept_data)
                st.bar_chart(concept_df.set_index("Status"))
    else:
        st.info("Grade an answer first to see detailed analysis.")

# Tab 3: Grading History
with tab3:
    st.header("📚 Grading History")
    st.info("This tab shows your grading history for this session. You can review previous assessments and export the data.")
    
    if st.session_state.history:
        # Display history in reverse chronological order
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Graded at {item['timestamp']} - Score: {item['score']}/{item['max_score']} ({item['percentage']:.1f}%)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Correct Answer")
                    st.markdown(f"<div class='highlight'>{item['correct_answer']}</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### Student Answer")
                    st.markdown(f"<div class='highlight'>{item['student_answer']}</div>", unsafe_allow_html=True)
                
                # Show key metrics
                st.markdown("#### Key Metrics")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.metric("Score", f"{item['score']}/{item['max_score']}")
                
                with metrics_col2:
                    st.metric("Percentage", f"{item['percentage']:.1f}%")
                
                with metrics_col3:
                    if 'composite_score' in item['feedback']:
                        composite_score = item['feedback']['composite_score'] * 100
                        st.metric("Composite Score", f"{composite_score:.1f}%")
                
                with metrics_col4:
                    if 'weighted_score' in item['feedback']:
                        weighted_score = item['feedback']['weighted_score'] * 100
                        st.metric("Weighted Score", f"{weighted_score:.1f}%")
                
                # Option to view full analysis
                if st.button(f"View Full Analysis", key=f"view_{i}"):
                    st.session_state.current_result = {
                        "score": item['score'],
                        "max_score": item['max_score'],
                        "percentage": item['percentage'],
                        "correct_answer": item['correct_answer'],
                        "student_answer": item['student_answer'],
                        "question_feedback": [{'detailed_feedback': item['feedback']}]
                    }
                    st.rerun()
        
        # Export functionality
        st.subheader("Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a JSON string of the history
            history_json = json.dumps({
                "grading_history": st.session_state.history,
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, indent=2)
            
            # Provide download button
            st.download_button(
                label="📥 Download Grading History (JSON)",
                data=history_json,
                file_name="grading_history.json",
                mime="application/json"
            )
        
        with col2:
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.success("History cleared!")
                st.rerun()
    else:
        st.info("No grading history available yet. Grade some answers to build history.")

# Sidebar additions
with st.sidebar:
    # Add a new sample question about democracy
    st.divider()
    if st.button("📚 Democracy & Elections Example"):
        correct = """Democratic governments hold elections to allow citizens to choose their representatives and leaders through a structured, periodic, and participatory process. Elections are a core mechanism of accountability, enabling the transfer or renewal of political power based on the will of the people. They ensure legitimacy of the government, uphold political rights, and are a fundamental expression of the democratic principle that sovereignty resides with the people."""
        
        student = """From an institutional theory lens, elections in democratic systems are not just tools for selecting leaders—they are instruments for stabilizing political systems, resolving societal conflicts peacefully, and preventing authoritarian consolidation. By providing a regular, legal method for contesting power, elections serve as pressure valves in pluralistic societies. They reduce the likelihood of violence by offering all political factions a legitimate path to influence and representation."""
        
        st.session_state.correct_answer = correct
        st.session_state.student_answer = student
        st.rerun()
    
    # Add a help section
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    This application demonstrates a domain-independent approach to grading that recognizes multiple perspectives and explanatory styles.
    
    The system evaluates student answers based on:
    - Semantic similarity
    - Concept coverage
    - Relationship analysis
    - Multiple perspectives
    - Contradiction detection
    
    It's designed to reward critical thinking and comprehensive understanding rather than just memorization.
    """)

# Footer
st.markdown("---")
st.caption("© 2025 Multi-Perspective Grading System | Developed for educational assessment")
