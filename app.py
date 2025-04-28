import streamlit as st
from evaluator import GradingSystem
import json
import pandas as pd
from datetime import datetime
from PIL import Image
import moondream as md

# Page configuration
st.set_page_config(
    page_title="Advanced Grading System",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_moondream_model():
    return md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI1YTk4NjkzNi02ZWRmLTQwYWUtODQwYS00NzU3YzBmODFmYjQiLCJvcmdfaWQiOiJLTWoxVHhTMGF1ZVN4MkVnekdZMjZnZVNYTFJWUnhyZCIsImlhdCI6MTc0NTgzMDk1NiwidmVyIjoxfQ.sT7l7x4WGAfNfy7IGyar-JaAG8JKCHYaEEyhgFCQvf4")

moondream_model = load_moondream_model()

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .st-emotion-cache-1y4p8pa {
        padding: 2rem 1rem;
    }
    .results-table {
        margin-top: 1rem;
    }
    .st-emotion-cache-1wmy9hl {
        max-width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'grading_history' not in st.session_state:
    st.session_state.grading_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'question_bank' not in st.session_state:
    st.session_state.question_bank = {}

# Initialize the grading system
@st.cache_resource
def get_grader():
    return GradingSystem()

grader = get_grader()

# Main title with emoji and subtitle
st.title("🎓 Advanced Grading System")
st.markdown("""
<p style='font-size: 1.2em; color: #666;'>
    An intelligent system for grading answers based on semantic similarity
</p>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("📋 Grading Configuration")
    
    # Question type selection with better descriptions
    question_type = st.selectbox(
        "Select Question Type",
        ["text", "exact", "numerical", "keyword", "formula"],
        help="Choose how you want to evaluate the answer",
        format_func=lambda x: {
            'text': '📝 Text (Semantic Similarity)',
            'exact': '✓ Exact Match',
            'numerical': '🔢 Numerical',
            'keyword': '🔑 Keyword Based',
            'formula': '📐 Formula/Expression'
        }[x]
    )
    
    points = st.number_input(
        "Maximum Points",
        min_value=1,
        value=10,
        help="Maximum points for this question"
    )

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["📝 Grade Answer", "📊 Grading History", "💾 Question Bank"])

with tab1:
    # Input area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✨ Correct Answer")
        correct_answer = st.text_area(
            "Reference answer",
            height=200,
            key="correct_answer",
            help="Enter the correct/reference answer here"
        )

    with col2:
        st.subheader("📝 Student Answer")
        uploaded_image = st.file_uploader("Upload handwritten answer image", type=["jpg", "jpeg", "png"], key="student_image")
        extracted_text = ""
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            with st.spinner("Extracting text from image..."):
                try:
                    result = moondream_model.query(image, "Convert this handwritten text to computer typography")
                    extracted_text = result["answer"]
                    st.success("Text extracted from image!")
                except Exception as e:
                    st.error(f"Text extraction failed: {str(e)}")
        student_answer = st.text_area(
            "Student's response",
            value=extracted_text,
            height=200,
            key="student_answer",
            help="Enter the student's answer here or upload an image above to auto-fill"
        )

    # Configuration based on question type
    config_col1, config_col2 = st.columns(2)
    grading_params = {}

    with config_col1:
        if question_type == "numerical":
            tolerance = st.number_input(
                "Tolerance",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                help="Acceptable difference from the correct answer"
            )
            grading_params["tolerance"] = tolerance

        elif question_type == "text":
            threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="Minimum similarity score required"
            )
            grading_params["threshold"] = threshold

    with config_col2:
        if question_type == "keyword":
            keywords = st.text_area(
                "Keywords (one per line)",
                help="Enter important keywords to look for"
            )
            if keywords:
                keywords = [k.strip() for k in keywords.split("\n") if k.strip()]
                required_matches = st.number_input(
                    "Required matches",
                    min_value=1,
                    max_value=len(keywords) if keywords else 1,
                    value=1
                )
                grading_params["keywords"] = keywords
                grading_params["required_matches"] = required_matches

        elif question_type == "formula":
            test_cases = st.text_area(
                "Test cases (one per line)",
                help="Enter values to test the formula"
            )
            if test_cases:
                test_cases = [float(x.strip()) for x in test_cases.split("\n") if x.strip()]
                grading_params["test_cases"] = test_cases

    # Save to question bank option
    save_to_bank = st.checkbox("Save to Question Bank", help="Save this question for future use")
    if save_to_bank:
        question_name = st.text_input("Question Name", help="Enter a name to identify this question")

    # Grade button with loading state
    if st.button("🎯 Grade Answer", type="primary"):
        if not correct_answer or not student_answer:
            st.error("⚠️ Please provide both correct and student answers.")
        else:
            with st.spinner("Grading in progress..."):
                try:
                    # Create assessment data
                    assessment_data = [{
                        'number': 1,
                        'type': question_type,
                        'student_answer': student_answer,
                        'correct_answer': correct_answer,
                        'points': points,
                        'grading_params': grading_params
                    }]

                    # Grade the assessment
                    result = grader.grade_assessment(assessment_data)
                    
                    # Save result to session state
                    st.session_state.current_result = result
                    
                    # Add to grading history
                    history_entry = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'question_type': question_type,
                        'score': result['total_score'],
                        'max_score': result['max_score'],
                        'percentage': result['percentage'],
                        'similarity': result['question_feedback'][0].get('similarity_percentage', None)
                    }
                    st.session_state.grading_history.append(history_entry)
                    
                    # Save to question bank if requested
                    if save_to_bank and question_name:
                        st.session_state.question_bank[question_name] = {
                            'type': question_type,
                            'correct_answer': correct_answer,
                            'points': points,
                            'grading_params': grading_params
                        }
                        st.success(f"✅ Question saved to bank as '{question_name}'!")

                    # Display results
                    st.markdown("### 📊 Grading Results")
                    
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.metric(
                            "Score",
                            f"{result['total_score']:.2f}/{result['max_score']}",
                            delta=f"{result['percentage']:.1f}%"
                        )
                    # with metric_cols[1]:
                    #     st.metric(
                    #         "Grade",
                    #         {
                    #             90: "A",
                    #             80: "B",
                    #             70: "C",
                    #             60: "D",
                    #             0: "F"
                    #         }[min([k for k in [90, 80, 70, 60, 0] if result['percentage'] >= k])]
                    #     )
                    with metric_cols[2]:
                        if result['question_feedback'][0]['similarity_percentage'] is not None:
                            st.metric(
                                "Similarity",
                                f"{result['question_feedback'][0]['similarity_percentage']:.1f}%"
                            )
                    
                    # Detailed feedback in an expander
                    with st.expander("📝 Detailed Feedback"):
                        st.json(result)

                except Exception as e:
                    st.error(f"❌ An error occurred while grading: {str(e)}")

with tab2:
    if st.session_state.grading_history:
        st.markdown("### 📈 Grading History")
        df = pd.DataFrame(st.session_state.grading_history)
        st.dataframe(
            df,
            column_config={
                'timestamp': "Time",
                'question_type': "Type",
                'score': st.column_config.NumberColumn("Score", format="%.2f"),
                'max_score': "Max Score",
                'percentage': st.column_config.NumberColumn("Percentage", format="%.1f%%"),
                'similarity': st.column_config.NumberColumn("Similarity", format="%.1f%%")
            },
            hide_index=True
        )
        
        # Download button for history
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download History",
            data=csv,
            file_name="grading_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No grading history available yet. Grade some answers to see them here!")

with tab3:
    st.markdown("### 📚 Saved Questions")
    if st.session_state.question_bank:
        for name, question in st.session_state.question_bank.items():
            with st.expander(f"📌 {name}"):
                st.write(f"**Type:** {question['type']}")
                st.write(f"**Points:** {question['points']}")
                st.write("**Correct Answer:**")
                st.code(question['correct_answer'])
                if question['grading_params']:
                    st.write("**Grading Parameters:**")
                    st.json(question['grading_params'])
                
                # Load question button
                if st.button(f"Load '{name}'", key=f"load_{name}"):
                    st.session_state['correct_answer'] = question['correct_answer']
                    st.experimental_rerun()
    else:
        st.info("No questions saved yet. Save some questions while grading to see them here!")

# Help section
with st.sidebar.expander("ℹ️ Help & Instructions"):
    st.markdown("""
    ### 📚 How to use this grading system:
    
    1. **Select Question Type** 📋
       - Text: For free-form answers
       - Exact: For precise matching
       - Numerical: For mathematical answers
       - Keyword: For specific term checking
       - Formula: For mathematical expressions
    
    2. **Configure Settings** ⚙️
       - Set maximum points
       - Adjust grading parameters
       - Save questions for reuse
    
    3. **Grade Answers** ✅
       - Enter reference answer
       - Input student response
       - Click Grade to evaluate
    
    4. **Review Results** 📊
       - See scores and metrics
       - Check detailed feedback
       - Track grading history
    """)
