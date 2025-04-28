import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class GradingSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.tfidf = TfidfVectorizer()
        self.bert_model = SentenceTransformer(model_name)

    def extract_answer(self, text):
        """Extract answer from question-answer format text"""
        match = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Extract answer if in Q&A format
        text = self.extract_answer(text)

        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text

    def exact_match_grading(self, student_answer, correct_answer, points):
        """Grade answers that require exact matches (like multiple choice)"""
        student_text = self.extract_answer(student_answer)
        correct_text = self.extract_answer(correct_answer)
        return points if student_text.strip().lower() == correct_text.strip().lower() else 0

    def numerical_grading(self, student_answer, correct_answer, points, tolerance=0.01):
        """Grade numerical answers with tolerance"""
        try:
            student_num = float(self.extract_answer(student_answer))
            correct_num = float(self.extract_answer(correct_answer))
            if abs(student_num - correct_num) <= tolerance:
                return points
            # Partial credit for close answers
            error_ratio = abs(student_num - correct_num) / correct_num
            if error_ratio <= 0.1:  # Within 10%
                return points * 0.5
            return 0
        except ValueError:
            return 0

    def semantic_similarity_grading(self, student_answer, correct_answer, points,
                                  threshold=0.7, grading_scheme=None):
        """Grade text answers using BERT-based semantic similarity"""
        # Extract and preprocess answers
        student_text = self.preprocess_text(student_answer)
        correct_text = self.preprocess_text(correct_answer)

        # Generate embeddings
        student_embedding = self.bert_model.encode(student_text, convert_to_tensor=True)
        correct_embedding = self.bert_model.encode(correct_text, convert_to_tensor=True)

        # Calculate similarity
        similarity = util.pytorch_cos_sim(student_embedding, correct_embedding).item()

        # Default grading scheme if none provided
        if grading_scheme is None:
            grading_scheme = [
                (0.9, 1.0),    # 90-100% similarity = full points
                (0.8, 0.8),    # 80-90% similarity = 80% points
                (0.7, 0.6),    # 70-80% similarity = 60% points
                (0.6, 0.4),    # 60-70% similarity = 40% points
                (0.5, 0.2)     # 50-60% similarity = 20% points
            ]

        # Apply grading scheme
        for min_sim, score_multiplier in grading_scheme:
            if similarity >= min_sim:
                return points * score_multiplier

        return 0

    def keyword_based_grading(self, student_answer, keywords, points, required_matches=None):
        """Grade based on presence of key words/phrases with semantic similarity"""
        if required_matches is None:
            required_matches = len(keywords)

        student_text = self.preprocess_text(student_answer)

        # Calculate semantic similarity for each keyword
        matches = 0
        for keyword in keywords:
            keyword_embedding = self.bert_model.encode(keyword, convert_to_tensor=True)
            student_embedding = self.bert_model.encode(student_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(keyword_embedding, student_embedding).item()

            if similarity > 0.7:  # Threshold for keyword match
                matches += 1

        if matches >= required_matches:
            return points
        # Partial credit based on proportion of matches
        return (points * matches) / required_matches

    def formula_grading(self, student_answer, correct_answer, points, test_cases):
        """Grade mathematical formulas using test cases"""
        try:
            # Extract formulas from text if needed
            student_formula = self.extract_answer(student_answer)
            correct_formula = self.extract_answer(correct_answer)

            # Remove spaces and convert to lowercase for formula comparison
            student_formula = ''.join(student_formula.split()).lower()
            correct_formula = ''.join(correct_formula.split()).lower()

            if student_formula == correct_formula:
                return points

            # Test with sample values if formulas don't match exactly
            correct_results = []
            student_results = []

            for test_case in test_cases:
                # Evaluate both formulas with test values
                correct_eval = eval(correct_formula, {"x": test_case})
                student_eval = eval(student_formula, {"x": test_case})

                correct_results.append(correct_eval)
                student_results.append(student_eval)

            # Compare results
            matches = sum(1 for i in range(len(test_cases))
                         if abs(student_results[i] - correct_results[i]) < 0.01)

            return points * (matches / len(test_cases))
        except:
            return 0

    def grade_question(self, question_type, student_answer, correct_answer, points, **kwargs):
        """Main grading function that routes to appropriate grading method"""
        grading_methods = {
            'exact': self.exact_match_grading,
            'numerical': self.numerical_grading,
            'text': self.semantic_similarity_grading,
            'keyword': self.keyword_based_grading,
            'formula': self.formula_grading
        }

        if question_type not in grading_methods:
            raise ValueError(f"Unsupported question type: {question_type}")

        return grading_methods[question_type](student_answer, correct_answer, points, **kwargs)

    def grade_assessment(self, assessment_data):
        """Grade entire assessment with multiple questions"""
        total_score = 0
        feedback = []

        for question in assessment_data:
            score = self.grade_question(
                question['type'],
                question['student_answer'],
                question['correct_answer'],
                question['points'],
                **question.get('grading_params', {})
            )

            # Calculate similarity percentage for text answers
            similarity_percentage = None
            if question['type'] == 'text':
                student_text = self.preprocess_text(question['student_answer'])
                correct_text = self.preprocess_text(question['correct_answer'])
                student_embedding = self.bert_model.encode(student_text, convert_to_tensor=True)
                correct_embedding = self.bert_model.encode(correct_text, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(student_embedding, correct_embedding).item()
                similarity_percentage = similarity * 100

            feedback.append({
                'question_number': question['number'],
                'score': score,
                'max_points': question['points'],
                'percentage': (score / question['points']) * 100,
                'similarity_percentage': similarity_percentage
            })

            total_score += score

        return {
            'total_score': total_score,
            'max_score': sum(q['points'] for q in assessment_data),
            'percentage': (total_score / sum(q['points'] for q in assessment_data)) * 100,
            'question_feedback': feedback
        }