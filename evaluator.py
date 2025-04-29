import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Import our advanced evaluation module
from evaluation import SemanticEvaluator

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class GradingSystem:
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        # Initialize our advanced semantic evaluator with support for theory, code, and numerical
        self.evaluator = SemanticEvaluator()
        
        # Common words to exclude from concept extraction
        self.common_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                      'while', 'of', 'to', 'in', 'for', 'on', 'by', 'with', 'about', 'against',
                      'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                      'from', 'up', 'down', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'i', 'you', 'he',
                      'she', 'it', 'we', 'they', 'their', 'this', 'that', 'these', 'those'}

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

    def extract_key_concepts(self, text):
        """Extract key concepts from text by removing stopwords and punctuation"""
        # Convert to lowercase
        text = text.lower()
        # Simple tokenization by splitting on whitespace and removing punctuation
        for p in string.punctuation:
            text = text.replace(p, ' ')
        tokens = text.split()
        # Remove common English stopwords (simplified list)
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                      'while', 'of', 'to', 'in', 'for', 'on', 'by', 'with', 'about', 'against',
                      'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                      'from', 'up', 'down', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'i', 'you', 'he',
                      'she', 'it', 'we', 'they', 'their', 'this', 'that', 'these', 'those'}
        tokens = [word for word in tokens if word not in stop_words]
        return set(tokens)
    
    def detect_contradictions(self, student_text, correct_text):
        """Detect contradictory concepts between student and correct answers using semantic analysis"""
        # Extract sentences for analysis
        student_sentences = [s.strip() for s in re.split(r'[.!?]', student_text) if s.strip()]
        correct_sentences = [s.strip() for s in re.split(r'[.!?]', correct_text) if s.strip()]
        
        contradiction_penalty = 0
        detected_contradictions = []
        
        # APPROACH 1: Semantic Entailment Analysis
        # Check if key statements in the correct answer are contradicted in the student answer
        # This is domain-independent and works for any subject matter
        
        # First, identify key sentences in the correct answer (typically first sentence and purpose statements)
        key_correct_sentences = []
        
        # Add the first sentence as it often contains the main point
        if correct_sentences:
            key_correct_sentences.append(correct_sentences[0])
        
        # Add sentences containing causal relationships or definitions
        causal_terms = ["because", "since", "as", "therefore", "thus", "hence"]
        definition_terms = ["is", "are", "means", "refers to", "defined as"]
        
        for sentence in correct_sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in causal_terms):
                if sentence not in key_correct_sentences:
                    key_correct_sentences.append(sentence)
            if any(term in sentence_lower for term in definition_terms):
                if sentence not in key_correct_sentences:
                    key_correct_sentences.append(sentence)
        
        # Check for analogy indicators in student answer
        analogy_terms = ["like", "as if", "imagine", "think of", "similar to", "resembles", "compared to"]
        uses_analogy = any(term in student_text.lower() for term in analogy_terms)
        
        # Calculate overall similarity between answers
        overall_similarity = self.evaluator.compute_similarity(student_text, correct_text, question_type="theory")
        
        # For each key sentence in correct answer, check if student answer contains contradictory statements
        for correct_sentence in key_correct_sentences:
            # Check each student sentence against this correct sentence
            for student_sentence in student_sentences:
                # Calculate semantic similarity between sentences
                similarity = self.evaluator.compute_similarity(correct_sentence, student_sentence, question_type="theory")
                
                # If sentences are discussing the same topic (medium similarity) but not in agreement (not high similarity)
                # This indicates they're talking about the same concept but with contradictory explanations
                
                # Adjust similarity threshold for analogy detection
                # If student is using analogies, we expect lower sentence-level similarity even when correct
                similarity_lower_bound = 0.3 if uses_analogy else 0.4
                similarity_upper_bound = 0.6 if uses_analogy else 0.7
                
                if similarity_lower_bound <= similarity <= similarity_upper_bound:
                    # Check for semantic contradiction using negation detection and causal analysis
                    correct_has_negation = any(neg in correct_sentence.lower() for neg in ["not", "n't", "never"])
                    student_has_negation = any(neg in student_sentence.lower() for neg in ["not", "n't", "never"])
                    
                    # If one has negation and the other doesn't, likely a contradiction
                    if correct_has_negation != student_has_negation:
                        # Skip if this is part of an analogy and overall similarity is high
                        if uses_analogy and overall_similarity > 0.75:
                            continue
                            
                        explanation = "Contradiction detected: Opposing statements about the same concept."
                        explanation += f"\nCorrect: '{correct_sentence[:100]}...'"
                        explanation += f"\nStudent: '{student_sentence[:100]}...'"
                        contradiction_penalty += 0.3
                        detected_contradictions.append(explanation)
                        continue
                    
                    # Check for different causal explanations (different 'because' statements)
                    if any(term in correct_sentence.lower() for term in causal_terms) and \
                       any(term in student_sentence.lower() for term in causal_terms):
                        # Skip if this is part of an analogy and overall similarity is high
                        if uses_analogy and overall_similarity > 0.75:
                            continue
                            
                        # If discussing same topic but giving different causal explanations
                        explanation = "Contradiction detected: Different causal explanations for the same phenomenon."
                        explanation += f"\nCorrect: '{correct_sentence[:100]}...'"
                        explanation += f"\nStudent: '{student_sentence[:100]}...'"
                        contradiction_penalty += 0.3
                        detected_contradictions.append(explanation)
        
        # APPROACH 2: Function/Purpose Analysis
        # Compare explanations of purpose or function between correct and student answers
        purpose_sentences_correct = [s for s in correct_sentences if any(term in s.lower() for term in 
                                    ["purpose", "function", "role", "used for", "in order to", "so that"])]  
        purpose_sentences_student = [s for s in student_sentences if any(term in s.lower() for term in 
                                    ["purpose", "function", "role", "used for", "in order to", "so that"])]  
        
        if purpose_sentences_correct and purpose_sentences_student:
            # Compare purpose statements
            for correct_purpose in purpose_sentences_correct:
                for student_purpose in purpose_sentences_student:
                    purpose_similarity = self.evaluator.compute_similarity(correct_purpose, student_purpose, question_type="theory")
                    if purpose_similarity < 0.5:  # Low similarity in purpose statements
                        explanation = "Contradiction detected: The student's explanation of purpose/function differs significantly from the correct answer."
                        explanation += f"\nCorrect: '{correct_purpose[:100]}...'"
                        explanation += f"\nStudent: '{student_purpose[:100]}...'"
                        contradiction_penalty += 0.3
                        detected_contradictions.append(explanation)
        
        # APPROACH 3: Process/Mechanism Analysis
        # Check if the student describes a fundamentally different process or mechanism
        process_terms = ["process", "mechanism", "how", "works by", "through", "via", "by means of"]
        process_sentences_correct = [s for s in correct_sentences if any(term in s.lower() for term in process_terms)]
        process_sentences_student = [s for s in student_sentences if any(term in s.lower() for term in process_terms)]
        
        if process_sentences_correct and process_sentences_student:
            # Compare process descriptions
            for correct_process in process_sentences_correct:
                for student_process in process_sentences_student:
                    process_similarity = self.evaluator.compute_similarity(correct_process, student_process, question_type="theory")
                    if process_similarity < 0.5:  # Low similarity in process descriptions
                        explanation = "Contradiction detected: The student describes a fundamentally different process or mechanism."
                        explanation += f"\nCorrect: '{correct_process[:100]}...'"
                        explanation += f"\nStudent: '{student_process[:100]}...'"
                        contradiction_penalty += 0.3
                        detected_contradictions.append(explanation)
        
        # Remove duplicates while preserving order
        unique_contradictions = []
        for item in detected_contradictions:
            if item not in unique_contradictions:
                unique_contradictions.append(item)
        
        # Check if this is likely an analogical explanation rather than a contradiction
        analogy_terms = ["like", "as if", "imagine", "think of", "similar to", "resembles", "compared to"]
        uses_analogy = any(term in student_text.lower() for term in analogy_terms)
        overall_similarity = self.evaluator.compute_similarity(student_text, correct_text, question_type="theory")
        
        # If using analogies and overall similarity is high, reduce or eliminate contradictions
        if uses_analogy and overall_similarity > 0.75 and contradiction_penalty > 0:
            # Clear contradictions if they're likely due to analogical explanation
            unique_contradictions = []
            contradiction_penalty = 0
            unique_contradictions.append(
                "Note: The student's answer uses analogies to explain the same concepts present in the correct answer. "
                "While the language differs, the core understanding appears to be correct."
            )
        # If we have high raw similarity but detected contradictions, provide a general explanation
        elif unique_contradictions and overall_similarity > 0.7:
            general_explanation = (
                "Note: While the student's answer uses similar terminology to the correct answer, "
                "it presents a fundamentally different or incorrect understanding of the core concepts. "
                "This is a common pattern in misconceptions where familiar terms are used "
                "but with an incorrect conceptual framework."
            )
            unique_contradictions.append(general_explanation)
        
        return min(contradiction_penalty, 0.9), unique_contradictions  # Cap at 90% penalty

    def extract_key_concepts_and_terms(self, text):
        """Extract key concepts and domain-specific terminology from text"""
        # Basic preprocessing
        text = text.lower()
        
        # Extract noun phrases and technical terms using simple patterns
        # This is a simplified approach - in a production system you might use NLP libraries
        
        # Common technical/academic phrases (2-3 word combinations)
        phrases = []
        words = text.split()
        
        # Extract potential technical terms (2-3 word phrases)
        for i in range(len(words)-1):
            if words[i] not in self.common_words and words[i+1] not in self.common_words:
                phrases.append(f"{words[i]} {words[i+1]}")
        
        for i in range(len(words)-2):
            if words[i] not in self.common_words and words[i+2] not in self.common_words:
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # Single technical terms (often nouns)
        technical_terms = [word for word in words 
                          if word not in self.common_words 
                          and len(word) > 3  # Avoid short words
                          and not any(c.isdigit() for c in word)]  # Avoid numbers
        
        # Combine and remove duplicates
        all_concepts = list(set(phrases + technical_terms))
        
        # Sort by length (longer phrases first) to prioritize specific concepts
        all_concepts.sort(key=len, reverse=True)
        
        return all_concepts
    
    def analyze_concept_coverage(self, student_text, correct_text):
        """Analyze how many key concepts from the correct answer are covered in the student answer"""
        # Extract key concepts from both texts
        correct_concepts = self.extract_key_concepts_and_terms(correct_text)
        
        # Initialize coverage tracking
        covered_concepts = []
        missing_concepts = []
        concept_coverage = 0.0
        
        # Preprocess student text once
        student_text_lower = student_text.lower()
        
        # Pre-split student sentences once instead of for each concept
        student_sentences = [s.strip() for s in re.split(r'[.!?]', student_text) if s.strip()]
        
        # Calculate similarity matrix for all sentences at once if there are many concepts
        # This is more efficient than computing similarity for each concept-sentence pair
        concept_limit = 10  # Reduced from 15 for better performance
        limited_concepts = correct_concepts[:concept_limit]
        
        # For each concept in the correct answer, check if it's covered in the student answer
        for concept in limited_concepts:
            # Check for exact match first (fastest)
            if concept in student_text_lower:
                covered_concepts.append(concept)
                continue
            
            # Use a faster approach: check if any words from the concept appear in the student text
            concept_words = set(concept.lower().split())
            if len(concept_words) > 1 and any(word in student_text_lower for word in concept_words if len(word) > 3):
                # Only do expensive semantic similarity if there's some word overlap
                # Find the best matching sentence
                best_similarity = 0
                for sentence in student_sentences:
                    # Skip very short sentences
                    if len(sentence.split()) < 3:
                        continue
                    similarity = self.evaluator.compute_similarity(concept, sentence, question_type="theory")
                    best_similarity = max(best_similarity, similarity)
                
                # If similarity is high enough, consider the concept covered
                if best_similarity > 0.7:
                    covered_concepts.append(concept)
                else:
                    missing_concepts.append(concept)
            else:
                missing_concepts.append(concept)
        
        # Calculate coverage percentage
        if limited_concepts:
            concept_coverage = len(covered_concepts) / len(limited_concepts)
        
        return {
            "covered_concepts": covered_concepts,
            "missing_concepts": missing_concepts,
            "concept_coverage": concept_coverage,
            "total_concepts": len(limited_concepts)
        }
    
    def analyze_perspectives(self, student_text):
        """Analyze the different perspectives or viewpoints presented in the student's answer"""
        # General perspective indicator terms
        general_perspective_terms = [
            "however", "although", "nevertheless", "conversely", "alternatively", 
            "on the other hand", "in contrast", "another view", "different perspective",
            "some argue", "others believe", "while some", "debate", "controversy",
            "different approach", "alternatively", "opposing view", "different interpretation"
        ]
        
        # Domain-specific perspective indicators
        domain_perspective_terms = [
            "viewpoint", "perspective", "approach", "framework", "model", "theory",
            "paradigm", "concept", "principle", "methodology", "mechanism", "system",
            "standpoint", "angle", "lens", "context", "in terms of", "with respect to",
            "according to", "based on", "from a", "in the context of"
        ]
        
        # Technical domain prefixes that indicate specialized perspectives
        technical_domains = [
            "engineering", "scientific", "biological", "chemical", "physical", "mathematical",
            "computational", "economic", "psychological", "sociological", "philosophical",
            "historical", "political", "ethical", "medical", "legal", "technical", "theoretical",
            "practical", "empirical", "experimental", "clinical", "industrial", "environmental",
            "mechanical", "electrical", "control systems", "thermodynamic", "quantum", "classical",
            "modern", "traditional", "conventional", "alternative", "holistic", "reductionist"
        ]
        
        # Domain-independent perspective indicators
        # These are linguistic patterns that suggest different perspectives, regardless of domain
        perspective_patterns = [
            # Functional description patterns
            ("function", ["serves as", "functions as", "acts as", "works as", "operates as", "used as", 
                        "purpose is", "role is", "function is", "interface", "modulate"]),
            
            # Causal description patterns
            ("causal", ["causes", "results in", "leads to", "produces", "generates", "creates", 
                       "because", "due to", "as a result", "consequently", "therefore", "thus"]),
            
            # Structural description patterns
            ("structural", ["consists of", "composed of", "made up of", "contains", "includes", 
                          "comprises", "structure", "component", "element", "part", "system"]),
            
            # Process description patterns
            ("process", ["process", "step", "stage", "phase", "sequence", "procedure", 
                        "method", "technique", "approach", "operation", "mechanism"]),
            
            # Comparative description patterns
            ("comparative", ["compared to", "in contrast", "unlike", "similar to", "different from", 
                           "whereas", "while", "instead of", "rather than", "alternatively"]),
            
            # Technical description patterns
            ("technical", ["technical", "specialized", "specific", "precise", "exact", 
                         "detailed", "complex", "sophisticated", "advanced", "professional", "dynamics"]),
            
            # Abstract/conceptual patterns
            ("abstract", ["concept", "theory", "principle", "framework", "model", "paradigm", 
                         "viewpoint", "perspective", "approach", "signal", "input", "output"]),
                         
            # Mechanical/physical patterns
            ("mechanical", ["physical", "mechanical", "motion", "force", "pressure", "movement", 
                           "rotation", "transfer", "transmission", "torque"])
        ]
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', student_text) if s.strip()]
        
        # Find sentences that indicate multiple perspectives
        perspective_sentences = []
        domain_perspective_sentences = []
        
        # Check for general perspective indicators
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for general perspective terms
            for term in general_perspective_terms:
                if term in sentence_lower:
                    perspective_sentences.append(sentence)
                    break
            
            # Check for domain-specific perspective indicators
            for term in domain_perspective_terms:
                if term in sentence_lower:
                    # Check if there's a technical domain mentioned
                    for domain in technical_domains:
                        if domain in sentence_lower:
                            domain_perspective_sentences.append(sentence)
                            break
                    # Also check for phrases like "from a ... viewpoint"
                    if "from a" in sentence_lower and "viewpoint" in sentence_lower:
                        domain_perspective_sentences.append(sentence)
                        break
                    if "from the" in sentence_lower and any(p in sentence_lower for p in ["perspective", "standpoint", "view"]):
                        domain_perspective_sentences.append(sentence)
                        break
        
        # Combine unique perspective sentences
        all_perspective_sentences = list(set(perspective_sentences + domain_perspective_sentences))
        
        # Special case: Check for entire sentences that represent a technical/alternative perspective
        # This handles cases where the perspective isn't explicitly marked with indicator terms
        if len(sentences) >= 2:  # Need at least 2 sentences to compare perspectives
            first_sentence_topics = self.extract_key_concepts_and_terms(sentences[0])
            
            for i in range(1, len(sentences)):
                sentence = sentences[i]
                # Check if this sentence introduces new technical terms not in the first sentence
                sentence_topics = self.extract_key_concepts_and_terms(sentence)
                
                # If there are significant new terms, it might be a new perspective
                new_topics = [topic for topic in sentence_topics if topic not in first_sentence_topics]
                if len(new_topics) >= 2 and any(len(topic) > 5 for topic in new_topics):
                    if sentence not in all_perspective_sentences:
                        all_perspective_sentences.append(sentence)
        
        # Check for perspective patterns in the text
        student_text_lower = student_text.lower()
        perspective_type_counts = {}
        perspective_matches = []
        
        # Count occurrences of different perspective patterns
        for perspective_type, patterns in perspective_patterns:
            perspective_type_counts[perspective_type] = 0
            for pattern in patterns:
                if pattern in student_text_lower:
                    perspective_type_counts[perspective_type] += 1
                    # Find the sentence containing this pattern
                    for sentence in sentences:
                        if pattern in sentence.lower() and sentence not in perspective_matches:
                            perspective_matches.append(sentence)
        
        # Identify perspective types with significant presence
        significant_perspectives = []
        for p_type, count in perspective_type_counts.items():
            # Consider a perspective type significant if it has at least 2 pattern matches
            if count >= 2:
                significant_perspectives.append((p_type, count))
        
        # If we have multiple significant perspective types, this indicates multiple perspectives
        if len(significant_perspectives) >= 2:
            # Sort perspective types by pattern count (highest first)
            significant_perspectives.sort(key=lambda x: x[1], reverse=True)
            
            # The presence of multiple perspective types indicates multiple perspectives
            # Add a bonus to the perspective score
            perspective_bonus = min(0.5, (len(significant_perspectives) - 1) * 0.25)
            
            # Add any new perspective sentences
            for sentence in perspective_matches:
                if sentence not in all_perspective_sentences:
                    all_perspective_sentences.append(sentence)
        
        # Calculate a perspective score based on the number and quality of perspective indicators
        # Domain-specific perspectives are weighted more heavily
        general_count = len(perspective_sentences)
        domain_count = len(domain_perspective_sentences)
        
        # Calculate weighted score - domain perspectives count more
        weighted_count = general_count + (domain_count * 2)
        
        # Add bonus for multiple perspective types (implicit perspectives)
        perspective_bonus = 0
        if 'perspective_bonus' in locals():
            perspective_bonus = locals()['perspective_bonus']
        
        # Special case: Check for contrasting explanation styles
        # This helps identify when two explanations use different approaches to describe the same concept
        if len(all_perspective_sentences) >= 3:
            # Check for contrasting explanation styles by looking at the perspective types
            if 'significant_perspectives' in locals() and len(significant_perspectives) >= 2:
                # If we have both mechanical/physical and abstract/conceptual perspectives,
                # this is a strong indicator of multiple perspectives
                perspective_types = [p[0] for p in significant_perspectives]
                if ('mechanical' in perspective_types and 'abstract' in perspective_types) or \
                   ('mechanical' in perspective_types and 'function' in perspective_types) or \
                   ('process' in perspective_types and 'abstract' in perspective_types):
                    # This is a clear case of contrasting explanation styles
                    perspective_bonus = max(perspective_bonus, 0.5)
        
        # Calculate final score - cap at 1.0
        perspective_score = min(1.0, (weighted_count / 3) + perspective_bonus)  # Cap at 1.0
        
        # Prepare the perspective types information
        perspective_types_info = []
        if 'significant_perspectives' in locals() and significant_perspectives:
            perspective_types_info = [(p_type, count) for p_type, count in significant_perspectives]
        
        return {
            "perspective_score": perspective_score,
            "perspective_count": len(all_perspective_sentences),
            "perspective_sentences": all_perspective_sentences,
            "general_perspective_count": general_count,
            "domain_perspective_count": domain_count,
            "perspective_types": perspective_types_info,
            "perspective_bonus": perspective_bonus if 'perspective_bonus' in locals() else 0
        }
        
    def analyze_conceptual_relationships(self, student_text, correct_text):
        """Analyze how well the student explains relationships between concepts"""
        # Extract sentences containing relationship indicators
        relationship_terms = ["because", "therefore", "thus", "leads to", "results in", "causes", 
                             "affects", "influences", "depends on", "related to", "connected to"]
        
        # Limit the number of relationship terms for better performance
        core_relationship_terms = ["because", "therefore", "thus", "causes", "results in"]
        
        correct_sentences = [s.strip() for s in re.split(r'[.!?]', correct_text) if s.strip()]
        student_sentences = [s.strip() for s in re.split(r'[.!?]', student_text) if s.strip()]
        
        # Extract relationship sentences - use only core terms for better performance
        relationship_sentences_correct = [s for s in correct_sentences 
                                        if any(term in s.lower() for term in core_relationship_terms)]
        relationship_sentences_student = [s for s in student_sentences 
                                        if any(term in s.lower() for term in core_relationship_terms)]
        
        # Limit the number of sentences to analyze for better performance
        relationship_sentences_correct = relationship_sentences_correct[:3]  # Analyze at most 3 relationships
        
        # Calculate relationship coverage
        relationship_matches = 0
        relationship_explanations = []
        
        # Skip this analysis if there are no relationship sentences to compare
        if not relationship_sentences_correct or not relationship_sentences_student:
            return {
                "relationship_coverage": 0,
                "relationship_explanations": [],
                "total_relationships": len(relationship_sentences_correct)
            }
        
        # For each relationship in the correct answer, find the best match in student answer
        for correct_rel in relationship_sentences_correct:
            best_similarity = 0
            best_match = ""
            
            # Only compare with sentences that have some word overlap for efficiency
            correct_words = set(correct_rel.lower().split())
            filtered_student_sentences = []
            
            for student_rel in relationship_sentences_student:
                student_words = set(student_rel.lower().split())
                # Check for some word overlap before doing expensive similarity computation
                if len(correct_words.intersection(student_words)) >= 2:
                    filtered_student_sentences.append(student_rel)
            
            # If no sentences with word overlap, use all sentences but limit the number
            if not filtered_student_sentences:
                filtered_student_sentences = relationship_sentences_student[:3]
            
            for student_rel in filtered_student_sentences:
                similarity = self.evaluator.compute_similarity(correct_rel, student_rel, question_type="theory")
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = student_rel
            
            # If similarity is high enough, consider the relationship explained
            if best_similarity > 0.6:
                relationship_matches += 1
                relationship_explanations.append({
                    "correct": correct_rel,
                    "student": best_match,
                    "similarity": best_similarity
                })
        
        # Calculate relationship coverage percentage
        relationship_coverage = 0
        if relationship_sentences_correct:
            relationship_coverage = relationship_matches / len(relationship_sentences_correct)
        
        return {
            "relationship_coverage": relationship_coverage,
            "relationship_explanations": relationship_explanations,
            "total_relationships": len(relationship_sentences_correct)
        }

    def semantic_similarity_grading(self, student_answer, correct_answer, points,
                                  threshold=0.7, grading_scheme=None):
        """Grade text answers using BERT-based semantic similarity with concept checking"""
        
        # Extract and preprocess answers
        student_text = self.preprocess_text(student_answer)
        correct_text = self.preprocess_text(correct_answer)

        # Calculate similarity using our advanced evaluator
        raw_similarity = self.evaluator.compute_similarity(student_text, correct_text, question_type="theory")
        
        # Store the raw similarity for feedback
        similarity = raw_similarity
        
        # Check for contradictions
        contradiction_penalty, contradiction_reasons = self.detect_contradictions(student_text, correct_text)
        
        # Apply contradiction penalty if found
        adjusted_similarity = similarity
        if contradiction_penalty > 0:
            adjusted_similarity = max(0.1, similarity - contradiction_penalty)  # Ensure minimum score of 0.1
            similarity = adjusted_similarity  # Update similarity with penalty applied
        
        # Analyze concept coverage
        concept_analysis = self.analyze_concept_coverage(student_text, correct_text)
        
        # Analyze conceptual relationships
        relationship_analysis = self.analyze_conceptual_relationships(student_text, correct_text)
        
        # Analyze perspectives in the student's answer
        perspective_analysis = self.analyze_perspectives(student_text)
        
        # Calculate a weighted score based on multiple factors
        # Updated weighting scheme that includes perspective analysis
        # - Raw similarity (35%)
        # - Concept coverage (30%)
        # - Relationship explanation (15%)
        # - Perspective analysis (20%) - increased weight for perspectives
        weighted_score = (
            (adjusted_similarity * 0.35) + 
            (concept_analysis['concept_coverage'] * 0.3) + 
            (relationship_analysis['relationship_coverage'] * 0.15) +
            (perspective_analysis['perspective_score'] * 0.2)
        )
        
        # Scale from [0,1] to [-1,1] to maintain compatibility with existing code
        similarity = weighted_score * 2 - 1

        # Create a more nuanced grading approach that considers all factors
        # This is a completely revised grading system that doesn't just rely on similarity
        
        # Calculate a composite score based on all our metrics
        # This gives us a more holistic evaluation of the answer quality
        composite_score = weighted_score
        
        # Apply bonuses for exceptional performance in specific areas
        # 1. Bonus for excellent concept coverage
        if concept_analysis['concept_coverage'] > 0.8:
            composite_score += 0.1  # Significant bonus for high concept coverage
        
        # 2. Bonus for multiple perspectives
        if perspective_analysis['perspective_score'] > 0.7:
            composite_score += 0.1  # Significant bonus for multiple perspectives
        
        # 3. Bonus for good relationship explanation
        if relationship_analysis['relationship_coverage'] > 0.7:
            composite_score += 0.05  # Moderate bonus for explaining relationships well
        
        # 4. Bonus for high similarity without contradictions
        if adjusted_similarity > 0.8 and contradiction_penalty == 0:
            composite_score += 0.05  # Moderate bonus for high similarity without contradictions
        
        # Cap the composite score at 1.0
        composite_score = min(1.0, composite_score)
        
        # New grading scheme based on composite score
        # This is more generous to reward conceptual understanding
        new_grading_scheme = [
            (0.85, 1.0),  # 85-100% composite score = full points
            (0.75, 0.9),  # 75-85% composite score = 90% points
            (0.65, 0.8),  # 65-75% composite score = 80% points
            (0.55, 0.7),  # 55-65% composite score = 70% points
            (0.45, 0.6),  # 45-55% composite score = 60% points
            (0.35, 0.5),  # 35-45% composite score = 50% points
            (0.25, 0.4),  # 25-35% composite score = 40% points
            (0.15, 0.3),  # 15-25% composite score = 30% points
            (0.05, 0.2)   # 5-15% composite score = 20% points
        ]
        
        # Use the provided grading scheme if specified, otherwise use our new scheme
        if grading_scheme is None:
            grading_scheme = new_grading_scheme

        # Apply grading scheme to the composite score instead of just similarity
        final_score = 0
        for min_score, score_multiplier in grading_scheme:
            if composite_score >= min_score:
                final_score = points * score_multiplier
                break

        # Prepare detailed feedback
        feedback = {
            "raw_similarity": raw_similarity,
            "adjusted_similarity": adjusted_similarity if contradiction_penalty > 0 else raw_similarity,
            "weighted_score": weighted_score,
            "composite_score": composite_score,  # Add the composite score to feedback
            "contradiction_penalty": contradiction_penalty,
            "contradiction_reasons": contradiction_reasons,
            "concept_analysis": concept_analysis,
            "relationship_analysis": relationship_analysis,
            "perspective_analysis": perspective_analysis,
            "final_score": final_score,
            "max_score": points,
            "grading_explanation": {  # Add explanations for the grade
                "concept_bonus": concept_analysis['concept_coverage'] > 0.8,
                "perspective_bonus": perspective_analysis['perspective_score'] > 0.7,
                "relationship_bonus": relationship_analysis['relationship_coverage'] > 0.7,
                "similarity_bonus": adjusted_similarity > 0.8 and contradiction_penalty == 0
            }
        }
        
        return final_score, feedback

    def keyword_based_grading(self, student_answer, keywords, points, required_matches=None):
        """Grade based on presence of key words/phrases with semantic similarity"""
        if required_matches is None:
            required_matches = len(keywords)

        student_text = self.preprocess_text(student_answer)

        # Calculate semantic similarity for each keyword using our advanced evaluator
        matches = 0
        for keyword in keywords:
            # Use our evaluator for better semantic matching
            similarity = self.evaluator.compute_similarity(keyword, student_text, question_type="theory")
            
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

    def code_grading(self, student_answer, correct_answer, points, threshold=0.7):
        """Grade code-based answers using CodeBERT"""
        student_code = self.extract_answer(student_answer)
        correct_code = self.extract_answer(correct_answer)
        
        # Use our specialized code evaluator
        similarity = self.evaluator.compute_similarity(student_code, correct_code, question_type="code")
        
        # Simple grading scheme for code
        if similarity >= 0.9:  # Near perfect match
            return points
        elif similarity >= 0.8:  # Good match
            return points * 0.9
        elif similarity >= 0.7:  # Acceptable match
            return points * 0.7
        elif similarity >= 0.5:  # Partial match
            return points * 0.5
        else:  # Poor match
            return points * 0.2
    
    def grade_question(self, question_type, student_answer, correct_answer, points, **kwargs):
        """Main grading function that routes to appropriate grading method"""
        grading_methods = {
            'exact': self.exact_match_grading,
            'numerical': self.numerical_grading,
            'text': self.semantic_similarity_grading,
            'keyword': self.keyword_based_grading,
            'formula': self.formula_grading,
            'code': self.code_grading  # Add support for code-based questions
        }

        if question_type not in grading_methods:
            raise ValueError(f"Unsupported question type: {question_type}")

        result = grading_methods[question_type](student_answer, correct_answer, points, **kwargs)
        
        # Check if the result is a tuple with feedback (for semantic_similarity_grading)
        if isinstance(result, tuple) and len(result) == 2:
            score, feedback = result
            return score, feedback
        else:
            # For other grading methods that don't yet return detailed feedback
            return result

    def grade_assessment(self, assessment_data):
        """Grade entire assessment with multiple questions"""
        total_score = 0
        feedback = []

        for question in assessment_data:
            result = self.grade_question(
                question['type'],
                question['student_answer'],
                question['correct_answer'],
                question['points'],
                **question.get('grading_params', {})
            )
            
            # Check if we got detailed feedback
            if isinstance(result, tuple) and len(result) == 2:
                score, detailed_feedback = result
                
                # Prepare the feedback entry with basic information
                feedback_entry = {
                    'question_number': question['number'],
                    'score': score,
                    'max_points': question['points'],
                    'percentage': (score / question['points']) * 100,
                    'detailed_feedback': detailed_feedback  # Include all detailed feedback
                }
                
                # Add similarity percentage from detailed feedback if available
                if 'raw_similarity' in detailed_feedback:
                    # Basic similarity metrics
                    feedback_entry['similarity_percentage'] = detailed_feedback['raw_similarity'] * 100
                    feedback_entry['adjusted_similarity_percentage'] = detailed_feedback['adjusted_similarity'] * 100
                    feedback_entry['contradiction_reasons'] = detailed_feedback['contradiction_reasons']
                    
                    # Add concept analysis and relationship data for the UI
                    if 'concept_analysis' in detailed_feedback:
                        # Already included in detailed_feedback, but we're ensuring it's accessible
                        feedback_entry['concept_analysis'] = detailed_feedback['concept_analysis']
                        feedback_entry['relationship_analysis'] = detailed_feedback['relationship_analysis']
            else:
                # For other question types without detailed feedback
                score = result
                
                # Calculate similarity percentage for text and code answers
                similarity_percentage = None
                if question['type'] in ['text', 'code']:
                    student_text = self.preprocess_text(question['student_answer'])
                    correct_text = self.preprocess_text(question['correct_answer'])
                    
                    # Use the appropriate evaluator based on question type
                    similarity = self.evaluator.compute_similarity(
                        student_text, 
                        correct_text, 
                        question_type="code" if question['type'] == 'code' else "theory"
                    )
                    similarity_percentage = similarity * 100
                
                feedback_entry = {
                    'question_number': question['number'],
                    'score': score,
                    'max_points': question['points'],
                    'percentage': (score / question['points']) * 100,
                    'similarity_percentage': similarity_percentage
                }
            
            feedback.append(feedback_entry)
            total_score += score

        return {
            'total_score': total_score,
            'max_score': sum(q['points'] for q in assessment_data),
            'percentage': (total_score / sum(q['points'] for q in assessment_data)) * 100,
            'question_feedback': feedback
        }