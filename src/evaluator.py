"""
RAG System Evaluator - Q4: Évaluation du système RAG/LLM
Evaluates the relevance and quality of RAG system responses.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics


@dataclass
class EvaluationMetrics:
    """Data class for evaluation metrics."""
    relevance_score: float
    completeness_score: float
    accuracy_score: float
    source_quality_score: float
    overall_score: float
    feedback: str


class RAGEvaluator:
    """
    Q4: RAG system evaluation with multiple metrics.
    Evaluates response relevance, completeness, accuracy, and source quality.
    """
    
    def __init__(self):
        """Initialize the RAG evaluator."""
        self.evaluation_history = []
    
    def evaluate_response(self, 
                         question: str,
                         answer: str,
                         sources: List[Dict],
                         expected_answer: Optional[str] = None,
                         ground_truth_sources: Optional[List[str]] = None) -> EvaluationMetrics:
        """
        Evaluate a RAG system response using multiple metrics.
        
        Args:
            question: Original question
            answer: Generated answer
            sources: List of source documents used
            expected_answer: Expected/reference answer (optional)
            ground_truth_sources: Expected source documents (optional)
            
        Returns:
            EvaluationMetrics with scores and feedback
        """
        print(f"📊 Evaluating response for: {question[:50]}...")
        
        # 1. Relevance Score (0-1): How well does the answer address the question?
        relevance_score = self._evaluate_relevance(question, answer)
        
        # 2. Completeness Score (0-1): How complete is the answer?
        completeness_score = self._evaluate_completeness(question, answer, expected_answer)
        
        # 3. Accuracy Score (0-1): How accurate is the answer based on sources?
        accuracy_score = self._evaluate_accuracy(answer, sources)
        
        # 4. Source Quality Score (0-1): Quality and relevance of sources
        source_quality_score = self._evaluate_source_quality(sources, ground_truth_sources)
        
        # 5. Overall Score: Weighted average
        overall_score = self._calculate_overall_score(
            relevance_score, completeness_score, accuracy_score, source_quality_score
        )
        
        # Generate feedback
        feedback = self._generate_feedback(
            relevance_score, completeness_score, accuracy_score, source_quality_score
        )
        
        metrics = EvaluationMetrics(
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            source_quality_score=source_quality_score,
            overall_score=overall_score,
            feedback=feedback
        )
        
        # Store in history
        self.evaluation_history.append({
            "question": question,
            "answer": answer,
            "metrics": metrics,
            "sources_count": len(sources)
        })
        
        return metrics
    
    def _evaluate_relevance(self, question: str, answer: str) -> float:
        """
        Evaluate how relevant the answer is to the question.
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            Relevance score (0-1)
        """
        # Simple heuristic-based relevance evaluation
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais', 'donc', 'car', 'ni', 'or'}
        question_words -= stop_words
        answer_words -= stop_words
        
        if not question_words:
            return 0.5  # Neutral score if no meaningful words
        
        # Calculate word overlap
        overlap = len(question_words.intersection(answer_words))
        relevance = min(1.0, overlap / len(question_words))
        
        # Bonus for direct question addressing
        question_indicators = ['quoi', 'qui', 'quand', 'où', 'comment', 'pourquoi', 'what', 'who', 'when', 'where', 'how', 'why']
        if any(indicator in question.lower() for indicator in question_indicators):
            if len(answer) > 50:  # Substantial answer
                relevance += 0.2
        
        return min(1.0, relevance)
    
    def _evaluate_completeness(self, question: str, answer: str, expected_answer: Optional[str] = None) -> float:
        """
        Evaluate how complete the answer is.
        
        Args:
            question: Original question
            answer: Generated answer
            expected_answer: Reference answer (optional)
            
        Returns:
            Completeness score (0-1)
        """
        # Length-based completeness (basic heuristic)
        if len(answer) < 20:
            return 0.2  # Very short answers are likely incomplete
        elif len(answer) < 50:
            return 0.5  # Short answers
        elif len(answer) < 150:
            return 0.8  # Medium answers
        else:
            return 1.0  # Long answers assumed more complete
        
        # If expected answer is provided, compare coverage
        if expected_answer:
            expected_words = set(expected_answer.lower().split())
            answer_words = set(answer.lower().split())
            
            if expected_words:
                coverage = len(expected_words.intersection(answer_words)) / len(expected_words)
                return min(1.0, coverage + 0.3)  # Bonus for having expected content
        
        return min(1.0, len(answer) / 200)  # Normalize by expected length
    
    def _evaluate_accuracy(self, answer: str, sources: List[Dict]) -> float:
        """
        Evaluate accuracy based on source quality and confidence.
        
        Args:
            answer: Generated answer
            sources: Source documents used
            
        Returns:
            Accuracy score (0-1)
        """
        if not sources:
            return 0.3  # Low accuracy if no sources
        
        # Average source similarity as proxy for accuracy
        similarities = [source.get('similarity', 0.0) for source in sources]
        avg_similarity = statistics.mean(similarities) if similarities else 0.0
        
        # Bonus for multiple high-quality sources
        high_quality_sources = sum(1 for s in similarities if s > 0.7)
        source_bonus = min(0.3, high_quality_sources * 0.1)
        
        return min(1.0, avg_similarity + source_bonus)
    
    def _evaluate_source_quality(self, sources: List[Dict], ground_truth_sources: Optional[List[str]] = None) -> float:
        """
        Evaluate the quality and relevance of sources used.
        
        Args:
            sources: Source documents used
            ground_truth_sources: Expected sources (optional)
            
        Returns:
            Source quality score (0-1)
        """
        if not sources:
            return 0.0
        
        # Average similarity score of sources
        similarities = [source.get('similarity', 0.0) for source in sources]
        avg_similarity = statistics.mean(similarities)
        
        # Diversity bonus (different sources)
        unique_sources = len(set(source.get('source', '') for source in sources))
        diversity_bonus = min(0.2, unique_sources * 0.05)
        
        # Ground truth matching (if provided)
        ground_truth_bonus = 0.0
        if ground_truth_sources:
            used_sources = [source.get('source', '') for source in sources]
            matches = sum(1 for gt in ground_truth_sources if any(gt in used for used in used_sources))
            ground_truth_bonus = min(0.3, matches / len(ground_truth_sources))
        
        return min(1.0, avg_similarity + diversity_bonus + ground_truth_bonus)
    
    def _calculate_overall_score(self, relevance: float, completeness: float, accuracy: float, source_quality: float) -> float:
        """
        Calculate weighted overall score.
        
        Args:
            relevance: Relevance score
            completeness: Completeness score
            accuracy: Accuracy score
            source_quality: Source quality score
            
        Returns:
            Overall weighted score (0-1)
        """
        # Weighted average (relevance and accuracy are most important)
        weights = {
            'relevance': 0.35,
            'completeness': 0.20,
            'accuracy': 0.30,
            'source_quality': 0.15
        }
        
        overall = (
            relevance * weights['relevance'] +
            completeness * weights['completeness'] +
            accuracy * weights['accuracy'] +
            source_quality * weights['source_quality']
        )
        
        return round(overall, 3)
    
    def _generate_feedback(self, relevance: float, completeness: float, accuracy: float, source_quality: float) -> str:
        """
        Generate human-readable feedback based on scores.
        
        Args:
            relevance: Relevance score
            completeness: Completeness score
            accuracy: Accuracy score
            source_quality: Source quality score
            
        Returns:
            Feedback string
        """
        feedback_parts = []
        
        # Relevance feedback
        if relevance >= 0.8:
            feedback_parts.append("✅ Réponse très pertinente")
        elif relevance >= 0.6:
            feedback_parts.append("🟡 Réponse moyennement pertinente")
        else:
            feedback_parts.append("❌ Réponse peu pertinente")
        
        # Completeness feedback
        if completeness >= 0.8:
            feedback_parts.append("✅ Réponse complète")
        elif completeness >= 0.6:
            feedback_parts.append("🟡 Réponse partiellement complète")
        else:
            feedback_parts.append("❌ Réponse incomplète")
        
        # Accuracy feedback
        if accuracy >= 0.8:
            feedback_parts.append("✅ Haute précision")
        elif accuracy >= 0.6:
            feedback_parts.append("🟡 Précision moyenne")
        else:
            feedback_parts.append("❌ Faible précision")
        
        # Source quality feedback
        if source_quality >= 0.8:
            feedback_parts.append("✅ Sources de haute qualité")
        elif source_quality >= 0.6:
            feedback_parts.append("🟡 Sources de qualité moyenne")
        else:
            feedback_parts.append("❌ Sources de faible qualité")
        
        return " | ".join(feedback_parts)
    
    def evaluate_batch(self, qa_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate multiple QA results in batch.
        
        Args:
            qa_results: Dictionary of question -> result mappings
            
        Returns:
            Batch evaluation results with statistics
        """
        print(f"\n📊 Evaluating {len(qa_results)} QA pairs...")
        
        batch_metrics = []
        detailed_results = {}
        
        for question, result in qa_results.items():
            metrics = self.evaluate_response(
                question=question,
                answer=result.get('answer', ''),
                sources=result.get('sources', [])
            )
            
            batch_metrics.append(metrics)
            detailed_results[question] = {
                'result': result,
                'metrics': metrics
            }
        
        # Calculate batch statistics
        avg_relevance = statistics.mean([m.relevance_score for m in batch_metrics])
        avg_completeness = statistics.mean([m.completeness_score for m in batch_metrics])
        avg_accuracy = statistics.mean([m.accuracy_score for m in batch_metrics])
        avg_source_quality = statistics.mean([m.source_quality_score for m in batch_metrics])
        avg_overall = statistics.mean([m.overall_score for m in batch_metrics])
        
        return {
            'detailed_results': detailed_results,
            'batch_statistics': {
                'avg_relevance': round(avg_relevance, 3),
                'avg_completeness': round(avg_completeness, 3),
                'avg_accuracy': round(avg_accuracy, 3),
                'avg_source_quality': round(avg_source_quality, 3),
                'avg_overall': round(avg_overall, 3),
                'total_questions': len(qa_results)
            }
        }
    
    def print_evaluation_report(self, metrics: EvaluationMetrics):
        """
        Print formatted evaluation report.
        
        Args:
            metrics: EvaluationMetrics instance
        """
        print(f"\n" + "="*60)
        print(f"📊 RAPPORT D'ÉVALUATION RAG")
        print("="*60)
        
        print(f"\n📈 SCORES DÉTAILLÉS:")
        print(f"   Pertinence:      {metrics.relevance_score:.1%}")
        print(f"   Complétude:      {metrics.completeness_score:.1%}")
        print(f"   Précision:       {metrics.accuracy_score:.1%}")
        print(f"   Qualité sources: {metrics.source_quality_score:.1%}")
        print(f"   Score global:    {metrics.overall_score:.1%}")
        
        print(f"\n💬 FEEDBACK:")
        print(f"   {metrics.feedback}")
        
        # Overall assessment
        if metrics.overall_score >= 0.8:
            assessment = "🟢 EXCELLENT"
        elif metrics.overall_score >= 0.6:
            assessment = "🟡 SATISFAISANT"
        else:
            assessment = "🔴 À AMÉLIORER"
        
        print(f"\n🎯 ÉVALUATION GLOBALE: {assessment}")
        print("="*60)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all evaluations performed.
        
        Returns:
            Summary statistics of evaluation history
        """
        if not self.evaluation_history:
            return {"message": "Aucune évaluation effectuée"}
        
        overall_scores = [eval_data['metrics'].overall_score for eval_data in self.evaluation_history]
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'average_score': round(statistics.mean(overall_scores), 3),
            'best_score': round(max(overall_scores), 3),
            'worst_score': round(min(overall_scores), 3),
            'score_distribution': {
                'excellent (≥80%)': sum(1 for score in overall_scores if score >= 0.8),
                'satisfaisant (60-79%)': sum(1 for score in overall_scores if 0.6 <= score < 0.8),
                'à améliorer (<60%)': sum(1 for score in overall_scores if score < 0.6)
            }
        }
