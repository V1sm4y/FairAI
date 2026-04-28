"""
Bias Insights Module - Uses Gemini API to generate insights for reducing AI bias.
"""

import os
from pathlib import Path

import google.generativeai as genai

# Configure Gemini API with the provided key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyB1jISZ3ddjiPgjc1yomdcnkz3P0ekuxxU")
genai.configure(api_key=GEMINI_API_KEY)


def generate_bias_insights(fairness_report: dict) -> str:
    """
    Generate insights on how to reduce AI bias based on fairness analysis results.
    
    Args:
        fairness_report: Dictionary containing fairness metrics and analysis results
        
    Returns:
        String containing AI-generated insights and recommendations
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Build a prompt from the fairness report
    prompt = _build_insight_prompt(fairness_report)
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {str(e)}"


def _build_insight_prompt(fairness_report: dict) -> str:
    """Build a detailed prompt for the Gemini API based on fairness report."""
    
    prompt = """You are an AI fairness expert. Based on the following fairness analysis results, 
provide actionable insights on how to reduce AI bias in the model. 

Focus on:
1. Which demographic groups are most affected by bias
2. Specific technical recommendations to reduce bias
3. Data collection and preprocessing strategies
4. Model selection and training adjustments

Fairness Analysis Results:
"""
    
    # Add fairness metrics to the prompt
    if "metrics" in fairness_report:
        prompt += "\nFairness Metrics:\n"
        for metric, value in fairness_report["metrics"].items():
            prompt += f"- {metric}: {value}\n"
    
    if "disparity_analysis" in fairness_report:
        prompt += "\nDisparity Analysis:\n"
        for key, value in fairness_report["disparity_analysis"].items():
            prompt += f"- {key}: {value}\n"
    
    if "risk_factors" in fairness_report:
        prompt += "\nIdentified Risk Factors:\n"
        for factor in fairness_report["risk_factors"]:
            prompt += f"- {factor}\n"
    
    prompt += """
Please provide specific, actionable recommendations in a clear, structured format.
"""
    
    return prompt


def get_bias_reduction_suggestions(sensitive_column: str, disparity_score: float) -> str:
    """
    Get specific suggestions for reducing bias related to a sensitive attribute.
    
    Args:
        sensitive_column: Name of the sensitive column (e.g., 'gender', 'race')
        disparity_score: Calculated disparity score
        
    Returns:
        String containing targeted suggestions
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""As an AI fairness expert, provide specific recommendations to reduce bias 
for the sensitive attribute '{sensitive_column}' which has a disparity score of {disparity_score}.

Provide:
1. Data-level interventions
2. Model-level interventions
3. Post-processing techniques
4. Evaluation strategies

Be concise and actionable."""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating suggestions: {str(e)}"