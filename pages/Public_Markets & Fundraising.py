import streamlit as st
import pandas as pd
import numpy as np
import json
import openai
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import traceback

# Import Ragas components
from ragas import SingleTurnSample
from ragas.metrics import (
    LLMContextRecall,
    NonLLMContextRecall,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    ResponseRelevancy,
    Faithfulness
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import nest_asyncio

# Apply nest_asyncio to make async code work in Streamlit
nest_asyncio.apply()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAGAS-Evaluator")

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
EVALUATION_MODEL = "gpt-3.5-turbo"

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables with defaults"""
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None
    
    if "ground_truths" not in st.session_state:
        st.session_state.ground_truths = []
    
    if "questions" not in st.session_state:
        st.session_state.questions = []
    
    if "answers" not in st.session_state:
        st.session_state.answers = []
    
    if "contexts" not in st.session_state:
        st.session_state.contexts = []
    
    if "reference_contexts" not in st.session_state:
        st.session_state.reference_contexts = []
    
    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0

def load_example_data():
    """Load example evaluation data"""
    example_data = {
        "questions": [
            "What are the main challenges mentioned in citizen ideas?",
            "Summarize the current status of municipal projects",
            "What common themes appear in citizen comments?"
        ],
        "answers": [
            "Based on the context provided, the main challenges mentioned in citizen ideas include infrastructure issues, public service deficiencies, and environmental concerns. Citizens have highlighted problems with road maintenance, inadequate public transportation, insufficient waste management, and lack of green spaces in urban areas.",
            "According to the context, municipal projects are in various stages of implementation. Some projects are in the planning phase, others are under construction, and a few have been completed. Budget constraints have affected timeline projections for several infrastructure initiatives. The water management project is currently 60% complete, while the park renovation is in early stages at about 25% completion.",
            "The common themes that appear in citizen comments include concerns about infrastructure development, requests for improved public services, environmental protection issues, and demands for greater transparency in governance. There are recurring mentions of road quality, public transportation reliability, and the need for more community engagement in decision-making processes."
        ],
        "contexts": [
            [
                "[citizen_ideas] Challenge: Poor road conditions throughout the city. Solution: Implement a systematic road maintenance program with quarterly inspections.",
                "[citizen_ideas] Challenge: Lack of accessible green spaces. Solution: Convert vacant lots into community gardens and small parks.",
                "[citizen_ideas] Challenge: Inefficient public transportation. Solution: Increase bus frequency and create dedicated lanes."
            ],
            [
                "[municipal_projects] Project: Central Water Management System\nStatus: In Progress\nBudget: $2.3M\nDescription: Implementation of smart water management infrastructure across downtown area. Currently 60% complete.",
                "[municipal_projects] Project: Eastside Road Expansion\nStatus: Planning\nBudget: $5.7M\nDescription: Expansion of main thoroughfare to reduce traffic congestion. Environmental assessment phase.",
                "[municipal_projects] Project: Community Park Renovation\nStatus: In Progress\nBudget: $1.2M\nDescription: Renovation of central community park including new playground equipment and walking paths. Currently 25% complete."
            ],
            [
                "[citizen_comments] The bus schedule is unreliable, especially during rush hour. I've been late to work three times this month because of delays.",
                "[citizen_comments] I appreciate the new online portal for submitting permit requests. It's much more efficient than the old paper system.",
                "[citizen_comments] Our neighborhood has been asking for road repairs for over two years. The potholes are damaging vehicles and creating safety hazards."
            ]
        ],
        "ground_truths": [
            "Citizen ideas mention challenges including poor road conditions, lack of green spaces, and inefficient public transportation. Solutions proposed include systematic maintenance programs, converting vacant lots to community spaces, and improving bus service.",
            "Municipal projects are in various stages from planning to implementation. The Central Water Management System is 60% complete with a $2.3M budget, Eastside Road Expansion is in planning with a $5.7M budget, and Community Park Renovation is 25% complete with a $1.2M budget.",
            "Common themes in citizen comments include concerns about public transportation reliability, appreciation for digital government services, and frustration with delayed infrastructure maintenance, particularly regarding road conditions."
        ],
        "reference_contexts": [
            [
                "[citizen_ideas] Challenge: Poor road conditions throughout the city. Solution: Implement a systematic road maintenance program with quarterly inspections.",
                "[citizen_ideas] Challenge: Lack of accessible green spaces. Solution: Convert vacant lots into community gardens and small parks.",
                "[citizen_ideas] Challenge: Inefficient public transportation. Solution: Increase bus frequency and create dedicated lanes.",
                "[citizen_ideas] Challenge: Limited recycling programs. Solution: Expand recycling collection to all neighborhoods and add composting options."
            ],
            [
                "[municipal_projects] Project: Central Water Management System\nStatus: In Progress\nBudget: $2.3M\nDescription: Implementation of smart water management infrastructure across downtown area. Currently 60% complete.",
                "[municipal_projects] Project: Eastside Road Expansion\nStatus: Planning\nBudget: $5.7M\nDescription: Expansion of main thoroughfare to reduce traffic congestion. Environmental assessment phase.",
                "[municipal_projects] Project: Community Park Renovation\nStatus: In Progress\nBudget: $1.2M\nDescription: Renovation of central community park including new playground equipment and walking paths. Currently 25% complete.",
                "[municipal_projects] Project: Downtown Revitalization\nStatus: Completed\nBudget: $3.4M\nDescription: Renovation of downtown area including new streetlights, sidewalks, and business facade improvements."
            ],
            [
                "[citizen_comments] The bus schedule is unreliable, especially during rush hour. I've been late to work three times this month because of delays.",
                "[citizen_comments] I appreciate the new online portal for submitting permit requests. It's much more efficient than the old paper system.",
                "[citizen_comments] Our neighborhood has been asking for road repairs for over two years. The potholes are damaging vehicles and creating safety hazards.",
                "[citizen_comments] The new recycling program has made it much easier for my family to reduce our waste. I hope it expands to include composting soon."
            ]
        ]
    }
    return example_data

def format_context_for_display(context_list):
    """Format context list for readable display"""
    if not context_list:
        return "No context available"
    
    formatted = []
    for i, ctx in enumerate(context_list):
        formatted.append(f"**Source {i+1}:**\n{ctx}")
    
    return "\n\n".join(formatted)

async def evaluate_samples(questions, answers, contexts, reference_contexts, ground_truths, 
                          metrics, openai_api_key, model_name="gpt-3.5-turbo"):
    """Evaluate samples using Ragas metrics"""
    # Initialize LLM and embeddings
    llm = ChatOpenAI(
        model=model_name,
        api_key=openai_api_key,
        temperature=0
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=openai_api_key
    )
    
    # Initialize metrics
    metric_instances = {}
    
    if "context_recall" in metrics:
        metric_instances["context_recall"] = LLMContextRecall(llm=llm)
    
    if "context_recall_non_llm" in metrics:
        metric_instances["context_recall_non_llm"] = NonLLMContextRecall()
    
    if "context_precision" in metrics:
        metric_instances["context_precision"] = LLMContextPrecisionWithoutReference(llm=llm)
    
    if "context_precision_with_ref" in metrics:
        metric_instances["context_precision_with_ref"] = LLMContextPrecisionWithReference(llm=llm)
    
    if "response_relevancy" in metrics:
        metric_instances["response_relevancy"] = ResponseRelevancy(llm=llm, embeddings=embeddings)
    
    if "answer_faithfulness" in metrics:
        metric_instances["answer_faithfulness"] = Faithfulness(llm=llm)
    
    # Evaluate each sample
    results = {metric_name: [] for metric_name in metric_instances}
    details = []
    
    for i in range(len(questions)):
        # Create Ragas sample
        full_sample = SingleTurnSample(
            user_input=questions[i],
            response=answers[i],
            reference=ground_truths[i] if i < len(ground_truths) else None,
            retrieved_contexts=contexts[i] if i < len(contexts) else [],
            reference_contexts=reference_contexts[i] if i < len(reference_contexts) else contexts[i]
        )
        
        sample_results = {}
        
        # Apply each metric
        for metric_name, metric in metric_instances.items():
            try:
                # Add logging BEFORE running the metric
                logger.info(f"Attempting to evaluate {metric_name} for sample {i}")
                logger.info(f"Sample data for {metric_name}: question={questions[i][:50]}...")
                logger.info(f"Sample context count: {len(contexts[i])}")
                if metric_name == "context_recall_non_llm":
                    # This metric doesn't need question or answer
                    non_llm_sample = SingleTurnSample(
                        retrieved_contexts=contexts[i] if i < len(contexts) else [],
                        reference_contexts=reference_contexts[i] if i < len(reference_contexts) else contexts[i]
                    )
                    score = await metric.single_turn_ascore(non_llm_sample)
                else:
                    # Use the full sample for other metrics
                    logger.info(f"Using full sample for {metric_name}")
                    score = await metric.single_turn_ascore(full_sample)
                logger.info(f"Successfully evaluated {metric_name}, score: {score}")
                results[metric_name].append(score)
                sample_results[metric_name] = score
            except Exception as e:
                logger.error(f"Error evaluating {metric_name} for sample {i}: {e}")
                logger.error(traceback.format_exc())
                results[metric_name].append(0.0)  # Default to 0 on error
                sample_results[metric_name] = 0.0
        
        details.append({
            "question": questions[i],
            "answer": answers[i],
            "context": contexts[i],
            "ground_truth": ground_truths[i] if i < len(ground_truths) else None,
            "scores": sample_results
        })
    
    return results, details

def main():
    """Main function for the Ragas evaluator UI"""
    # Initialize session state
    initialize_session_state()
    
    # Apply custom styling
    st.markdown(
        """
        <style>
        .main-header {
            background-color: #f63366;
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .section-header {
            background-color: #4285f4;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        .metric-card {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        
        .score-number {
            font-size: 24px;
            font-weight: bold;
            color: #4285f4;
        }
        
        .question-box {
            background-color: #e8f0fe;
            border-left: 5px solid #4285f4;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .answer-box {
            background-color: #fce8e6;
            border-left: 5px solid #ea4335;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .ground-truth-box {
            background-color: #e6f4ea;
            border-left: 5px solid #34a853;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .context-box {
            background-color: #fff8e1;
            border-left: 5px solid #fbbc04;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            max-height: 300px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Main header
    st.markdown('<div class="main-header"><h1>CivicCatalyst Chatbot Evaluator</h1></div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if api_key:
            openai.api_key = api_key
        
        st.markdown("---")
        
        # Evaluation options
        st.subheader("Evaluation Options")
        
        # Model selection
        model_name = st.selectbox(
            "Evaluation Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0
        )
        
        # Select which metrics to use
        st.write("Select metrics to evaluate:")
        use_context_recall = st.checkbox("Context Recall (LLM)", value=True, 
                                        help="Measures if the answer captures the key information from the context (using LLM)")
        use_context_recall_non_llm = st.checkbox("Context Recall (Non-LLM)", value=False, 
                                              help="Measures context recall without using an LLM")
        use_context_precision = st.checkbox("Context Precision", value=True, 
                                          help="Measures if the context provided was precise and useful")
        use_context_precision_with_ref = st.checkbox("Context Precision (with Reference)", value=True, 
                                                  help="Measures context precision with reference")
        use_response_relevancy = st.checkbox("Response Relevancy", value=True, 
                                           help="Measures if the answer is relevant to the question")
        use_answer_faithfulness = st.checkbox("Answer Faithfulness", value=True, 
                                            help="Measures if the answer is factually consistent with the provided context")
        
        # Create metrics list based on selections
        selected_metrics = []
        if use_context_recall:
            selected_metrics.append("context_recall")
        if use_context_recall_non_llm:
            selected_metrics.append("context_recall_non_llm")
        if use_context_precision:
            selected_metrics.append("context_precision")
        if use_context_precision_with_ref:
            selected_metrics.append("context_precision_with_ref")
        if use_response_relevancy:
            selected_metrics.append("response_relevancy")
        if use_answer_faithfulness:
            selected_metrics.append("answer_faithfulness")
        
        if not selected_metrics:
            st.warning("Please select at least one metric for evaluation.")
        
        st.markdown("---")
        
        # Helpful information
        with st.expander("What is RAGAS?"):
            st.write("""
            RAGAS is a framework for evaluating Retrieval Augmented Generation (RAG) systems. 
            It provides several metrics to assess the quality of a RAG system:
            
            - **Context Recall**: Measures if the generated answer captures all key information from the context.
            - **Context Precision**: Evaluates if the context was precise and useful for answering the question.
            - **Response Relevancy**: Evaluates if the answer addresses the question.
            - **Answer Faithfulness**: Measures if the generated answer is factually consistent with the retrieved context.
            
            Higher scores (closer to 1.0) indicate better performance.
            """)
        
        # Load example data button
        if st.button("Load Example Data"):
            example_data = load_example_data()
            st.session_state.questions = example_data["questions"]
            st.session_state.answers = example_data["answers"]
            st.session_state.contexts = example_data["contexts"]
            st.session_state.ground_truths = example_data["ground_truths"]
            st.session_state.reference_contexts = example_data["reference_contexts"]
            st.success("Example data loaded!")
            st.rerun()
    
    # Main content area with tabs
    tabs = st.tabs(["Data Input", "Evaluation Results", "Detailed Analysis"])
    
    # Data Input Tab
    with tabs[0]:
        st.markdown('<div class="section-header"><h2>Input Evaluation Data</h2></div>', unsafe_allow_html=True)
        
        # Instruction for the user
        st.info("""
        To evaluate your chatbot, you need to provide:
        1. Questions that were asked
        2. Answers provided by the chatbot
        3. Context/documents that were retrieved
        4. Reference contexts (ideal contexts that should have been retrieved)
        5. Ground truth answers (what the correct answer should be)
        
        You can add multiple QA pairs for a more comprehensive evaluation.
        """)
        
        # Option to upload JSON file with evaluation data
        uploaded_file = st.file_uploader("Upload evaluation data (JSON format)", type=["json"])
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                if all(k in data for k in ["questions", "answers", "contexts"]):
                    st.session_state.questions = data["questions"]
                    st.session_state.answers = data["answers"]
                    st.session_state.contexts = data["contexts"]
                    st.session_state.ground_truths = data.get("ground_truths", [])
                    st.session_state.reference_contexts = data.get("reference_contexts", [])
                    st.success(f"Loaded {len(data['questions'])} evaluation samples from file!")
                else:
                    st.error("Invalid JSON format. File must contain at least 'questions', 'answers', and 'contexts' keys.")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        st.markdown("---")
        
        # Manual data entry section
        st.subheader("Manual Data Entry")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            # Add button
            if st.button("Add New Question"):
                st.session_state.questions.append("")
                st.session_state.answers.append("")
                st.session_state.contexts.append([])
                st.session_state.reference_contexts.append([])
                st.session_state.ground_truths.append("")
                st.session_state.current_question_index = len(st.session_state.questions) - 1
        
        with col2:
            # Clear button
            if st.button("Clear All Data"):
                st.session_state.questions = []
                st.session_state.answers = []
                st.session_state.contexts = []
                st.session_state.reference_contexts = []
                st.session_state.ground_truths = []
                st.session_state.current_question_index = 0
                st.success("All data cleared!")
        
        with col3:
            # Download current data as JSON
            if st.session_state.questions and st.button("Download Data"):
                data = {
                    "questions": st.session_state.questions,
                    "answers": st.session_state.answers,
                    "contexts": st.session_state.contexts,
                    "reference_contexts": st.session_state.reference_contexts,
                    "ground_truths": st.session_state.ground_truths
                }
                json_str = json.dumps(data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="evaluation_data.json",
                    mime="application/json"
                )
        
        # Display number of questions
        st.write(f"Number of questions: {len(st.session_state.questions)}")
        
        # Navigation between questions if we have any
        if st.session_state.questions:
            # Select which question to edit
            st.session_state.current_question_index = st.selectbox(
                "Select question to edit:",
                range(len(st.session_state.questions)),
                format_func=lambda i: f"Question {i+1}: {st.session_state.questions[i][:50]}...",
                index=st.session_state.current_question_index
            )
            
            idx = st.session_state.current_question_index
            
            # Question input
            st.markdown('<div class="question-box">', unsafe_allow_html=True)
            st.subheader("Question")
            question = st.text_area("Enter question:", 
                                   value=st.session_state.questions[idx] if idx < len(st.session_state.questions) else "", 
                                   height=80,
                                   key=f"question_{idx}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if question and idx < len(st.session_state.questions):
                st.session_state.questions[idx] = question
            
            # Answer input
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.subheader("Chatbot Answer")
            answer = st.text_area("Enter the chatbot's answer:", 
                                 value=st.session_state.answers[idx] if idx < len(st.session_state.answers) else "", 
                                 height=150,
                                 key=f"answer_{idx}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if answer and idx < len(st.session_state.answers):
                st.session_state.answers[idx] = answer
            
            # Ground truth input
            st.markdown('<div class="ground-truth-box">', unsafe_allow_html=True)
            st.subheader("Ground Truth Answer")
            ground_truth = st.text_area("Enter the correct ground truth answer:", 
                                       value=st.session_state.ground_truths[idx] if idx < len(st.session_state.ground_truths) and st.session_state.ground_truths[idx] else "", 
                                       height=150,
                                       key=f"ground_truth_{idx}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if ground_truth and idx < len(st.session_state.ground_truths):
                st.session_state.ground_truths[idx] = ground_truth
            
            # Context input
            st.markdown('<div class="context-box">', unsafe_allow_html=True)
            st.subheader("Retrieved Contexts")
            
            # Initialize contexts for this question if needed
            if idx >= len(st.session_state.contexts) or not st.session_state.contexts[idx]:
                st.session_state.contexts[idx] = [""]
            
            for i, ctx in enumerate(st.session_state.contexts[idx]):
                col1, col2 = st.columns([10, 1])
                with col1:
                    context = st.text_area(f"Retrieved Context {i+1}:", 
                                          value=ctx, 
                                          height=100,
                                          key=f"context_{idx}_{i}")
                    
                    # Update context in session state
                    if context and i < len(st.session_state.contexts[idx]):
                        st.session_state.contexts[idx][i] = context
                
                with col2:
                    if st.button("🗑️", key=f"delete_ctx_{idx}_{i}") and len(st.session_state.contexts[idx]) > 1:
                        st.session_state.contexts[idx].pop(i)
                        st.rerun()
            
            if st.button("Add Retrieved Context", key=f"add_context_{idx}"):
                st.session_state.contexts[idx].append("")
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Reference Context input
            st.markdown('<div class="context-box">', unsafe_allow_html=True)
            st.subheader("Reference Contexts (Optional)")
            st.caption("These are the ideal contexts that should have been retrieved. If not provided, the retrieved contexts will be used.")
            
            # Initialize reference contexts for this question if needed
            if idx >= len(st.session_state.reference_contexts) or not st.session_state.reference_contexts[idx]:
                st.session_state.reference_contexts[idx] = [""]
            
            for i, ctx in enumerate(st.session_state.reference_contexts[idx]):
                col1, col2 = st.columns([10, 1])
                with col1:
                    ref_context = st.text_area(f"Reference Context {i+1}:", 
                                              value=ctx, 
                                              height=100,
                                              key=f"ref_context_{idx}_{i}")
                    
                    # Update reference context in session state
                    if ref_context and i < len(st.session_state.reference_contexts[idx]):
                        st.session_state.reference_contexts[idx][i] = ref_context
                
                with col2:
                    if st.button("🗑️", key=f"delete_ref_ctx_{idx}_{i}") and len(st.session_state.reference_contexts[idx]) > 1:
                        st.session_state.reference_contexts[idx].pop(i)
                        st.rerun()
            
            if st.button("Add Reference Context", key=f"add_ref_context_{idx}"):
                st.session_state.reference_contexts[idx].append("")
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Delete current question button
            if st.button("Delete This Question", key=f"delete_question_{idx}"):
                if len(st.session_state.questions) > 0:
                    st.session_state.questions.pop(idx)
                    st.session_state.answers.pop(idx) if idx < len(st.session_state.answers) else None
                    st.session_state.contexts.pop(idx) if idx < len(st.session_state.contexts) else None
                    st.session_state.reference_contexts.pop(idx) if idx < len(st.session_state.reference_contexts) else None
                    st.session_state.ground_truths.pop(idx) if idx < len(st.session_state.ground_truths) else None
                    
                    if st.session_state.current_question_index >= len(st.session_state.questions) and st.session_state.current_question_index > 0:
                        st.session_state.current_question_index = len(st.session_state.questions) - 1
                    
                    st.success("Question deleted!")
                    st.rerun()
        
        # Run evaluation button at the bottom
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🚀 Run Evaluation", disabled=not (st.session_state.questions and openai.api_key and selected_metrics)):
                # Validate data
                for i, (q, a, c) in enumerate(zip(
                    st.session_state.questions, 
                    st.session_state.answers, 
                    st.session_state.contexts
                )):
                    if not q or not a or not c:
                        st.error(f"Question {i+1} has missing data. Please complete all required fields.")
                        st.stop()
                
                # Ensure reference_contexts has same length as questions
                while len(st.session_state.reference_contexts) < len(st.session_state.questions):
                    st.session_state.reference_contexts.append([])
                
                # Ensure ground_truths has same length as questions
                while len(st.session_state.ground_truths) < len(st.session_state.questions):
                    st.session_state.ground_truths.append("")
                
                # Run the evaluation
                with st.spinner("Running evaluation... This may take a few minutes."):
                    try:
                        # Run asyncio evaluation
                        results, details = asyncio.run(evaluate_samples(
                            st.session_state.questions,
                            st.session_state.answers,
                            st.session_state.contexts,
                            st.session_state.reference_contexts,
                            st.session_state.ground_truths,
                            selected_metrics,
                            openai.api_key,
                            model_name
                        ))
                        
                        # Store results in session
                        st.session_state.evaluation_results = {
                            "metrics": results,
                            "details": details
                        }
                        
                        st.success("Evaluation completed successfully!")
                        # Switch to results tab
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error running evaluation: {str(e)}")
                        st.error(traceback.format_exc())
        
        with col2:
            st.markdown("""
            <div style='background-color:#e8f0fe; padding:10px; border-radius:5px;'>
                <b>Note:</b> Evaluation requires an OpenAI API key and will use tokens.
            </div>
            """, unsafe_allow_html=True)
    
    # Evaluation Results Tab
    with tabs[1]:
        st.markdown('<div class="section-header"><h2>Evaluation Results</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.evaluation_results is not None:
            results = st.session_state.evaluation_results["metrics"]
            
            # Calculate average scores
            avg_scores = {}
            for metric_name, scores in results.items():
                if scores:  # Check if not empty
                    avg_scores[metric_name] = sum(scores) / len(scores)
            
            # Overall scores
            st.subheader("Overall Scores")
            
            # Display metric cards in columns
            col_count = min(len(avg_scores), 3)  # Maximum 3 columns per row
            rows = (len(avg_scores) + col_count - 1) // col_count  # Ceiling division
            
            for row in range(rows):
                cols = st.columns(col_count)
                for i in range(col_count):
                    idx = row * col_count + i
                    if idx < len(avg_scores):
                        metric_name = list(avg_scores.keys())[idx]
                        score = avg_scores[metric_name]
                        
                        # Format metric name for display
                        display_name = metric_name.replace("_", " ").title()
                        
                        with cols[i]:
                            # Determine color based on score
                            if score >= 0.8:
                                color = "#34a853"  # Green
                            elif score >= 0.6:
                                color = "#fbbc04"  # Yellow
                            else:
                                color = "#ea4335"  # Red
                                
                            st.markdown(f"""
                            <div class="metric-card" style="border-top: 3px solid {color};">
                                <h3>{display_name}</h3>
                                <div class="score-number" style="color: {color};">{score:.3f}</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Visualization
            st.subheader("Visualizations")
            
            # Create dataframe for plotting
            results_df = pd.DataFrame([avg_scores])
            
            # Plot bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            metric_names = [name.replace("_", " ").title() for name in avg_scores.keys()]
            values = list(avg_scores.values())
            
            # Sort by score
            sorted_indices = np.argsort(values)[::-1]
            sorted_names = [metric_names[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]
            
            # Create horizontal bar chart
            bars = ax.barh(sorted_names, sorted_values, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted_values))))
            
            # Add text labels
            for i, v in enumerate(sorted_values):
                ax.text(v + 0.01, i, f"{v:.3f}", va='center')
            
            # Customize chart
            ax.set_xlabel('Score (0-1)')
            ax.set_title('Metric Scores')
            ax.set_xlim(0, 1.1)  # Set x-axis limit
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Display chart
            st.pyplot(fig)
            
            # Display radar chart if we have enough metrics
            if len(avg_scores) >= 3:
                st.subheader("Radar Chart")
                
                # Create radar chart
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, polar=True)
                
                # Prepare data for radar chart
                categories = metric_names
                values = list(avg_scores.values())
                
                # Number of variables
                N = len(categories)
                
                # Compute angle for each axis
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Add values (and close the loop)
                values += values[:1]
                
                # Draw the chart
                ax.plot(angles, values, linewidth=2, linestyle='solid')
                ax.fill(angles, values, alpha=0.25)
                
                # Set category labels
                plt.xticks(angles[:-1], categories, size=12)
                
                # Draw y-axis labels (0 to 1)
                ax.set_rlabel_position(0)
                plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
                plt.ylim(0, 1)
                
                # Add title
                plt.title("Metrics Radar Chart", size=15, y=1.1)
                
                # Display the chart
                st.pyplot(fig)
            
            # Per-question scores
            st.subheader("Per-Question Scores")
            
            # Create a dataframe with scores for each question
            question_scores = []
            for i, question in enumerate(st.session_state.questions):
                row = {"Question": f"Q{i+1}: {question[:50]}..."}
                for metric_name, scores in results.items():
                    if i < len(scores):
                        row[metric_name.replace("_", " ").title()] = scores[i]
                question_scores.append(row)
            
            question_df = pd.DataFrame(question_scores)
            
            # Style the dataframe
            def highlight_cells(val):
                if isinstance(val, float):
                    if val >= 0.8:
                        return 'background-color: #e6f4ea'  # Light green
                    elif val >= 0.6:
                        return 'background-color: #fff8e1'  # Light yellow
                    else:
                        return 'background-color: #fce8e6'  # Light red
                return ''
            
            styled_df = question_df.style.applymap(highlight_cells)
            st.dataframe(styled_df, use_container_width=True)
            
            # Download results
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                # Download as CSV
                csv = question_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="evaluation_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download as JSON
                results_json = {
                    "average_scores": avg_scores,
                    "per_question_scores": results,
                    "questions": st.session_state.questions,
                    "answers": st.session_state.answers,
                    "evaluation_timestamp": datetime.now().isoformat()
                }
                json_str = json.dumps(results_json, indent=2)
                st.download_button(
                    label="Download Results as JSON",
                    data=json_str,
                    file_name="evaluation_results.json",
                    mime="application/json"
                )
        else:
            st.info("No evaluation results yet. Please run an evaluation from the Data Input tab.")
    
    # Detailed Analysis Tab
    with tabs[2]:
        st.markdown('<div class="section-header"><h2>Detailed Analysis</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.evaluation_results is not None:
            # Get details from evaluation results
            details = st.session_state.evaluation_results["details"]
            
            # Select a question to analyze
            selected_q_idx = st.selectbox(
                "Select question to analyze:",
                range(len(details)),
                format_func=lambda i: f"Question {i+1}: {details[i]['question'][:50]}..."
            )
            
            # Get the selected question details
            q_detail = details[selected_q_idx]
            
            # Display question details
            st.markdown("---")
            
            # Question
            st.markdown('<div class="question-box">', unsafe_allow_html=True)
            st.subheader("Question")
            st.write(q_detail["question"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Side by side comparison
            col1, col2 = st.columns(2)
            
            with col1:
                # Generated answer
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.subheader("Chatbot Answer")
                st.write(q_detail["answer"])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Ground truth
                st.markdown('<div class="ground-truth-box">', unsafe_allow_html=True)
                st.subheader("Ground Truth")
                if q_detail["ground_truth"]:
                    st.write(q_detail["ground_truth"])
                else:
                    st.write("No ground truth provided")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Context used
            st.markdown('<div class="context-box">', unsafe_allow_html=True)
            st.subheader("Context Used")
            st.write(format_context_for_display(q_detail["context"]))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Scores for this question
            st.subheader("Metric Scores")
            
            q_scores = q_detail["scores"]
            metric_cols = st.columns(min(3, len(q_scores)))
            
            i = 0
            for metric_name, score in q_scores.items():
                with metric_cols[i % len(metric_cols)]:
                    # Determine color based on score
                    if score >= 0.8:
                        color = "#34a853"  # Green
                        performance = "Good"
                    elif score >= 0.6:
                        color = "#fbbc04"  # Yellow
                        performance = "Average"
                    else:
                        color = "#ea4335"  # Red
                        performance = "Poor"
                    
                    # Display score card
                    st.markdown(f"""
                    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid {color};">
                        <h4 style="margin-top: 0;">{metric_name.replace('_', ' ').title()}</h4>
                        <div style="font-size: 24px; font-weight: bold; color: {color};">{score:.3f}</div>
                        <div>Performance: {performance}</div>
                    </div>
                    """, unsafe_allow_html=True)
                i += 1
            
            # Analysis and recommendations
            st.markdown("---")
            st.subheader("Analysis & Recommendations")
            
            # Check for low scores and provide recommendations
            low_scores = {k: v for k, v in q_scores.items() if v < 0.7}
            
            if low_scores:
                st.warning("Areas for improvement:")
                
                for metric_name, score in low_scores.items():
                    improvement_suggestions = {
                        "context_recall": "The answer is missing important information from the context. Review the RAG retrieval to ensure all relevant information is captured.",
                        "context_recall_non_llm": "The retrieved context is missing key information that was in the reference context. Improve your retrieval system to fetch more comprehensive information.",
                        "context_precision": "The retrieved context contains too much irrelevant information. Refine your retrieval system to be more precise and focused.",
                        "context_precision_with_ref": "The retrieved context isn't as precise as it could be compared to the reference. Work on improving the relevance of retrieved documents.",
                        "response_relevancy": "The answer doesn't fully address the question. Train the model to focus more on answering exactly what was asked.",
                        "answer_faithfulness": "The answer contains information not supported by the context. Ensure the model doesn't add unsupported facts or speculations."
                    }
                    
                    display_name = metric_name.replace("_", " ").title()
                    suggestion = improvement_suggestions.get(metric_name, "Improve this aspect of your system.")
                    
                    st.markdown(f"""
                    <div style="background-color:#fef7e5; padding:15px; border-radius:5px; margin-bottom:15px; border-left:3px solid #fbbc04;">
                        <strong>{display_name}</strong> ({score:.2f}): {suggestion}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("This response performs well across all metrics! Keep up the good work.")
            
            # Generate detailed analysis with OpenAI
            if st.button("Generate In-depth Analysis (Uses API)", key=f"detailed_analysis_{selected_q_idx}"):
                try:
                    with st.spinner("Generating analysis..."):
                        # Format the data for the API call
                        scores_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v:.3f}" for k, v in q_scores.items()])
                        
                        analysis_prompt = f"""
                        Analyze the following question-answer pair from a retrieval-augmented chatbot:
                        
                        Question: {q_detail['question']}
                        
                        Chatbot Answer: {q_detail['answer']}
                        
                        Ground Truth: {q_detail['ground_truth'] if q_detail['ground_truth'] else 'Not provided'}
                        
                        Retrieved Context:
                        {format_context_for_display(q_detail['context'])}
                        
                        Evaluation Metrics:
                        {scores_text}
                        
                        Please provide:
                        1. A detailed analysis of why this answer received these scores
                        2. Specific strengths of the answer and retrieval
                        3. Concrete areas for improvement
                        4. Actionable suggestions to improve both the retrieval system and answer generation
                        """
                        
                        response = openai.ChatCompletion.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are an expert in evaluating retrieval-augmented chatbots. Provide clear, specific analysis and actionable suggestions."},
                                {"role": "user", "content": analysis_prompt}
                            ],
                            max_tokens=1000,
                            temperature=0.0
                        )
                        
                        analysis = response["choices"][0]["message"]["content"]
                        
                        st.markdown(f"""
                        <div style="background-color:#f1f8e9; padding:20px; border-radius:5px; border:1px solid #dcedc8;">
                            <h4>Expert Analysis</h4>
                            {analysis}
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating analysis: {str(e)}")
                    st.error(traceback.format_exc())
        else:
            st.info("No evaluation results yet. Please run an evaluation from the Data Input tab.")
    
    # Footer
    st.markdown("---")
    st.caption("RAGAS Evaluator for CivicCatalyst Chatbot | Developed with ❤️ using Streamlit and RAGAS")

if __name__ == "__main__":
    main()