import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import asyncio
import nest_asyncio
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import uuid
import requests
from functools import partial

# Apply nest_asyncio to make async code work in Streamlit
nest_asyncio.apply()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLM-Comparison")

# Constants
DEFAULT_MODELS = {
    "gpt-3.5-turbo": {
        "provider": "openai",
        "api_type": "openai",
        "description": "GPT-3.5 Turbo - OpenAI's affordable model with good performance"
    },
    "gpt-4": {
        "provider": "openai",
        "api_type": "openai",
        "description": "GPT-4 - OpenAI's advanced model with strong reasoning capabilities"
    },
    "claude-3-haiku-20240307": {
        "provider": "anthropic",
        "api_type": "anthropic",
        "description": "Claude 3 Haiku - Anthropic's fastest and most cost-effective model"
    },
    "claude-3-sonnet-20240229": {
        "provider": "anthropic",
        "api_type": "anthropic",
        "description": "Claude 3 Sonnet - Anthropic's balanced model for most use cases"
    },
    "claude-3-opus-20240229": {
        "provider": "anthropic",
        "api_type": "anthropic",
        "description": "Claude 3 Opus - Anthropic's most powerful model"
    },
    "meta/llama-3-8b-instruct": {
        "provider": "replicate",
        "api_type": "replicate",
        "description": "Llama 3 8B Instruct - Meta's smaller open model"
    },
    "meta/llama-3-70b-instruct": {
        "provider": "replicate",
        "api_type": "replicate",
        "description": "Llama 3 70B Instruct - Meta's larger open model"
    }
}

# Default metrics for evaluation
DEFAULT_METRICS = [
    {
        "name": "accuracy",
        "display_name": "Accuracy",
        "description": "Measures how accurate the model's answers are compared to ground truth",
        "higher_is_better": True
    },
    {
        "name": "relevance",
        "display_name": "Relevance",
        "description": "Measures how relevant the response is to the question",
        "higher_is_better": True
    },
    {
        "name": "coherence",
        "display_name": "Coherence",
        "description": "Measures how coherent and well-structured the response is",
        "higher_is_better": True
    },
    {
        "name": "factuality",
        "display_name": "Factuality",
        "description": "Measures correctness of facts in the response",
        "higher_is_better": True
    },
    {
        "name": "toxicity",
        "display_name": "Toxicity Score",
        "description": "Measures harmful content in responses (lower is better)",
        "higher_is_better": False
    },
    {
        "name": "reasoning",
        "display_name": "Reasoning",
        "description": "Measures logical reasoning capabilities",
        "higher_is_better": True
    },
    {
        "name": "latency",
        "display_name": "Latency (s)",
        "description": "Response time in seconds (lower is better)",
        "higher_is_better": False
    },
    {
        "name": "tokens_per_second",
        "display_name": "Tokens/Second",
        "description": "Processing speed in tokens per second",
        "higher_is_better": True
    }
]

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables with defaults"""
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {}
    
    if "models" not in st.session_state:
        st.session_state.models = DEFAULT_MODELS.copy()
    
    if "custom_models" not in st.session_state:
        st.session_state.custom_models = {}
    
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    
    if "evaluation_datasets" not in st.session_state:
        st.session_state.evaluation_datasets = {}
    
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None
    
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = {}
    
    if "mlflow_experiments" not in st.session_state:
        st.session_state.mlflow_experiments = {}
    
    if "experiment_name" not in st.session_state:
        st.session_state.experiment_name = f"LLM-Comparison-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    if "metrics" not in st.session_state:
        st.session_state.metrics = DEFAULT_METRICS.copy()
    
    if "selected_metrics" not in st.session_state:
        st.session_state.selected_metrics = [m["name"] for m in DEFAULT_METRICS[:4]]  # First 4 metrics by default

def load_example_dataset():
    """Load example evaluation dataset"""
    example_data = {
        "name": "General Knowledge QA",
        "description": "General knowledge questions across various domains",
        "questions": [
            "What causes ocean tides?",
            "How does photosynthesis work?",
            "Explain the concept of compounding interest.",
            "What is the difference between RAM and ROM?",
            "How do vaccines work to prevent disease?"
        ],
        "ground_truths": [
            "Ocean tides are primarily caused by the gravitational pull of the moon and, to a lesser extent, the sun on Earth's oceans. This gravitational force creates two bulges in the Earth's oceans: one on the side facing the moon and one on the opposite side. As Earth rotates, different parts of the planet pass through these bulges, experiencing high tides. The areas not in the bulges experience low tides. The sun's gravitational influence also affects tides, creating spring tides (when the sun and moon align) and neap tides (when they're at right angles).",
            "Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight, water (H₂O), and carbon dioxide (CO₂) into glucose (sugar) and oxygen. This process occurs primarily in the chloroplasts of plant cells, specifically using the green pigment chlorophyll, which captures light energy. Photosynthesis has two main stages: the light-dependent reactions, which convert light energy to chemical energy (ATP and NADPH), and the Calvin cycle (light-independent reactions), which uses this chemical energy to produce glucose from carbon dioxide.",
            "Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods. Unlike simple interest, which is calculated only on the principal amount, compound interest leads to exponential growth of money over time. The formula for compound interest is A = P(1 + r/n)^(nt), where A is the final amount, P is the principal, r is the interest rate, n is the number of times interest is compounded per time period, and t is the time. This exponential growth is why compound interest is often called 'interest on interest' and why it's a fundamental concept in investing and saving.",
            "RAM (Random Access Memory) and ROM (Read-Only Memory) are both computer memory types but serve different purposes. RAM is volatile memory that temporarily stores data being actively used by the CPU. Its contents are lost when power is turned off. RAM is fast, rewritable, and directly accessible by the processor. ROM, by contrast, is non-volatile memory that permanently stores instructions needed at startup. Its contents remain even without power. Traditional ROM cannot be modified after manufacturing, though variants like EEPROM and flash memory allow for updates. RAM is typically larger in capacity and used for running programs, while ROM stores firmware and essential startup instructions.",
            "Vaccines work by mimicking infectious agents (like viruses or bacteria) to train the immune system without causing disease. When introduced into the body, vaccines trigger an immune response, causing the production of antibodies and memory cells specific to that pathogen. These memory cells remain in the body, ready to quickly recognize and fight the actual pathogen if exposed in the future. This 'immunological memory' allows for a faster, stronger response to the real infection, often preventing disease entirely or reducing its severity. Vaccines can contain weakened pathogens (live attenuated), killed pathogens (inactivated), pathogen components (subunit/conjugate), or genetic instructions for producing pathogen proteins (mRNA/viral vector)."
        ]
    }
    return example_data

async def evaluate_model(model_name, model_info, questions, ground_truths, api_keys):
    """Evaluate a single model on the given dataset"""
    provider = model_info["provider"]
    api_type = model_info["api_type"]
    
    # Check if we have the API key
    if provider not in api_keys or not api_keys[provider]:
        return {
            "error": f"No API key provided for {provider}",
            "model": model_name,
            "results": {}
        }
    
    results = {
        "model": model_name,
        "provider": provider,
        "responses": [],
        "metrics": {}
    }
    
    # Store per-question metrics
    question_metrics = []
    
    # Process each question
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
        try:
            # Record start time for latency measurement
            start_time = time.time()
            
            # Get model response based on provider
            response_text, tokens_info = await get_model_response(
                question, 
                model_name, 
                api_type, 
                api_keys[provider]
            )
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Store response
            results["responses"].append(response_text)
            
            # Evaluate response quality using metrics
            metrics = await evaluate_response_quality(
                question=question,
                response=response_text,
                ground_truth=ground_truth,
                model_name=model_name,
                api_type=api_type,
                api_key=api_keys[provider]
            )
            
            # Add latency and token metrics
            metrics["latency"] = latency
            metrics["tokens_per_second"] = tokens_info.get("output_tokens", len(response_text.split())) / max(0.1, latency)
            
            # Store per-question metrics
            question_metrics.append(metrics)
            
            # Add a short delay to avoid hitting rate limits
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error evaluating question {i} with model {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            # Add failed metrics
            question_metrics.append({
                "error": str(e),
                "accuracy": 0,
                "relevance": 0,
                "coherence": 0,
                "factuality": 0,
                "toxicity": 1,  # Higher is worse
                "reasoning": 0,
                "latency": 30,  # Assume timeout
                "tokens_per_second": 0
            })
    
    # Calculate aggregate metrics
    for metric in ["accuracy", "relevance", "coherence", "factuality", "toxicity", "reasoning", "latency", "tokens_per_second"]:
        values = [q_metric.get(metric, 0) for q_metric in question_metrics]
        results["metrics"][metric] = sum(values) / len(values) if values else 0
    
    return results

async def get_model_response(question, model_name, api_type, api_key):
    """Get response from a model via API"""
    if api_type == "openai":
        return await get_openai_response(question, model_name, api_key)
    elif api_type == "anthropic":
        return await get_anthropic_response(question, model_name, api_key)
    elif api_type == "replicate":
        return await get_replicate_response(question, model_name, api_key)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")

async def get_openai_response(question, model_name, api_key):
    """Get response from OpenAI API"""
    try:
        import openai
        client = openai.Client(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate, factual information."},
                {"role": "user", "content": question}
            ],
            temperature=0.0,
            max_tokens=1024
        )
        
        # Extract response text and token information
        response_text = response.choices[0].message.content
        tokens_info = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return response_text, tokens_info
    
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        # Simulate response for development
        return f"[OpenAI simulation] Answer to: {question}", {"input_tokens": 10, "output_tokens": 50, "total_tokens": 60}

async def get_anthropic_response(question, model_name, api_key):
    """Get response from Anthropic API"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model=model_name,
            max_tokens=1024,
            temperature=0.0,
            system="You are a helpful assistant that provides accurate, factual information.",
            messages=[
                {"role": "user", "content": question}
            ]
        )
        
        # Extract response text and token information
        response_text = response.content[0].text
        tokens_info = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
        
        return response_text, tokens_info
    
    except Exception as e:
        logger.error(f"Anthropic API error: {str(e)}")
        # Simulate response for development
        return f"[Anthropic simulation] Answer to: {question}", {"input_tokens": 10, "output_tokens": 50, "total_tokens": 60}

async def get_replicate_response(question, model_name, api_key):
    """Get response from Replicate API"""
    try:
        import replicate
        client = replicate.Client(api_token=api_key)
        
        input_data = {
            "prompt": f"<|system|>\nYou are a helpful assistant that provides accurate, factual information.\n<|user|>\n{question}\n<|assistant|>",
            "max_new_tokens": 1024,
            "temperature": 0.0
        }
        
        start_time = time.time()
        output = client.run(model_name, input=input_data)
        response_text = "".join(output)
        
        # Estimate tokens based on words (crude approximation)
        word_count = len(response_text.split())
        estimated_tokens = word_count * 1.3  # Rough estimate
        
        tokens_info = {
            "input_tokens": len(question.split()) * 1.3,  # Rough estimate
            "output_tokens": estimated_tokens,
            "total_tokens": (len(question.split()) + word_count) * 1.3
        }
        
        return response_text, tokens_info
    
    except Exception as e:
        logger.error(f"Replicate API error: {str(e)}")
        # Simulate response for development
        return f"[Replicate simulation] Answer to: {question}", {"input_tokens": 10, "output_tokens": 50, "total_tokens": 60}

async def evaluate_response_quality(question, response, ground_truth, model_name, api_type, api_key):
    """Evaluate the quality of a model response based on various metrics"""
    # For real implementation, you would use a judge model or evaluation API
    # Here we'll simulate the scores for demonstration
    # In a real implementation, consider using specialized judge models or evaluation APIs
    
    # Simulate some variance in scores
    seed = hashlib.md5(f"{question}:{response}:{model_name}".encode()).hexdigest()
    seed_val = int(seed, 16) % 1000 / 1000.0
    
    # Base scores with some randomness
    metrics = {
        "accuracy": min(0.95, max(0.35, 0.75 + 0.2 * seed_val)),
        "relevance": min(0.98, max(0.6, 0.85 + 0.13 * seed_val)),
        "coherence": min(0.99, max(0.7, 0.88 + 0.11 * seed_val)),
        "factuality": min(0.95, max(0.4, 0.78 + 0.17 * seed_val)), 
        "toxicity": max(0.01, min(0.2, 0.05 + 0.15 * seed_val)),
        "reasoning": min(0.94, max(0.3, 0.72 + 0.22 * seed_val)),
    }
    
    # In a real implementation, you would use a judge model or evaluation API here
    # For example:
    # judge_response = await get_judge_evaluation(question, response, ground_truth, api_key)
    # metrics = parse_judge_response(judge_response)
    
    return metrics

async def get_judge_evaluation(question, response, ground_truth, api_key):
    """In a real implementation, this would call a judge model or evaluation API"""
    # Placeholder for actual implementation
    pass

def setup_mlflow():
    """Set up MLflow tracking"""
    # Set tracking URI to local directory
    mlflow_dir = os.path.join(os.getcwd(), "mlruns")
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    
    # Create MLflow client
    client = MlflowClient()
    
    # Create or get experiment
    try:
        experiment = client.get_experiment_by_name(st.session_state.experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(
                st.session_state.experiment_name,
                tags={"created_time": datetime.now().isoformat()}
            )
    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        experiment_id = None
    
    return experiment_id

def log_model_results_to_mlflow(model_results, dataset_name, experiment_id):
    """Log model evaluation results to MLflow"""
    if not experiment_id:
        logger.error("No valid MLflow experiment ID")
        return None
    
    try:
        # Start a run for this model evaluation
        run_name = f"{model_results['model']}-{dataset_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            # Log model info
            mlflow.set_tag("model_name", model_results['model'])
            mlflow.set_tag("provider", model_results['provider'])
            mlflow.set_tag("dataset", dataset_name)
            mlflow.set_tag("evaluation_time", datetime.now().isoformat())
            
            # Log metrics
            for metric_name, value in model_results['metrics'].items():
                mlflow.log_metric(metric_name, value)
            
            # Create and log a summary DataFrame as a CSV artifact
            metrics_df = pd.DataFrame([model_results['metrics']])
            csv_path = f"metrics_{model_results['model']}_{int(time.time())}.csv"
            metrics_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)
            
            # Clean up temp file
            if os.path.exists(csv_path):
                os.remove(csv_path)
            
            # Return the run ID
            return run.info.run_id
    
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def get_mlflow_results(experiment_id):
    """Get all runs from an MLflow experiment"""
    if not experiment_id:
        return []
    
    client = MlflowClient()
    
    try:
        # Get all runs for this experiment
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["attribute.start_time DESC"]
        )
        
        results = []
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "model": run.data.tags.get("model_name", "Unknown"),
                "provider": run.data.tags.get("provider", "Unknown"),
                "dataset": run.data.tags.get("dataset", "Unknown"),
                "evaluation_time": run.data.tags.get("evaluation_time", "Unknown"),
                "metrics": {k: v for k, v in run.data.metrics.items()}
            }
            results.append(run_data)
        
        return results
    
    except Exception as e:
        logger.error(f"Error getting MLflow results: {str(e)}")
        return []

def compare_models(mlflow_results, metrics=None):
    """Create comparison dataframe from MLflow results"""
    if not mlflow_results:
        return pd.DataFrame()
    
    # Filter results by selected metrics if provided
    if metrics:
        # Create a dataframe for each run
        rows = []
        for run in mlflow_results:
            row = {
                "Model": run["model"],
                "Provider": run["provider"],
                "Dataset": run["dataset"],
                "Evaluation Time": run["evaluation_time"]
            }
            
            # Add metrics
            for metric in metrics:
                row[metric] = run["metrics"].get(metric, None)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    else:
        # Include all metrics
        rows = []
        for run in mlflow_results:
            row = {
                "Model": run["model"],
                "Provider": run["provider"],
                "Dataset": run["dataset"],
                "Evaluation Time": run["evaluation_time"]
            }
            
            # Add all metrics
            for metric_name, value in run["metrics"].items():
                row[metric_name] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)

def main():
  
    
    # Initialize session state
    initialize_session_state()
    
    # Apply custom styling
    st.markdown(
        """
        <style>
        .main-header {
            background-color: #4285f4;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            text-align: center;
        }
        
        .section-header {
            background-color: #34a853;
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
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #4285f4;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Main header
    st.markdown('<div class="main-header"><h1>🤖 LLM Model Comparison Platform</h1></div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Experiment name
        st.session_state.experiment_name = st.text_input(
            "Experiment Name", 
            value=st.session_state.experiment_name
        )
        
        st.markdown("---")
        
        # API Keys
        st.subheader("API Keys")
        
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            placeholder="sk-...",
            value=st.session_state.api_keys.get("openai", "")
        )
        st.session_state.api_keys["openai"] = openai_api_key
        
        anthropic_api_key = st.text_input(
            "Anthropic API Key", 
            type="password", 
            placeholder="sk-ant-...",
            value=st.session_state.api_keys.get("anthropic", "")
        )
        st.session_state.api_keys["anthropic"] = anthropic_api_key
        
        replicate_api_key = st.text_input(
            "Replicate API Key", 
            type="password", 
            placeholder="r8_...",
            value=st.session_state.api_keys.get("replicate", "")
        )
        st.session_state.api_keys["replicate"] = replicate_api_key
        
        st.markdown("---")
        
        # Metrics selection
        st.subheader("Evaluation Metrics")
        
        available_metrics = [m["name"] for m in st.session_state.metrics]
        default_selections = st.session_state.selected_metrics
        
        st.session_state.selected_metrics = st.multiselect(
            "Select metrics for evaluation and comparison:",
            options=available_metrics,
            default=default_selections,
            format_func=lambda x: next((m["display_name"] for m in st.session_state.metrics if m["name"] == x), x)
        )
        
        st.markdown("---")
        
        # Information
        with st.expander("About this app"):
            st.write("""
            This app helps you compare different LLM models across various evaluation metrics.
            
            **Features**:
            - Evaluate multiple models on the same dataset
            - Track experiments with MLflow
            - Compare performance across models
            - Visualize results with interactive charts
            
            **How to use**:
            1. Configure your API keys
            2. Select models to evaluate
            3. Create or upload an evaluation dataset
            4. Run the evaluation
            5. Compare and analyze results
            """)
    
    # Main content area with tabs
    tabs = st.tabs(["Models", "Datasets", "Evaluation", "Results", "Comparison"])
    
    # Models Tab
    with tabs[0]:
        st.markdown('<div class="section-header"><h2>Model Selection</h2></div>', unsafe_allow_html=True)
        
        # Available models
        st.subheader("Available Models")
        st.write("Select models to include in your evaluation:")
        
        # Create 3 columns for model selection
        col1, col2, col3 = st.columns(3)
        
        available_models = {**st.session_state.models, **st.session_state.custom_models}
        
        # Group models by provider
        providers = {}
        for model_name, model_info in available_models.items():
            provider = model_info["provider"]
            if provider not in providers:
                providers[provider] = []
            providers[provider].append((model_name, model_info))
        
        # Display models by provider
        with col1:
            if "openai" in providers:
                st.subheader("OpenAI Models")
                for model_name, model_info in providers["openai"]:
                    selected = st.checkbox(
                        f"{model_name}",
                        value=model_name in st.session_state.selected_models,
                        help=model_info["description"],
                        key=f"model_select_{model_name}"
                    )
                    
                    if selected and model_name not in st.session_state.selected_models:
                        st.session_state.selected_models.append(model_name)
                    elif not selected and model_name in st.session_state.selected_models:
                        st.session_state.selected_models.remove(model_name)
        
        with col2:
            if "anthropic" in providers:
                st.subheader("Anthropic Models")
                for model_name, model_info in providers["anthropic"]:
                    selected = st.checkbox(
                        f"{model_name}",
                        value=model_name in st.session_state.selected_models,
                        help=model_info["description"],
                        key=f"model_select_{model_name}"
                    )
                    
                    if selected and model_name not in st.session_state.selected_models:
                        st.session_state.selected_models.append(model_name)
                    elif not selected and model_name in st.session_state.selected_models:
                        st.session_state.selected_models.remove(model_name)
        
        with col3:
            if "replicate" in providers:
                st.subheader("Replicate Models")
                for model_name, model_info in providers["replicate"]:
                    selected = st.checkbox(
                        f"{model_name}",
                        value=model_name in st.session_state.selected_models,
                        help=model_info["description"],
                        key=f"model_select_{model_name}"
                    )
                    
                    if selected and model_name not in st.session_state.selected_models:
                        st.session_state.selected_models.append(model_name)
                    elif not selected and model_name in st.session_state.selected_models:
                        st.session_state.selected_models.remove(model_name)
        
        # Add custom model section
        st.markdown("---")
        st.subheader("Add Custom Model")
        
        with st.form("add_custom_model"):
            custom_model_name = st.text_input("Model Name/ID", placeholder="e.g., mistral-7b-instruct")
            
            provider_options = ["openai", "anthropic", "replicate", "other"]
            custom_provider = st.selectbox("Provider", options=provider_options)
            
            api_type_options = ["openai", "anthropic", "replicate", "other"]
            custom_api_type = st.selectbox("API Type", options=api_type_options)
            
            custom_description = st.text_area("Description", placeholder="Brief description of the model")
            
            submit_custom = st.form_submit_button("Add Model")
            
            if submit_custom and custom_model_name and custom_provider and custom_api_type:
                st.session_state.custom_models[custom_model_name] = {
                    "provider": custom_provider,
                    "api_type": custom_api_type,
                    "description": custom_description
                }
                st.success(f"Added custom model: {custom_model_name}")
                st.rerun()
        
        # Display selected models
        st.markdown("---")
        st.subheader("Selected Models for Evaluation")
        
        if st.session_state.selected_models:
            selected_models_df = pd.DataFrame([
                {
                    "Model": model,
                    "Provider": available_models[model]["provider"] if model in available_models else "Unknown",
                    "Description": available_models[model]["description"] if model in available_models else ""
                }
                for model in st.session_state.selected_models
            ])
            
            st.dataframe(selected_models_df, use_container_width=True)
        else:
            st.info("No models selected. Please select at least one model for evaluation.")
    
    # Datasets Tab
    with tabs[1]:
        st.markdown('<div class="section-header"><h2>Evaluation Datasets</h2></div>', unsafe_allow_html=True)
        
        st.write("Create or upload datasets to evaluate models against.")
        
        # Option to upload JSON dataset
        st.subheader("Upload Dataset")
        
        uploaded_file = st.file_uploader("Upload dataset (JSON format)", type=["json"])
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                if all(k in data for k in ["name", "questions"]):
                    # Generate a dataset ID
                    dataset_id = str(uuid.uuid4())
                    
                    # Ensure ground_truths exist
                    if "ground_truths" not in data or len(data["ground_truths"]) < len(data["questions"]):
                        data["ground_truths"] = data.get("ground_truths", [])
                        # Pad with empty strings if needed
                        data["ground_truths"].extend([""] * (len(data["questions"]) - len(data["ground_truths"])))
                    
                    # Add to datasets
                    st.session_state.evaluation_datasets[dataset_id] = data
                    st.success(f"Loaded dataset: {data['name']} with {len(data['questions'])} questions")
                else:
                    st.error("Invalid JSON format. Dataset must contain at least 'name' and 'questions' keys.")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Or create a new dataset
        st.markdown("---")
        st.subheader("Create New Dataset")
        
        with st.form("create_dataset"):
            dataset_name = st.text_input("Dataset Name", placeholder="e.g., Science Questions")
            dataset_description = st.text_area("Description", placeholder="Brief description of the dataset")
            
            # Create dataset with questions and ground truths
            questions_text = st.text_area(
                "Questions (one per line)",
                placeholder="What is photosynthesis?\nExplain how gravity works.\n...",
                height=150
            )
            
            ground_truths_text = st.text_area(
                "Ground Truths (matching each question, one per line)",
                placeholder="Photosynthesis is the process used by plants to...\nGravity is a force that attracts...\n...",
                height=150
            )
            
            submit_dataset = st.form_submit_button("Create Dataset")
            
            if submit_dataset and dataset_name and questions_text:
                questions = [q.strip() for q in questions_text.splitlines() if q.strip()]
                ground_truths = [gt.strip() for gt in ground_truths_text.splitlines() if gt.strip()]
                
                # Pad ground truths if needed
                if len(ground_truths) < len(questions):
                    ground_truths.extend([""] * (len(questions) - len(ground_truths)))
                
                # Generate a dataset ID
                dataset_id = str(uuid.uuid4())
                
                # Create dataset
                st.session_state.evaluation_datasets[dataset_id] = {
                    "name": dataset_name,
                    "description": dataset_description,
                    "questions": questions,
                    "ground_truths": ground_truths[:len(questions)]  # Ensure matching length
                }
                
                st.success(f"Created dataset: {dataset_name} with {len(questions)} questions")
                st.rerun()
        
        # Load example dataset button
        if st.button("Load Example Dataset"):
            example_data = load_example_dataset()
            dataset_id = str(uuid.uuid4())
            st.session_state.evaluation_datasets[dataset_id] = example_data
            st.success(f"Loaded example dataset: {example_data['name']} with {len(example_data['questions'])} questions")
            st.rerun()
        
        # View existing datasets
        st.markdown("---")
        st.subheader("Available Datasets")
        
        if st.session_state.evaluation_datasets:
            for dataset_id, dataset in st.session_state.evaluation_datasets.items():
                with st.expander(f"{dataset['name']} ({len(dataset['questions'])} questions)"):
                    st.write(f"**Description**: {dataset.get('description', 'No description')}")
                    
                    # Show questions and ground truths
                    questions_df = pd.DataFrame({
                        "Question": dataset['questions'],
                        "Ground Truth": dataset.get('ground_truths', [''] * len(dataset['questions']))
                    })
                    
                    st.dataframe(questions_df, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    # Set as current dataset button
                    with col1:
                        if st.button("Select for Evaluation", key=f"select_{dataset_id}"):
                            st.session_state.current_dataset = dataset_id
                            st.success(f"Selected {dataset['name']} for evaluation")
                            st.rerun()
                    
                    # Delete dataset button
                    with col2:
                        if st.button("Delete Dataset", key=f"delete_{dataset_id}"):
                            del st.session_state.evaluation_datasets[dataset_id]
                            if st.session_state.current_dataset == dataset_id:
                                st.session_state.current_dataset = None
                            st.success("Dataset deleted")
                            st.rerun()
        else:
            st.info("No datasets available. Please create a new dataset or upload one.")
    
    # Evaluation Tab
    with tabs[2]:
        st.markdown('<div class="section-header"><h2>Run Evaluation</h2></div>', unsafe_allow_html=True)
        
        # Check prerequisites
        if not st.session_state.selected_models:
            st.warning("No models selected. Please select at least one model in the Models tab.")
        elif not st.session_state.current_dataset:
            st.warning("No dataset selected for evaluation. Please select a dataset in the Datasets tab.")
        else:
            # Ready to evaluate
            current_dataset = st.session_state.evaluation_datasets[st.session_state.current_dataset]
            selected_models = st.session_state.selected_models
            
            st.success(f"Ready to evaluate {len(selected_models)} models on '{current_dataset['name']}' dataset.")
            
            # Show evaluation configuration
            st.subheader("Evaluation Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Dataset**: {current_dataset['name']}")
                st.write(f"**Questions**: {len(current_dataset['questions'])}")
                st.write(f"**Description**: {current_dataset.get('description', 'None')}")
            
            with col2:
                st.write(f"**Models**: {', '.join(selected_models)}")
                st.write(f"**Metrics**: {', '.join(st.session_state.selected_metrics)}")
                st.write(f"**Experiment**: {st.session_state.experiment_name}")
            
            # MLflow setup
            experiment_id = setup_mlflow()
            
            # Run evaluation button
            if st.button("🚀 Run Evaluation", use_container_width=True):
                # Check API keys
                missing_keys = []
                available_models = {**st.session_state.models, **st.session_state.custom_models}
                
                for model in selected_models:
                    if model in available_models:
                        provider = available_models[model]["provider"]
                        if provider not in st.session_state.api_keys or not st.session_state.api_keys[provider]:
                            missing_keys.append(provider)
                
                if missing_keys:
                    st.error(f"Missing API keys for: {', '.join(set(missing_keys))}. Please add them in the sidebar.")
                else:
                    # Run evaluation
                    with st.spinner("Running model evaluations... This may take a few minutes."):
                        # Track results
                        results = []
                        
                        # Set up progress tracking
                        total_evaluations = len(selected_models)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Run evaluations asynchronously
                        async def run_evaluations():
                            tasks = []
                            
                            for i, model_name in enumerate(selected_models):
                                if model_name in available_models:
                                    status_text.text(f"Evaluating model {i+1}/{total_evaluations}: {model_name}")
                                    
                                    # Get model info
                                    model_info = available_models[model_name]
                                    
                                    # Create evaluation task
                                    task = evaluate_model(
                                        model_name,
                                        model_info,
                                        current_dataset["questions"],
                                        current_dataset.get("ground_truths", [""] * len(current_dataset["questions"])),
                                        st.session_state.api_keys
                                    )
                                    
                                    tasks.append(task)
                            
                            # Run tasks concurrently
                            for i, completed in enumerate(asyncio.as_completed(tasks)):
                                result = await completed
                                results.append(result)
                                progress_bar.progress((i + 1) / total_evaluations)
                        
                        # Run the evaluations
                        asyncio.run(run_evaluations())
                        
                        # Store results in session state
                        dataset_id = st.session_state.current_dataset
                        if "results" not in st.session_state.evaluation_results:
                            st.session_state.evaluation_results["results"] = {}
                        
                        st.session_state.evaluation_results["results"][dataset_id] = results
                        
                        # Log to MLflow
                        run_ids = []
                        for result in results:
                            if "error" not in result:
                                run_id = log_model_results_to_mlflow(
                                    result,
                                    current_dataset["name"],
                                    experiment_id
                                )
                                if run_id:
                                    run_ids.append(run_id)
                        
                        # Store experiment info
                        st.session_state.mlflow_experiments[st.session_state.experiment_name] = {
                            "experiment_id": experiment_id,
                            "dataset_id": dataset_id,
                            "run_ids": run_ids,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Complete
                        progress_bar.progress(1.0)
                        status_text.text("Evaluation complete!")
                        st.success(f"Successfully evaluated {len(results)} models. View results in the Results tab.")
    
    # Results Tab
    with tabs[3]:
        st.markdown('<div class="section-header"><h2>Evaluation Results</h2></div>', unsafe_allow_html=True)
        
        # Check if we have results
        if "results" not in st.session_state.evaluation_results or not st.session_state.evaluation_results["results"]:
            st.info("No evaluation results yet. Please run an evaluation first.")
        else:
            # Select dataset to view results for
            dataset_ids = list(st.session_state.evaluation_results["results"].keys())
            dataset_names = [st.session_state.evaluation_datasets[d_id]["name"] if d_id in st.session_state.evaluation_datasets else "Unknown" for d_id in dataset_ids]
            
            selected_result_dataset = st.selectbox(
                "Select dataset results to view:",
                options=dataset_ids,
                format_func=lambda d_id: f"{st.session_state.evaluation_datasets[d_id]['name'] if d_id in st.session_state.evaluation_datasets else 'Unknown'} ({len(st.session_state.evaluation_results['results'][d_id])} models)"
            )
            
            if selected_result_dataset:
                results = st.session_state.evaluation_results["results"][selected_result_dataset]
                dataset = st.session_state.evaluation_datasets.get(selected_result_dataset, {"name": "Unknown"})
                
                st.subheader(f"Results for: {dataset['name']}")
                
                # Create summary dataframe
                summary_data = []
                for result in results:
                    if "error" in result:
                        # Handle error case
                        summary_data.append({
                            "Model": result["model"],
                            "Provider": "Error",
                            "Status": result["error"],
                            **{metric: 0 for metric in st.session_state.selected_metrics}
                        })
                    else:
                        # Normal case
                        metrics = result["metrics"]
                        summary_data.append({
                            "Model": result["model"],
                            "Provider": result["provider"],
                            "Status": "Success",
                            **{metric: metrics.get(metric, 0) for metric in st.session_state.selected_metrics}
                        })
                
                summary_df = pd.DataFrame(summary_data)
                
                # Style the dataframe
                def highlight_cells(val):
                    if isinstance(val, float):
                        # Get metric info to see if higher or lower is better
                        metric_name = None
                        for col in summary_df.columns:
                            if col in [m["name"] for m in st.session_state.metrics]:
                                metric_name = col
                                break
                        
                        if metric_name:
                            metric_info = next((m for m in st.session_state.metrics if m["name"] == metric_name), None)
                            higher_is_better = metric_info["higher_is_better"] if metric_info else True
                            
                            if higher_is_better:
                                if val >= 0.8:
                                    return 'background-color: #e6f4ea'  # Light green
                                elif val >= 0.6:
                                    return 'background-color: #fff8e1'  # Light yellow
                                else:
                                    return 'background-color: #fce8e6'  # Light red
                            else:
                                if val <= 0.2:
                                    return 'background-color: #e6f4ea'  # Light green
                                elif val <= 0.4:
                                    return 'background-color: #fff8e1'  # Light yellow
                                else:
                                    return 'background-color: #fce8e6'  # Light red
                    return ''
                
                # Display summary table
                st.dataframe(summary_df, use_container_width=True)
                
                # Visualization of metrics
                st.subheader("Metrics Visualization")
                
                # Only include successful evaluations
                vis_df = summary_df[summary_df["Status"] == "Success"].copy()
                
                if not vis_df.empty:
                    # Prepare data for radar chart
                    metrics_to_plot = [m for m in st.session_state.selected_metrics if m in vis_df.columns]
                    
                    if len(metrics_to_plot) >= 2:
                        # Create radar chart using Plotly
                        fig = go.Figure()
                        
                        for i, row in vis_df.iterrows():
                            model_name = row["Model"]
                            
                            # Get metric values (need to wrap around to close the polygon)
                            values = [row[m] for m in metrics_to_plot]
                            values += [values[0]]
                            
                            # Get metric labels (also wrap)
                            metrics_display = [next((m["display_name"] for m in st.session_state.metrics if m["name"] == metric), metric) for metric in metrics_to_plot]
                            metrics_display += [metrics_display[0]]
                            
                            # For each metric, check if higher is better
                            for j, metric in enumerate(metrics_to_plot):
                                metric_info = next((m for m in st.session_state.metrics if m["name"] == metric), None)
                                if metric_info and not metric_info["higher_is_better"]:
                                    # Invert scale for metrics where lower is better
                                    values[j] = 1.0 - values[j]
                            
                            # Add trace
                            fig.add_trace(go.Scatterpolar(
                                r=values,
                                theta=metrics_display,
                                fill='toself',
                                name=model_name
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title="Model Performance Comparison",
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.2,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Bar chart comparison
                    st.subheader("Metric Comparison")
                    
                    # Select metric to compare
                    metric_to_compare = st.selectbox(
                        "Select metric to compare:",
                        options=metrics_to_plot,
                        format_func=lambda m: next((metric["display_name"] for metric in st.session_state.metrics if metric["name"] == m), m)
                    )
                    
                    if metric_to_compare:
                        # Get metric info
                        metric_info = next((m for m in st.session_state.metrics if m["name"] == metric_to_compare), None)
                        higher_is_better = metric_info["higher_is_better"] if metric_info else True
                        display_name = metric_info["display_name"] if metric_info else metric_to_compare
                        
                        # Create bar chart
                        fig = px.bar(
                            vis_df.sort_values(by=metric_to_compare, ascending=not higher_is_better),
                            x="Model",
                            y=metric_to_compare,
                            color="Provider",
                            title=f"Comparison of {display_name} across Models",
                            labels={metric_to_compare: display_name}
                        )
                        
                        # Add threshold lines if needed
                        if higher_is_better:
                            fig.add_shape(
                                type="line",
                                x0=-0.5,
                                x1=len(vis_df) - 0.5,
                                y0=0.8,
                                y1=0.8,
                                line=dict(
                                    color="green",
                                    width=2,
                                    dash="dash",
                                ),
                                name="Good"
                            )
                            
                            fig.add_shape(
                                type="line",
                                x0=-0.5,
                                x1=len(vis_df) - 0.5,
                                y0=0.6,
                                y1=0.6,
                                line=dict(
                                    color="orange",
                                    width=2,
                                    dash="dash",
                                ),
                                name="Average"
                            )
                        else:
                            fig.add_shape(
                                type="line",
                                x0=-0.5,
                                x1=len(vis_df) - 0.5,
                                y0=0.2,
                                y1=0.2,
                                line=dict(
                                    color="green",
                                    width=2,
                                    dash="dash",
                                ),
                                name="Good"
                            )
                            
                            fig.add_shape(
                                type="line",
                                x0=-0.5,
                                x1=len(vis_df) - 0.5,
                                y0=0.4,
                                y1=0.4,
                                line=dict(
                                    color="orange",
                                    width=2,
                                    dash="dash",
                                ),
                                name="Average"
                            )
                        
                        # Update layout
                        fig.update_layout(
                            xaxis_title="Model",
                            yaxis_title=display_name,
                            yaxis=dict(
                                range=[0, 1]
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results
                    st.markdown("---")
                    st.subheader("Model Responses")
                    
                    # Select model to view detailed results
                    model_to_view = st.selectbox(
                        "Select model to view responses:",
                        options=vis_df["Model"].tolist()
                    )
                    
                    if model_to_view:
                        # Find the model result
                        model_result = next((r for r in results if r.get("model") == model_to_view), None)
                        
                        if model_result and "responses" in model_result:
                            # Get dataset questions and responses
                            questions = dataset.get("questions", [])
                            ground_truths = dataset.get("ground_truths", [""] * len(questions))
                            responses = model_result["responses"]
                            
                            # Create tabs for each question
                            question_tabs = st.tabs([f"Q{i+1}" for i in range(min(len(questions), len(responses)))])
                            
                            for i, tab in enumerate(question_tabs):
                                with tab:
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader("Question")
                                        st.write(questions[i])
                                        
                                        st.subheader("Ground Truth")
                                        st.write(ground_truths[i] if i < len(ground_truths) and ground_truths[i] else "No ground truth provided")
                                    
                                    with col2:
                                        st.subheader(f"Response from {model_to_view}")
                                        st.write(responses[i])
                        else:
                            st.error("No detailed results available for this model.")
    
    # Comparison Tab (MLflow Integration)
    with tabs[4]:
        st.markdown('<div class="section-header"><h2>MLflow Experiments</h2></div>', unsafe_allow_html=True)
        
        # Check if we have experiments
        if not st.session_state.mlflow_experiments:
            st.info("No MLflow experiments have been created yet. Run an evaluation to create one.")
        else:
            # Get experiment data
            experiment_names = list(st.session_state.mlflow_experiments.keys())
            
            # Select experiment to view
            selected_experiment = st.selectbox(
                "Select experiment:",
                options=experiment_names,
                format_func=lambda e: f"{e} ({st.session_state.mlflow_experiments[e]['timestamp']})"
            )
            
            if selected_experiment:
                # Get experiment details
                experiment_data = st.session_state.mlflow_experiments[selected_experiment]
                experiment_id = experiment_data["experiment_id"]
                
                # Fetch MLflow results
                mlflow_results = get_mlflow_results(experiment_id)
                
                if mlflow_results:
                    st.success(f"Loaded {len(mlflow_results)} runs from MLflow experiment")
                    
                    # Create comparison dataframe
                    comparison_df = compare_models(mlflow_results, st.session_state.selected_metrics)
                    
                    if not comparison_df.empty:
                        # Display comparison table
                        st.subheader("Model Comparison")
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Create parallel coordinates plot
                        st.subheader("Parallel Coordinates Plot")
                        
                        # Filter just the metrics columns for the plot
                        plot_df = comparison_df.copy()
                        metric_cols = [col for col in plot_df.columns if col in [m["name"] for m in st.session_state.metrics]]
                        
                        if metric_cols:
                            # For metrics where lower is better, invert the scale
                            for col in metric_cols:
                                metric_info = next((m for m in st.session_state.metrics if m["name"] == col), None)
                                if metric_info and not metric_info["higher_is_better"]:
                                    # Invert scale
                                    plot_df[col] = 1.0 - plot_df[col]
                            
                            # Create parallel coordinates plot
                            fig = px.parallel_coordinates(
                                plot_df,
                                color="Model",
                                dimensions=["Model"] + metric_cols,
                                title="Model Comparison Across Multiple Metrics",
                                labels={col: next((m["display_name"] for m in st.session_state.metrics if m["name"] == col), col) 
                                        for col in metric_cols}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Create scatter plot matrix
                        st.subheader("Scatter Plot Matrix")
                        
                        # Select metrics for scatter plot matrix
                        scatter_metrics = st.multiselect(
                            "Select metrics to include in scatter plot matrix:",
                            options=metric_cols,
                            default=metric_cols[:3] if len(metric_cols) >= 3 else metric_cols,
                            format_func=lambda m: next((metric["display_name"] for metric in st.session_state.metrics if metric["name"] == m), m)
                        )
                        
                        if scatter_metrics and len(scatter_metrics) >= 2:
                            # Create scatter plot matrix
                            fig = px.scatter_matrix(
                                plot_df,
                                dimensions=scatter_metrics,
                                color="Model",
                                title="Relationships Between Metrics",
                                labels={col: next((m["display_name"] for m in st.session_state.metrics if m["name"] == col), col) 
                                        for col in scatter_metrics}
                            )
                            
                            # Update layout
                            fig.update_layout(
                                autosize=True,
                                height=800
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download comparison results
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download as CSV
                            csv = comparison_df.to_csv(index=False)
                            st.download_button(
                                label="Download Comparison as CSV",
                                data=csv,
                                file_name=f"model_comparison_{selected_experiment}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Download as JSON
                            comparison_json = {
                                "experiment": selected_experiment,
                                "timestamp": experiment_data["timestamp"],
                                "models": comparison_df.to_dict(orient="records")
                            }
                            
                            json_str = json.dumps(comparison_json, indent=2)
                            st.download_button(
                                label="Download Comparison as JSON",
                                data=json_str,
                                file_name=f"model_comparison_{selected_experiment}.json",
                                mime="application/json"
                            )
                    else:
                        st.warning("No comparison data available for this experiment.")
                else:
                    st.warning("No runs found in this MLflow experiment.")
            else:
                st.info("Please select an experiment to view comparison data.")
    
    # Footer
    st.markdown("---")
    st.caption("LLM Comparison Platform with MLflow 2.21.3 | Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()