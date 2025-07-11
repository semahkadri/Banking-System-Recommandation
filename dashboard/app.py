"""
Streamlit Dashboard for Bank Check Prediction System

Simple dashboard for making predictions and viewing results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Direct imports to avoid package complexity
from src.models.prediction_model import CheckPredictionModel
from src.models.model_manager import ModelManager
from src.data_processing.dataset_builder import DatasetBuilder
from src.models.recommendation_manager import RecommendationManager
from src.api.recommendation_api import RecommendationAPI
from src.utils.data_utils import format_currency_tnd

# Configure page
st.set_page_config(
    page_title="Bank Check Prediction Dashboard",
    page_icon=":bank:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()
if 'recommendation_manager' not in st.session_state:
    st.session_state.recommendation_manager = RecommendationManager()
if 'recommendation_api' not in st.session_state:
    st.session_state.recommendation_api = RecommendationAPI()

def load_prediction_model():
    """Load the prediction model."""
    try:
        # Use the ModelManager to get the active model
        model_manager = ModelManager()
        active_model = model_manager.get_active_model()
        
        if active_model is not None:
            return active_model
        else:
            # Check for legacy prediction_model.json
            model_path = Path("data/models/prediction_model.json")
            if model_path.exists():
                model = CheckPredictionModel()
                model.load_model(str(model_path))
                return model
            else:
                return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def load_dataset():
    """Load the processed dataset."""
    try:
        dataset_path = Path("data/processed/dataset_final.csv")
        
        if dataset_path.exists():
            return pd.read_csv(dataset_path)
        else:
            st.warning("Dataset not found. Please run the data processing pipeline first.")
            return None
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

def main():
    """Main dashboard application."""
    
    # Header
    st.title("🏦 Bank Check Prediction Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "🏠 Home",
            "🔮 Predictions",
            "📊 Model Performance", 
            "📈 Data Analytics",
            "⚙️ Model Management",
            "🎯 Recommendations",
            "📋 Recommendation Analytics"
        ]
    )
    
    # Load model and dataset if not already loaded
    if st.session_state.prediction_model is None:
        with st.spinner("Loading prediction model..."):
            st.session_state.prediction_model = load_prediction_model()
    
    # Also check if we need to reload from ModelManager (in case model was trained)
    if st.session_state.prediction_model is None:
        try:
            active_model = st.session_state.model_manager.get_active_model()
            if active_model is not None:
                st.session_state.prediction_model = active_model
        except Exception:
            pass
    
    if st.session_state.dataset is None:
        with st.spinner("Loading dataset..."):
            st.session_state.dataset = load_dataset()
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Predictions":
        show_predictions_page()
    elif page == "📊 Model Performance":
        show_performance_page()
    elif page == "📈 Data Analytics":
        show_analytics_page()
    elif page == "⚙️ Model Management":
        show_management_page()
    elif page == "🎯 Recommendations":
        show_recommendations_page()
    elif page == "📋 Recommendation Analytics":
        show_recommendation_analytics_page()

def show_home_page():
    """Display the home page."""
    
    st.header("Welcome to the Bank Check Prediction System")
    
    # Overview cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Model Status",
            value="Ready" if st.session_state.prediction_model and st.session_state.prediction_model.is_trained else "Not Ready",
            delta="Trained" if st.session_state.prediction_model and st.session_state.prediction_model.is_trained else "Needs Training"
        )
    
    with col2:
        dataset_size = len(st.session_state.dataset) if st.session_state.dataset is not None else 0
        st.metric(
            label="Dataset Size",
            value=f"{dataset_size:,}",
            delta="Records"
        )
    
    with col3:
        st.metric(
            label="Version",
            value="1.0.0",
            delta="Production"
        )
    
    with col4:
        st.metric(
            label="Features",
            value="15",
            delta="ML Features"
        )
    
    st.markdown("---")
    
    # System Overview
    st.subheader("System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Objectives
        - **Predict number of checks** a client will issue
        - **Predict maximum authorized amount** per check
        - **Analyze client behavior** patterns
        - **Support decision making** for check allocation
        """)
        
        st.markdown("""
        ### Features
        - **User-selectable models** with 3 ML algorithms
        - **Real-time predictions** for banking applications
        - **Interactive dashboard** for analysis
        - **Model performance monitoring**
        """)
    
    with col2:
        if st.session_state.prediction_model and st.session_state.prediction_model.is_trained:
            metrics = st.session_state.prediction_model.metrics
            
            st.markdown("### Model Performance")
            
            # Create metrics visualization
            fig = go.Figure()
            
            models = ['Number of Checks', 'Maximum Amount']
            r2_scores = [
                metrics.get('nbr_cheques', {}).get('r2', 0),
                metrics.get('montant_max', {}).get('r2', 0)
            ]
            
            fig.add_trace(go.Bar(
                x=models,
                y=r2_scores,
                name='R² Score',
                marker_color=['#FF6B6B', '#4ECDC4']
            ))
            
            fig.update_layout(
                title="Model R² Scores",
                yaxis_title="R² Score",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Model not loaded. Please check the model management page.")

def show_predictions_page():
    """Display the predictions page."""
    
    st.header("Client Predictions")
    
    if not st.session_state.prediction_model or not st.session_state.prediction_model.is_trained:
        st.error("Prediction model is not available. Please check the model management page.")
        return
    
    # Single client prediction
    st.subheader("Single Client Prediction")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Client Information")
            client_id = st.text_input("Client ID", value="client_test_001")
            marche = st.selectbox("Market", ["Particuliers", "PME", "TPE", "GEI", "TRE", "PRO"])
            csp = st.text_input("CSP", value="Cadre")
            segment = st.text_input("Segment", value="Segment_A")
            secteur = st.text_input("Activity Sector", value="Services")
            
        with col2:
            st.markdown("### Financial Information")
            revenu = st.number_input("Estimated Revenue", min_value=0.0, value=50000.0)
            nbr_2024 = st.number_input("Number of Checks 2024", min_value=0, value=5)
            montant_2024 = st.number_input("Max Amount 2024", min_value=0.0, value=30000.0)
            ecart_nbr = st.number_input("Check Number Difference", value=2)
            ecart_montant = st.number_input("Amount Difference", value=5000.0)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Behavioral Information")
            demande_derogation = st.checkbox("Has Requested Derogation")
            mobile_banking = st.checkbox("Uses Mobile Banking")
            ratio_cheques = st.slider("Check Payment Ratio", 0.0, 1.0, 0.3)
            
        with col4:
            st.markdown("### Payment Information")
            nb_methodes = st.number_input("Number of Payment Methods", min_value=0, value=3)
            montant_moyen_cheque = st.number_input("Average Check Amount", min_value=0.0, value=1500.0)
            montant_moyen_alt = st.number_input("Average Alternative Amount", min_value=0.0, value=800.0)
        
        submitted = st.form_submit_button("Predict", use_container_width=True)
        
        if submitted:
            # Prepare client data
            client_data = {
                'CLI': client_id,
                'CLIENT_MARCHE': marche,
                'CSP': csp,
                'Segment_NMR': segment,
                'CLT_SECTEUR_ACTIVITE_LIB': secteur,
                'Revenu_Estime': revenu,
                'Nbr_Cheques_2024': nbr_2024,
                'Montant_Max_2024': montant_2024,
                'Ecart_Nbr_Cheques_2024_2025': ecart_nbr,
                'Ecart_Montant_Max_2024_2025': ecart_montant,
                'A_Demande_Derogation': int(demande_derogation),
                'Ratio_Cheques_Paiements': ratio_cheques,
                'Utilise_Mobile_Banking': int(mobile_banking),
                'Nombre_Methodes_Paiement': nb_methodes,
                'Montant_Moyen_Cheque': montant_moyen_cheque,
                'Montant_Moyen_Alternative': montant_moyen_alt
            }
            
            # Make prediction
            try:
                if st.session_state.prediction_model is None:
                    st.error("No trained model available. Please train a model first in the Model Management section.")
                    return
                
                result = st.session_state.prediction_model.predict(client_data)
                
                # Display results
                st.success("Prediction completed successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Predicted Number of Checks",
                        value=result['predicted_nbr_cheques'],
                        delta=f"vs {nbr_2024} in 2024"
                    )
                
                with col2:
                    st.metric(
                        label="Predicted Maximum Amount",
                        value=format_currency_tnd(result['predicted_montant_max']),
                        delta=f"vs {format_currency_tnd(montant_2024)} in 2024"
                    )
                
                with col3:
                    confidence = result['model_confidence']
                    avg_confidence = (confidence['nbr_cheques_r2'] + confidence['montant_max_r2']) / 2
                    st.metric(
                        label="Model Confidence",
                        value=f"{avg_confidence:.1%}",
                        delta="Average R² Score"
                    )
                
                # Detailed results
                with st.expander("Detailed Results"):
                    st.json(result)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

def show_performance_page():
    """Display model performance page."""
    
    st.header("Model Performance Analysis")
    
    if not st.session_state.prediction_model or not st.session_state.prediction_model.is_trained:
        st.error("Model not available. Please check the model management page.")
        return
    
    metrics = st.session_state.prediction_model.metrics
    
    # Model selection info
    if st.session_state.prediction_model and st.session_state.prediction_model.is_trained:
        model_info = st.session_state.prediction_model.get_model_info()
        selected_model = model_info.get('model_type', 'unknown')
        
        model_names = {
            'linear': 'Linear Regression',
            'gradient_boost': 'Gradient Boosting',
            'neural_network': 'Neural Network'
        }
        
        st.info(f"**Current Model**: {model_names.get(selected_model, selected_model)}")
    
    # Performance metrics overview
    st.subheader("Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Number of Checks Model")
        nbr_metrics = metrics.get('nbr_cheques', {})
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("R² Score", f"{nbr_metrics.get('r2', 0):.4f}")
            st.metric("MAE", f"{nbr_metrics.get('mae', 0):.4f}")
        with metric_col2:
            st.metric("MSE", f"{nbr_metrics.get('mse', 0):.4f}")
            st.metric("RMSE", f"{nbr_metrics.get('rmse', 0):.4f}")
    
    with col2:
        st.markdown("### Maximum Amount Model")
        montant_metrics = metrics.get('montant_max', {})
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("R² Score", f"{montant_metrics.get('r2', 0):.4f}")
            st.metric("MAE", f"{montant_metrics.get('mae', 0):,.2f}")
        with metric_col2:
            st.metric("MSE", f"{montant_metrics.get('mse', 0):,.0f}")
            st.metric("RMSE", f"{montant_metrics.get('rmse', 0):,.2f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    importance = st.session_state.prediction_model.get_feature_importance()
    if importance:
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance (Based on Model Weights)"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def show_analytics_page():
    """Display data analytics page."""
    
    st.header("Data Analytics & Insights")
    
    if st.session_state.dataset is None:
        st.error("Dataset not available. Please check the data processing pipeline.")
        return
    
    df = st.session_state.dataset
    
    # Dataset overview
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clients", len(df))
    with col2:
        avg_checks = df['Target_Nbr_Cheques_Futur'].mean()
        st.metric("Avg Checks", f"{avg_checks:.1f}")
    with col3:
        avg_amount = df['Target_Montant_Max_Futur'].mean()
        st.metric("Avg Max Amount", format_currency_tnd(avg_amount, 0))
    with col4:
        derogation_rate = df['A_Demande_Derogation'].mean() * 100
        st.metric("Derogation Rate", f"{derogation_rate:.1f}%")
    
    # Market distribution
    st.subheader("Market Distribution")
    
    market_counts = df['CLIENT_MARCHE'].value_counts()
    fig = px.pie(values=market_counts.values, names=market_counts.index, title="Client Distribution by Market")
    st.plotly_chart(fig, use_container_width=True)
    
    # Target distribution
    st.subheader("Target Variables Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df,
            x='Target_Nbr_Cheques_Futur',
            title="Distribution of Number of Checks"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df,
            x='Target_Montant_Max_Futur',
            title="Distribution of Maximum Amount"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_management_page():
    """Display model management page with advanced multi-model support."""
    
    st.header("🔧 Advanced Model Management")
    
    # Get model manager
    model_manager = st.session_state.model_manager
    
    # Tabs for different management functions
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Train Models", "📚 Model Library", "📊 Model Comparison", "⚙️ Data Pipeline"])
    
    with tab1:
        st.subheader("Train New Models")
        
        # Model selection for training
        model_options = {
            'linear': '⚡ Linear Regression',
            'neural_network': '🧠 Neural Network',
            'gradient_boost': '🚀 Gradient Boosting'
        }
        
        selected_model = st.selectbox(
            "Choose algorithm to train:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            key="model_selection"
        )
        
        # Training button
        if st.button("🎯 Train New Model", type="primary", use_container_width=True):
            if st.session_state.dataset is not None:
                train_new_model(selected_model, None)
            else:
                st.error("Dataset not available. Please run the data pipeline first.")
    
    with tab2:
        st.subheader("📚 Saved Models Library")
        
        # List all saved models
        saved_models = model_manager.list_models()
        
        if saved_models:
            # Active model indicator
            active_model = model_manager.get_active_model()
            if active_model:
                active_id = model_manager.active_model_id
                active_info = next((m for m in saved_models if m["model_id"] == active_id), None)
                if active_info:
                    st.success(f"🎯 **Active Model**: {active_info['model_name']} ({active_info['performance_summary']['overall_score']} accuracy)")
            
            st.markdown("---")
            
            # Model cards
            for model in saved_models:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                    
                    with col1:
                        is_active = model.get("is_active", False)
                        status_icon = "🎯" if is_active else "📦"
                        st.markdown(f"**{status_icon} {model['model_name']}**")
                        st.caption(f"Type: {model['model_type']} | Created: {model['created_date'][:10]}")
                    
                    with col2:
                        if "performance_summary" in model:
                            perf = model["performance_summary"]
                            st.metric("Checks", perf["checks_accuracy"])
                            st.metric("Amounts", perf["amount_accuracy"])
                    
                    with col3:
                        if "performance_summary" in model:
                            st.metric("Overall", perf["overall_score"])
                        
                        if not is_active:
                            if st.button("🎯 Activate", key=f"activate_{model['model_id']}", use_container_width=True):
                                try:
                                    model_manager.set_active_model(model['model_id'])
                                    st.session_state.prediction_model = model_manager.get_active_model()
                                    st.success(f"✅ Activated: {model['model_name']}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to activate model: {e}")
                    
                    with col4:
                        if st.button("🗑️ Delete", key=f"delete_{model['model_id']}", use_container_width=True):
                            try:
                                model_manager.delete_model(model['model_id'])
                                if is_active:
                                    st.session_state.prediction_model = None
                                st.success(f"🗑️ Deleted: {model['model_name']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete model: {e}")
                
                st.markdown("---")
        else:
            st.info("📝 No models saved yet. Train your first model in the 'Train Models' tab!")
    
    with tab3:
        st.subheader("📊 Model Performance Comparison")
        
        comparison = model_manager.get_model_comparison()
        
        if comparison["summary"]["total_models"] > 0:
            # Best performers
            st.markdown("### 🏆 Best Performers")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "checks" in comparison["best_performers"]:
                    best = comparison["best_performers"]["checks"]
                    st.metric(
                        "🔢 Best for Checks",
                        best["accuracy"],
                        help=f"Model: {best['model_name']}"
                    )
            
            with col2:
                if "amounts" in comparison["best_performers"]:
                    best = comparison["best_performers"]["amounts"]
                    st.metric(
                        "💰 Best for Amounts",
                        best["accuracy"],
                        help=f"Model: {best['model_name']}"
                    )
            
            with col3:
                if "overall" in comparison["best_performers"]:
                    best = comparison["best_performers"]["overall"]
                    st.metric(
                        "🎯 Best Overall",
                        best["accuracy"],
                        help=f"Model: {best['model_name']}"
                    )
            
            # Performance chart
            if saved_models:
                st.markdown("### 📈 Performance Visualization")
                
                chart_data = []
                for model in saved_models:
                    if "performance_summary" in model:
                        metrics = model["metrics"]
                        chart_data.append({
                            "Model": model["model_name"],
                            "Type": model["model_type"],
                            "Checks Accuracy": metrics.get("nbr_cheques", {}).get("r2", 0) * 100,
                            "Amount Accuracy": metrics.get("montant_max", {}).get("r2", 0) * 100,
                            "Active": "🎯 Active" if model.get("is_active") else "📦 Saved"
                        })
                
                if chart_data:
                    import plotly.express as px
                    import pandas as pd
                    
                    df = pd.DataFrame(chart_data)
                    
                    fig = px.scatter(
                        df,
                        x="Checks Accuracy",
                        y="Amount Accuracy",
                        color="Type",
                        symbol="Active",
                        size=[100] * len(df),
                        hover_data=["Model"],
                        title="Model Performance Comparison",
                        labels={
                            "Checks Accuracy": "Checks Prediction Accuracy (%)",
                            "Amount Accuracy": "Amount Prediction Accuracy (%)"
                        }
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Train some models first to see performance comparisons!")
    
    with tab4:
        st.subheader("⚙️ Data Processing Pipeline")
        
        # Pipeline status
        pipeline_status = check_pipeline_status()
        
        if pipeline_status["completed"]:
            st.success(f"✅ Pipeline completed: {pipeline_status['records']:,} client records processed")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("📊 Total Clients", f"{pipeline_status['records']:,}")
                st.metric("🔧 Features", pipeline_status.get('features', 'N/A'))
            
            with col2:
                st.metric("📁 Data Files", f"{pipeline_status.get('files', 'N/A')}")
                st.metric("⏱️ Last Run", pipeline_status.get('last_run', 'N/A'))
        else:
            st.warning("⚠️ Data pipeline not completed")
        
        # Pipeline controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Run Data Pipeline", type="primary", use_container_width=True):
                run_data_pipeline()
        
        with col2:
            if pipeline_status["completed"]:
                if st.button("📊 View Data Statistics", use_container_width=True):
                    show_data_statistics()

def train_new_model(model_type: str, model_name: str = None):
    """Train a new model with the enhanced model manager."""
    model_manager = st.session_state.model_manager
    
    # Show training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Convert dataframe to list of dicts
        status_text.text("📊 Preparing training data...")
        progress_bar.progress(10)
        training_data = st.session_state.dataset.to_dict('records')
        
        # Initialize model with selected type
        status_text.text("🔧 Initializing model...")
        progress_bar.progress(20)
        model = CheckPredictionModel()
        model.set_model_type(model_type)
        
        # Create real-time log container
        log_container = st.empty()
        terminal_logs = []
        
        # Custom stdout capture for real-time updates
        import io
        import contextlib
        import sys
        
        class StreamlitLogger:
            def __init__(self, log_container, terminal_logs, progress_bar, status_text):
                self.log_container = log_container
                self.terminal_logs = terminal_logs
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.original_stdout = sys.stdout
            
            def write(self, text):
                if text.strip() and "[TERMINAL]" in text:
                    self.terminal_logs.append(text.strip())
                    # Update progress based on logs
                    if "TRAINING NUMBER OF CHECKS MODEL" in text:
                        self.progress_bar.progress(30)
                        self.status_text.text("🔵 Training checks prediction model...")
                    elif "TRAINING MAXIMUM AMOUNT MODEL" in text:
                        self.progress_bar.progress(60)
                        self.status_text.text("💰 Training amount prediction model...")
                    elif "RESULTS" in text:
                        self.progress_bar.progress(85)
                        self.status_text.text("📈 Evaluating model performance...")
                    elif "COMPLETED" in text:
                        self.progress_bar.progress(90)
                        self.status_text.text("✅ Training completed!")
                    
                    # Show latest logs
                    recent_logs = self.terminal_logs[-8:]
                    log_text = "\n".join(recent_logs)
                    self.log_container.text_area("🖥️ Training Progress", log_text, height=150)
                
                self.original_stdout.write(text)
            
            def flush(self):
                self.original_stdout.flush()
        
        # Train model with real-time logging
        logger = StreamlitLogger(log_container, terminal_logs, progress_bar, status_text)
        
        model_names = {
            'linear': 'Linear Regression',
            'neural_network': 'Neural Network', 
            'gradient_boost': 'Gradient Boosting'
        }
        
        status_text.text(f"🚀 Training {model_names[model_type]}...")
        with contextlib.redirect_stdout(logger):
            model.fit(training_data)
        
        # Save model with enhanced manager
        status_text.text("💾 Saving model...")
        progress_bar.progress(95)
        
        model_id = model_manager.save_model(model, model_name)
        model_manager.set_active_model(model_id)
        st.session_state.prediction_model = model_manager.get_active_model()
        
        progress_bar.progress(100)
        status_text.text("🎉 Training completed successfully!")
        
        # Success message with model info
        saved_model_info = model_manager.model_registry["models"][model_id]
        st.success(f"✅ Model '{saved_model_info['model_name']}' trained and saved successfully!")
        
        # Show performance metrics
        if hasattr(model, 'metrics') and model.metrics:
            st.markdown("### 📊 Training Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                nbr_r2 = model.metrics.get('nbr_cheques', {}).get('r2', 0)
                st.metric(
                    "🔢 Checks Accuracy", 
                    f"{nbr_r2:.1%}",
                    help="How accurately the model predicts number of checks"
                )
            
            with col2:
                amount_r2 = model.metrics.get('montant_max', {}).get('r2', 0)
                st.metric(
                    "💰 Amount Accuracy", 
                    f"{amount_r2:.1%}",
                    help="How accurately the model predicts maximum amounts"
                )
            
            with col3:
                avg_accuracy = (nbr_r2 + amount_r2) / 2
                st.metric(
                    "📈 Overall Score", 
                    f"{avg_accuracy:.1%}",
                    help="Average prediction accuracy across both targets"
                )
        
        # Show training logs
        with st.expander("📋 Complete Training Logs"):
            all_logs = "\n".join(terminal_logs)
            st.text_area("", all_logs, height=200)
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Training failed: {e}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.text(traceback.format_exc())

def check_pipeline_status():
    """Check the status of the data processing pipeline."""
    try:
        dataset_path = Path("data/processed/dataset_final.csv")
        stats_path = Path("data/processed/dataset_statistics.json")
        
        if dataset_path.exists() and stats_path.exists():
            # Load statistics
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            return {
                "completed": True,
                "records": stats.get("dataset_overview", {}).get("total_clients", 0),
                "features": stats.get("dataset_overview", {}).get("total_features", 0),
                "files": len(list(Path("data/processed").glob("*.csv"))) + len(list(Path("data/processed").glob("*.json"))),
                "last_run": dataset_path.stat().st_mtime
            }
        else:
            return {"completed": False}
    except Exception:
        return {"completed": False}

def run_data_pipeline():
    """Run the complete data processing pipeline."""
    with st.spinner("Running complete data processing pipeline..."):
        try:
            builder = DatasetBuilder()
            final_dataset = builder.run_complete_pipeline()
            st.session_state.dataset = pd.DataFrame(final_dataset)
            st.success("✅ Data pipeline completed successfully!")
            st.info(f"📊 Dataset contains {len(final_dataset):,} client records")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")

def show_data_statistics():
    """Show detailed data statistics."""
    try:
        stats_path = Path("data/processed/dataset_statistics.json")
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            st.json(stats)
        else:
            st.warning("Statistics file not found")
    except Exception as e:
        st.error(f"Failed to load statistics: {e}")

def show_recommendations_page():
    """Display the recommendations page."""
    
    st.header("🎯 Système de Recommandations Personnalisées")
    st.markdown("Générez des recommandations personnalisées pour vos clients bancaires")
    
    # Tabs for different recommendation features
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Client Individuel", 
        "📊 Analyse par Segment", 
        "🔍 Profil Détaillé",
        "⚙️ Gestion des Services"
    ])
    
    with tab1:
        st.subheader("Recommandations pour un Client")
        
        # Client selection
        if st.session_state.dataset is not None:
            client_ids = st.session_state.dataset['CLI'].unique()
            selected_client = st.selectbox(
                "Sélectionnez un client:",
                options=client_ids,
                help="Choisissez un client pour générer des recommandations personnalisées"
            )
            
            if st.button("🎯 Générer Recommandations", type="primary"):
                with st.spinner("Génération des recommandations..."):
                    try:
                        # Get recommendations
                        recommendations = st.session_state.recommendation_api.get_client_recommendations(selected_client)
                        
                        if recommendations.get('status') == 'success':
                            rec_data = recommendations['data']
                            
                            # Display client info
                            st.markdown("### 📋 Informations Client")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                behavior_segment = rec_data.get('behavior_profile', {}).get('behavior_segment', 'N/A')
                                st.metric("Segment Comportemental", behavior_segment)
                            
                            with col2:
                                check_score = rec_data.get('behavior_profile', {}).get('check_dependency_score', 0)
                                st.metric("Dépendance Chèques", f"{check_score * 100:.1f}%")
                            
                            with col3:
                                digital_score = rec_data.get('behavior_profile', {}).get('digital_adoption_score', 0)
                                st.metric("Adoption Digitale", f"{digital_score * 100:.1f}%")
                            
                            with col4:
                                reduction_estimate = rec_data.get('impact_estimations', {}).get('pourcentage_reduction', 0)
                                st.metric("Réduction Estimée", f"{reduction_estimate:.1f}%")
                            
                            # Display recommendations
                            st.markdown("### 🎯 Recommandations Personnalisées")
                            
                            for i, rec in enumerate(rec_data.get('recommendations', [])):
                                with st.expander(f"📌 {rec.get('service_info', {}).get('nom', 'Service')} - Score: {rec.get('scores', {}).get('global', 0):.2f}"):
                                    service_info = rec.get('service_info', {})
                                    
                                    st.markdown(f"**Description:** {service_info.get('description', 'N/A')}")
                                    st.markdown(f"**Objectif:** {service_info.get('cible', 'N/A')}")
                                    st.markdown(f"**Coût:** {format_currency_tnd(service_info.get('cout', 0), 0)}")
                                    
                                    # Avantages
                                    avantages = service_info.get('avantages', [])
                                    if avantages:
                                        st.markdown("**Avantages:**")
                                        for avantage in avantages:
                                            st.markdown(f"• {avantage}")
                                    
                                    # Scores détaillés
                                    scores = rec.get('scores', {})
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Score de Base", f"{scores.get('base', 0):.2f}")
                                    with col2:
                                        st.metric("Score d'Urgence", f"{scores.get('urgency', 0):.2f}")
                                    with col3:
                                        st.metric("Score de Faisabilité", f"{scores.get('feasibility', 0):.2f}")
                                    
                                    # Bouton d'adoption
                                    if st.button(f"✅ Marquer comme Adopté", key=f"adopt_{i}"):
                                        adoption_result = st.session_state.recommendation_api.record_service_adoption(
                                            selected_client, rec.get('service_id')
                                        )
                                        if adoption_result.get('status') == 'success':
                                            st.success("✅ Adoption enregistrée avec succès!")
                                        else:
                                            st.error("❌ Erreur lors de l'enregistrement")
                            
                            # Impact estimé
                            st.markdown("### 📈 Impact Estimé")
                            impact = rec_data.get('impact_estimations', {})
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Réduction Chèques", f"{impact.get('reduction_cheques_estimee', 0):.1f}")
                            with col2:
                                st.metric("Pourcentage Réduction", f"{impact.get('pourcentage_reduction', 0):.1f}%")
                            with col3:
                                st.metric("Bénéfice Estimé", format_currency_tnd(impact.get('benefice_bancaire_estime', 0)))
                            
                            # Détails financiers
                            if impact.get('economies_operationnelles', 0) > 0:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Économies Opérationnelles", format_currency_tnd(impact.get('economies_operationnelles', 0)))
                                with col2:
                                    st.metric("Revenus Additionnels", format_currency_tnd(impact.get('revenus_additionnels', 0)))
                            
                            # Insights avancés
                            advanced_insights = rec_data.get('advanced_insights', {})
                            if advanced_insights:
                                st.markdown("### 🔍 Insights Avancés")
                                
                                # Insights comportementaux
                                behavioral_insights = advanced_insights.get('behavioral_insights', [])
                                if behavioral_insights:
                                    st.markdown("#### 📊 Analyse Comportementale")
                                    for insight in behavioral_insights:
                                        st.info(insight)
                                
                                # Prédictions d'évolution
                                evolution = advanced_insights.get('evolution_predictions', {})
                                if evolution:
                                    st.markdown("#### 📈 Prédictions d'Évolution")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Trajectoire", evolution.get('trajectory', 'N/A').title())
                                    with col2:
                                        st.metric("Usage Chèques 6M", f"{evolution.get('check_usage_6m', 0):.0f}")
                                    with col3:
                                        urgency = evolution.get('intervention_urgency', 'low')
                                        color = "🔴" if urgency == 'high' else "🟡" if urgency == 'medium' else "🟢"
                                        st.metric("Urgence", f"{color} {urgency.title()}")
                                
                                # Insights spécifiques au segment
                                segment_insights = advanced_insights.get('segment_specific_insights', {})
                                if segment_insights:
                                    st.markdown("#### 🎯 Insights Segment")
                                    st.markdown(f"**Description:** {segment_insights.get('description', 'N/A')}")
                                    st.markdown(f"**Approche:** {segment_insights.get('approach', 'N/A')}")
                                
                                # Potentiel de valeur
                                value_potential = advanced_insights.get('value_potential', {})
                                if value_potential:
                                    st.markdown("#### 💰 Potentiel de Valeur")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Valeur Totale", format_currency_tnd(value_potential.get('total_value_potential', 0)))
                                    with col2:
                                        category = value_potential.get('value_category', 'LOW')
                                        color = "🟢" if category == 'HIGH' else "🟡" if category == 'MEDIUM' else "🔴"
                                        st.metric("Catégorie", f"{color} {category}")
                                
                                # Plan d'action
                                actions = advanced_insights.get('recommended_actions', [])
                                if actions:
                                    st.markdown("#### 📋 Plan d'Action")
                                    for action in actions:
                                        priority = action.get('priority', 'LOW')
                                        color = "🔴" if priority == 'HIGH' else "🟡" if priority == 'MEDIUM' else "🟢"
                                        st.markdown(f"**{color} {action.get('action', 'Action')}**")
                                        st.markdown(f"- Timeline: {action.get('timeline', 'N/A')}")
                                        st.markdown(f"- Priorité: {priority}")
                                        st.markdown("---")
                                
                                # Indicateurs de succès
                                success_indicators = advanced_insights.get('success_indicators', {})
                                if success_indicators:
                                    st.markdown("#### 📊 Indicateurs de Succès")
                                    primary_kpis = success_indicators.get('primary_kpis', {})
                                    if primary_kpis:
                                        st.markdown("**KPIs Principaux:**")
                                        st.markdown(f"- {primary_kpis.get('check_reduction_target', 'N/A')}")
                                        st.markdown(f"- {primary_kpis.get('service_adoption_target', 'N/A')}")
                                        st.markdown(f"- Timeline: {primary_kpis.get('timeline', 'N/A')}")
                                    
                                    measurement_plan = success_indicators.get('measurement_plan', {})
                                    if measurement_plan:
                                        st.markdown("**Plan de Mesure:**")
                                        st.markdown(f"- Fréquence: {measurement_plan.get('frequency', 'N/A')}")
                                        st.markdown(f"- Points de Révision: {', '.join(measurement_plan.get('review_points', []))}")
                        
                        else:
                            st.error(f"Erreur: {recommendations.get('error', 'Erreur inconnue')}")
                    
                    except Exception as e:
                        st.error(f"Erreur lors de la génération des recommandations: {e}")
        
        else:
            st.warning("⚠️ Aucun dataset disponible. Veuillez d'abord exécuter le pipeline de données.")
    
    with tab2:
        st.subheader("Analyse par Segment")
        
        # Segment selection
        col1, col2 = st.columns(2)
        
        with col1:
            segment_filter = st.selectbox(
                "Filtrer par Segment:",
                options=['', 'S1 Excellence', 'S2 Premium', 'S3 Essentiel', 'S4 Avenir', 'S5 Univers', 'NON SEGMENTE', 'NON SEGMENTE CORPO', 'NOUVEAU CLIENT', 'CLIENT INACTIF', 'A Developper Flux', 'A Intensifier Engagements'],
                index=0
            )
        
        with col2:
            market_filter = st.selectbox(
                "Filtrer par Marché:",
                options=['', 'Particuliers', 'PME', 'TPE', 'GEI', 'TRE', 'PRO'],
                index=0
            )
        
        if st.button("📊 Analyser le Segment", type="primary"):
            with st.spinner("Analyse du segment..."):
                try:
                    segment_analysis = st.session_state.recommendation_api.get_segment_recommendations(
                        segment_filter if segment_filter else None,
                        market_filter if market_filter else None
                    )
                    
                    if segment_analysis.get('status') == 'success':
                        data = segment_analysis['data']
                        
                        st.markdown(f"### 📊 Analyse de {data['total_clients_analyzed']} clients")
                        
                        # Résumé par segment comportemental
                        segment_summary = data.get('segment_summary', {})
                        
                        for behavior_segment, stats in segment_summary.items():
                            with st.expander(f"📈 {behavior_segment} - {stats['count']} clients"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Taux d'Impact Moyen", f"{stats['avg_impact']:.1f}%")
                                
                                with col2:
                                    st.metric("Nombre de Clients", stats['count'])
                                
                                # Services les plus recommandés
                                st.markdown("**Services les plus recommandés:**")
                                common_services = stats.get('common_services', {})
                                for service, count in sorted(common_services.items(), key=lambda x: x[1], reverse=True)[:5]:
                                    st.markdown(f"• {service}: {count} recommandations")
                    
                    else:
                        st.error(f"Erreur: {segment_analysis.get('error', 'Erreur inconnue')}")
                
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse du segment: {e}")
    
    with tab3:
        st.subheader("Profil Détaillé d'un Client")
        
        if st.session_state.dataset is not None:
            client_ids = st.session_state.dataset['CLI'].unique()
            selected_client = st.selectbox(
                "Sélectionnez un client pour l'analyse:",
                options=client_ids,
                key="profile_client"
            )
            
            if st.button("🔍 Analyser le Profil", type="primary"):
                with st.spinner("Analyse du profil client..."):
                    try:
                        profile_analysis = st.session_state.recommendation_api.get_client_profile_analysis(selected_client)
                        
                        if profile_analysis.get('status') == 'success':
                            data = profile_analysis['data']
                            
                            # Informations de base
                            st.markdown("### 📋 Informations de Base")
                            basic_info = data.get('basic_info', {})
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Marché", basic_info.get('marche', 'N/A'))
                            with col2:
                                st.metric("Segment", basic_info.get('segment', 'N/A'))
                            with col3:
                                st.metric("Revenu Estimé", format_currency_tnd(basic_info.get('revenu_estime', 0), 0))
                            
                            # Profil comportemental
                            st.markdown("### 🎯 Profil Comportemental")
                            behavior_profile = data.get('behavior_profile', {})
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Dépendance Chèques", f"{behavior_profile.get('check_dependency_score', 0) * 100:.1f}%")
                            with col2:
                                st.metric("Adoption Digitale", f"{behavior_profile.get('digital_adoption_score', 0) * 100:.1f}%")
                            with col3:
                                st.metric("Évolution Paiements", f"{behavior_profile.get('payment_evolution_score', 0) * 100:.1f}%")
                            with col4:
                                st.metric("Profil de Risque", f"{behavior_profile.get('risk_profile_score', 0) * 100:.1f}%")
                            
                            # Comparaison avec les pairs
                            st.markdown("### 📊 Comparaison avec les Pairs")
                            peer_comparison = data.get('peer_comparison', {})
                            
                            if 'error' not in peer_comparison:
                                st.markdown(f"**Nombre de pairs:** {peer_comparison.get('peer_count', 0)}")
                                
                                metric_comparisons = peer_comparison.get('metric_comparisons', {})
                                for metric, stats in metric_comparisons.items():
                                    with st.expander(f"📈 {metric}"):
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Valeur Client", f"{stats.get('client_value', 0):.1f}")
                                        with col2:
                                            st.metric("Moyenne Pairs", f"{stats.get('peer_mean', 0):.1f}")
                                        with col3:
                                            st.metric("Rang Percentile", f"{stats.get('percentile_rank', 0):.1f}%")
                        
                        else:
                            st.error(f"Erreur: {profile_analysis.get('error', 'Erreur inconnue')}")
                    
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse du profil: {e}")
        
        else:
            st.warning("⚠️ Aucun dataset disponible.")
    
    with tab4:
        st.subheader("Gestion des Services")
        
        # Services disponibles
        services_info = st.session_state.recommendation_api.get_available_services()
        
        if services_info.get('status') == 'success':
            services = services_info['data']['services']
            
            st.markdown("### 🛠️ Services Disponibles")
            
            for service_id, service_info in services.items():
                with st.expander(f"🔧 {service_info['nom']}"):
                    st.markdown(f"**Description:** {service_info['description']}")
                    st.markdown(f"**Coût:** {format_currency_tnd(service_info['cout'], 0)}")
                    st.markdown(f"**Objectif:** {service_info['cible']}")
                    
                    # Avantages
                    avantages = service_info.get('avantages', [])
                    if avantages:
                        st.markdown("**Avantages:**")
                        for avantage in avantages:
                            st.markdown(f"• {avantage}")
        
        # Segments comportementaux
        st.markdown("### 🎯 Segments Comportementaux")
        segments_info = st.session_state.recommendation_api.get_behavior_segments()
        
        if segments_info.get('status') == 'success':
            segments = segments_info['data']['segments']
            
            for segment_id, segment_info in segments.items():
                with st.expander(f"👥 {segment_id}"):
                    st.markdown(f"**Description:** {segment_info['description']}")
                    st.markdown(f"**Approche:** {segment_info['approach']}")
                    
                    # Caractéristiques
                    characteristics = segment_info.get('characteristics', [])
                    if characteristics:
                        st.markdown("**Caractéristiques:**")
                        for char in characteristics:
                            st.markdown(f"• {char}")

def show_recommendation_analytics_page():
    """Display the recommendation analytics page."""
    
    st.header("📋 Analyse des Recommandations")
    st.markdown("Suivi et analyse de l'efficacité du système de recommandations")
    
    # Tabs for different analytics
    tab1, tab2 = st.tabs([
        "📊 Statistiques d'Adoption", 
        "🎯 Rapport d'Efficacité"
    ])
    
    with tab1:
        st.subheader("Statistiques d'Adoption")
        
        # Period selection
        period_days = st.selectbox(
            "Période d'analyse:",
            options=[30, 60, 90, 180, 365],
            index=0,
            help="Sélectionnez la période pour l'analyse des adoptions"
        )
        
        if st.button("📊 Calculer les Statistiques", type="primary"):
            with st.spinner("Calcul des statistiques..."):
                try:
                    stats = st.session_state.recommendation_api.get_adoption_statistics(period_days)
                    
                    if stats.get('status') == 'success':
                        data = stats['data']
                        
                        # Métriques principales
                        st.markdown("### 📊 Métriques Principales")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Taux d'Adoption Global", f"{data.get('overall_adoption_rate', 0):.1f}%")
                        with col2:
                            st.metric("Total Recommandations", f"{data.get('total_recommendations', 0):,}")
                        with col3:
                            st.metric("Total Adoptions", f"{data.get('total_adoptions', 0):,}")
                        with col4:
                            st.metric("Période (jours)", data.get('period_days', 0))
                        
                        # Taux d'adoption par service
                        service_rates = data.get('service_adoption_rates', {})
                        
                        if service_rates:
                            st.markdown("### 🎯 Taux d'Adoption par Service")
                            
                            # Graphique
                            services = list(service_rates.keys())
                            rates = list(service_rates.values())
                            
                            fig = px.bar(
                                x=services, 
                                y=rates,
                                title="Taux d'Adoption par Service",
                                labels={'x': 'Service', 'y': 'Taux d\'Adoption (%)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Tableau détaillé
                            for service, rate in sorted(service_rates.items(), key=lambda x: x[1], reverse=True):
                                st.metric(f"📌 {service}", f"{rate:.1f}%")
                    
                    else:
                        st.error(f"Erreur: {stats.get('error', 'Erreur inconnue')}")
                
                except Exception as e:
                    st.error(f"Erreur lors du calcul des statistiques: {e}")
    
    with tab2:
        st.subheader("Rapport d'Efficacité")
        
        if st.button("📈 Générer le Rapport", type="primary"):
            with st.spinner("Génération du rapport..."):
                try:
                    report = st.session_state.recommendation_api.get_effectiveness_report()
                    
                    if report.get('status') == 'success':
                        data = report['data']
                        
                        # Métriques globales
                        st.markdown("### 🎯 Performance Globale")
                        
                        adoption_rates = data.get('adoption_rates', {})
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**30 jours:**")
                            rate_30d = adoption_rates.get('30_days', {})
                            st.metric("Taux d'Adoption", f"{rate_30d.get('overall_adoption_rate', 0):.1f}%")
                        
                        with col2:
                            st.markdown("**90 jours:**")
                            rate_90d = adoption_rates.get('90_days', {})
                            st.metric("Taux d'Adoption", f"{rate_90d.get('overall_adoption_rate', 0):.1f}%")
                        
                        # Analyse par segment
                        st.markdown("### 👥 Analyse par Segment")
                        segment_analysis = data.get('segment_analysis', {})
                        
                        for segment, stats in segment_analysis.items():
                            with st.expander(f"📊 {segment}"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Clients Analysés", stats.get('recommended', 0))
                                with col2:
                                    st.metric("Adoptions", stats.get('adopted', 0))
                                with col3:
                                    st.metric("Taux d'Adoption", f"{stats.get('adoption_rate', 0):.1f}%")
                        
                        # Services populaires
                        st.markdown("### 🏆 Services les Plus Adoptés")
                        popular_services = data.get('popular_services', {})
                        
                        for service, count in list(popular_services.items())[:10]:
                            st.metric(f"🔧 {service}", f"{count} adoptions")
                        
                        # Impact financier
                        st.markdown("### 💰 Impact Financier")
                        financial_impact = data.get('financial_impact', {})
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Revenus Annuels", format_currency_tnd(financial_impact.get('total_annual_revenue', 0), 0))
                        with col2:
                            st.metric("Revenus par Client", format_currency_tnd(financial_impact.get('average_revenue_per_client', 0), 0))
                        with col3:
                            st.metric("Clients Servis", data.get('total_clients_served', 0))
                    
                    else:
                        st.error(f"Erreur: {report.get('error', 'Erreur inconnue')}")
                
                except Exception as e:
                    st.error(f"Erreur lors de la génération du rapport: {e}")

if __name__ == "__main__":
    main()