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

# Configuration de la page
st.set_page_config(
    page_title="Tableau de Bord - Prédiction Bancaire",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de l'état de session
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
    """Chargement du modèle de prédiction."""
    try:
        # Utiliser le ModelManager pour obtenir le modèle actif
        model_manager = ModelManager()
        active_model = model_manager.get_active_model()
        
        if active_model is not None:
            return active_model
        else:
            # Vérifier l'ancien fichier prediction_model.json
            model_path = Path("data/models/prediction_model.json")
            if model_path.exists():
                model = CheckPredictionModel()
                model.load_model(str(model_path))
                return model
            else:
                return None
    except Exception as e:
        st.error(f"Échec du chargement du modèle: {e}")
        return None

def load_dataset():
    """Chargement du dataset traité."""
    try:
        dataset_path = Path("data/processed/dataset_final.csv")
        
        if dataset_path.exists():
            return pd.read_csv(dataset_path)
        else:
            st.warning("Dataset non trouvé. Veuillez d'abord exécuter le pipeline de traitement des données.")
            return None
    except Exception as e:
        st.error(f"Échec du chargement du dataset: {e}")
        return None

def main():
    """Application principale du tableau de bord."""
    
    # En-tête
    st.title("🏦 Tableau de Bord - Prédiction Bancaire")
    st.markdown("---")
    
    # Barre latérale
    st.sidebar.title("🧭 Navigation")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choisissez une page:",
        [
            "🏠 Accueil",
            "🔮 Prédictions",
            "📊 Performance des Modèles", 
            "📈 Analyse des Données",
            "⚙️ Gestion des Modèles",
            "🎯 Recommandations",
            "📋 Analyse des Recommandations"
        ]
    )
    
    # Chargement du modèle et du dataset si pas déjà chargés
    if st.session_state.prediction_model is None:
        with st.spinner("Chargement du modèle de prédiction..."):
            st.session_state.prediction_model = load_prediction_model()
    
    # Vérifier aussi si on doit recharger depuis ModelManager (au cas où le modèle a été entraîné)
    if st.session_state.prediction_model is None:
        try:
            active_model = st.session_state.model_manager.get_active_model()
            if active_model is not None:
                st.session_state.prediction_model = active_model
        except Exception:
            pass
    
    if st.session_state.dataset is None:
        with st.spinner("Chargement du dataset..."):
            st.session_state.dataset = load_dataset()
    
    # Routage vers la page appropriée
    if page == "🏠 Accueil":
        show_home_page()
    elif page == "🔮 Prédictions":
        show_predictions_page()
    elif page == "📊 Performance des Modèles":
        show_performance_page()
    elif page == "📈 Analyse des Données":
        show_analytics_page()
    elif page == "⚙️ Gestion des Modèles":
        show_management_page()
    elif page == "🎯 Recommandations":
        show_recommendations_page()
    elif page == "📋 Analyse des Recommandations":
        show_recommendation_analytics_page()

def show_home_page():
    """Affichage de la page d'accueil."""
    
    st.header("🏠 Bienvenue dans le Système de Prédiction Bancaire")
    
    # Cartes de vue d'ensemble
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Statut du Modèle",
            value="Prêt" if st.session_state.prediction_model and st.session_state.prediction_model.is_trained else "Non Prêt",
            delta="Entraîné" if st.session_state.prediction_model and st.session_state.prediction_model.is_trained else "Nécessite Entraînement"
        )
    
    with col2:
        dataset_size = len(st.session_state.dataset) if st.session_state.dataset is not None else 0
        st.metric(
            label="Taille du Dataset",
            value=f"{dataset_size:,}",
            delta="Enregistrements"
        )
    
    with col3:
        st.metric(
            label="Version",
            value="1.0.0",
            delta="Production"
        )
    
    with col4:
        st.metric(
            label="Caractéristiques",
            value="15",
            delta="Variables ML"
        )
    
    st.markdown("---")
    
    # Vue d'ensemble du système
    st.subheader("📋 Vue d'Ensemble du Système")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Objectifs
        - **Prédire le nombre de chèques** qu'un client émettra
        - **Prédire le montant maximum autorisé** par chèque
        - **Analyser les modèles de comportement** des clients
        - **Soutenir la prise de décision** pour l'allocation des chèques
        """)
        
        st.markdown("""
        ### ⚡ Fonctionnalités
        - **Modèles sélectionnables** avec 3 algorithmes ML
        - **Prédictions en temps réel** pour applications bancaires
        - **Tableau de bord interactif** pour l'analyse
        - **Surveillance des performances** des modèles
        """)
    
    with col2:
        if st.session_state.prediction_model and st.session_state.prediction_model.is_trained:
            metrics = st.session_state.prediction_model.metrics
            
            st.markdown("### 📊 Performance du Modèle")
            
            # Créer la visualisation des métriques
            fig = go.Figure()
            
            models = ['Nombre de Chèques', 'Montant Maximum']
            r2_scores = [
                metrics.get('nbr_cheques', {}).get('r2', 0),
                metrics.get('montant_max', {}).get('r2', 0)
            ]
            
            fig.add_trace(go.Bar(
                x=models,
                y=r2_scores,
                name='Score R²',
                marker_color=['#FF6B6B', '#4ECDC4']
            ))
            
            fig.update_layout(
                title="Scores R² du Modèle",
                yaxis_title="Score R²",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Modèle non chargé. Veuillez vérifier la page de gestion des modèles.")

def show_predictions_page():
    """Affichage de la page des prédictions."""
    
    st.header("🔮 Prédictions Client")
    
    if not st.session_state.prediction_model or not st.session_state.prediction_model.is_trained:
        st.error("Modèle de prédiction non disponible. Veuillez vérifier la page de gestion des modèles.")
        return
    
    # Prédiction pour un client unique
    st.subheader("👤 Prédiction Client Individuel")
    
    # Formulaire de saisie
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Informations Client")
            client_id = st.text_input("ID Client", value="client_test_001")
            marche = st.selectbox("Marché", ["Particuliers", "PME", "TPE", "GEI", "TRE", "PRO"])
            csp = st.text_input("CSP", value="Cadre")
            segment = st.text_input("Segment", value="Segment_A")
            secteur = st.text_input("Secteur d'Activité", value="Services")
            
        with col2:
            st.markdown("### 💰 Informations Financières")
            revenu = st.number_input("Revenu Estimé", min_value=0.0, value=50000.0)
            nbr_2024 = st.number_input("Nombre de Chèques 2024", min_value=0, value=5)
            montant_2024 = st.number_input("Montant Max 2024", min_value=0.0, value=30000.0)
            ecart_nbr = st.number_input("Différence Nombre Chèques", value=2)
            ecart_montant = st.number_input("Différence Montant", value=5000.0)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 📊 Informations Comportementales")
            demande_derogation = st.checkbox("A Demandé une Dérogation")
            mobile_banking = st.checkbox("Utilise Mobile Banking")
            ratio_cheques = st.slider("Ratio Paiements Chèques", 0.0, 1.0, 0.3)
            
        with col4:
            st.markdown("### 💳 Informations Paiement")
            nb_methodes = st.number_input("Nombre de Méthodes de Paiement", min_value=0, value=3)
            montant_moyen_cheque = st.number_input("Montant Moyen Chèque", min_value=0.0, value=1500.0)
            montant_moyen_alt = st.number_input("Montant Moyen Alternatif", min_value=0.0, value=800.0)
        
        submitted = st.form_submit_button("🔮 Prédire", use_container_width=True)
        
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
            
            # Faire la prédiction
            try:
                if st.session_state.prediction_model is None:
                    st.error("Aucun modèle entraîné disponible. Veuillez d'abord entraîner un modèle dans la section Gestion des Modèles.")
                    return
                
                result = st.session_state.prediction_model.predict(client_data)
                
                # Afficher les résultats
                st.success("✅ Prédiction terminée avec succès!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Nombre de Chèques Prédit",
                        value=result['predicted_nbr_cheques'],
                        delta=f"vs {nbr_2024} en 2024"
                    )
                
                with col2:
                    st.metric(
                        label="Montant Maximum Prédit",
                        value=format_currency_tnd(result['predicted_montant_max']),
                        delta=f"vs {format_currency_tnd(montant_2024)} en 2024"
                    )
                
                with col3:
                    confidence = result['model_confidence']
                    avg_confidence = (confidence['nbr_cheques_r2'] + confidence['montant_max_r2']) / 2
                    st.metric(
                        label="Confiance du Modèle",
                        value=f"{avg_confidence:.1%}",
                        delta="Score R² Moyen"
                    )
                
                # Résultats détaillés
                with st.expander("📊 Résultats Détaillés"):
                    st.json(result)
                
            except Exception as e:
                st.error(f"❌ Échec de la prédiction: {e}")

def show_performance_page():
    """Affichage de la page de performance des modèles."""
    
    st.header("📊 Analyse des Performances des Modèles")
    
    if not st.session_state.prediction_model or not st.session_state.prediction_model.is_trained:
        st.error("Modèle non disponible. Veuillez vérifier la page de gestion des modèles.")
        return
    
    metrics = st.session_state.prediction_model.metrics
    
    # Informations sur la sélection du modèle
    if st.session_state.prediction_model and st.session_state.prediction_model.is_trained:
        model_info = st.session_state.prediction_model.get_model_info()
        selected_model = model_info.get('model_type', 'unknown')
        
        model_names = {
            'linear': 'Régression Linéaire',
            'gradient_boost': 'Gradient Boosting',
            'neural_network': 'Réseau de Neurones'
        }
        
        st.info(f"**Modèle Actuel**: {model_names.get(selected_model, selected_model)}")
    
    # Vue d'ensemble des métriques de performance
    st.subheader("📈 Métriques de Performance des Modèles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔢 Modèle Nombre de Chèques")
        nbr_metrics = metrics.get('nbr_cheques', {})
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Score R²", f"{nbr_metrics.get('r2', 0):.4f}")
            st.metric("MAE", f"{nbr_metrics.get('mae', 0):.4f}")
        with metric_col2:
            st.metric("MSE", f"{nbr_metrics.get('mse', 0):.4f}")
            st.metric("RMSE", f"{nbr_metrics.get('rmse', 0):.4f}")
    
    with col2:
        st.markdown("### 💰 Modèle Montant Maximum")
        montant_metrics = metrics.get('montant_max', {})
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Score R²", f"{montant_metrics.get('r2', 0):.4f}")
            st.metric("MAE", f"{montant_metrics.get('mae', 0):,.2f}")
        with metric_col2:
            st.metric("MSE", f"{montant_metrics.get('mse', 0):,.0f}")
            st.metric("RMSE", f"{montant_metrics.get('rmse', 0):,.2f}")
    
    # Importance des caractéristiques
    st.subheader("🎯 Importance des Caractéristiques")
    
    importance = st.session_state.prediction_model.get_feature_importance()
    if importance:
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['Caractéristique', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Caractéristique',
            orientation='h',
            title="Importance des Caractéristiques (Basée sur les Poids du Modèle)"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def show_analytics_page():
    """Affichage de la page d'analyse des données."""
    
    st.header("📈 Analyse des Données & Insights")
    
    if st.session_state.dataset is None:
        st.error("Dataset non disponible. Veuillez vérifier le pipeline de traitement des données.")
        return
    
    df = st.session_state.dataset
    
    # Vue d'ensemble du dataset
    st.subheader("📊 Vue d'Ensemble du Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clients", len(df))
    with col2:
        avg_checks = df['Target_Nbr_Cheques_Futur'].mean()
        st.metric("Chèques Moyens", f"{avg_checks:.1f}")
    with col3:
        avg_amount = df['Target_Montant_Max_Futur'].mean()
        st.metric("Montant Max Moyen", format_currency_tnd(avg_amount, 0))
    with col4:
        derogation_rate = df['A_Demande_Derogation'].mean() * 100
        st.metric("Taux de Dérogation", f"{derogation_rate:.1f}%")
    
    # Distribution par marché
    st.subheader("🏢 Distribution par Marché")
    
    market_counts = df['CLIENT_MARCHE'].value_counts()
    fig = px.pie(values=market_counts.values, names=market_counts.index, title="Distribution des Clients par Marché")
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution des variables cibles
    st.subheader("🎯 Distribution des Variables Cibles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df,
            x='Target_Nbr_Cheques_Futur',
            title="Distribution du Nombre de Chèques"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df,
            x='Target_Montant_Max_Futur',
            title="Distribution du Montant Maximum"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_management_page():
    """Affichage de la page de gestion des modèles avec support multi-modèles avancé."""
    
    st.header("⚙️ Gestion Avancée des Modèles")
    
    # Obtenir le gestionnaire de modèles
    model_manager = st.session_state.model_manager
    
    # Onglets pour différentes fonctions de gestion
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Entraîner Modèles", "📚 Bibliothèque Modèles", "📊 Comparaison Modèles", "⚙️ Pipeline Données"])
    
    with tab1:
        st.subheader("🏋️ Entraîner de Nouveaux Modèles")
        
        # Sélection du modèle pour l'entraînement
        model_options = {
            'linear': '⚡ Régression Linéaire',
            'neural_network': '🧠 Réseau de Neurones',
            'gradient_boost': '🚀 Gradient Boosting'
        }
        
        selected_model = st.selectbox(
            "Choisissez l'algorithme à entraîner:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            key="model_selection"
        )
        
        # Bouton d'entraînement
        if st.button("🎯 Entraîner Nouveau Modèle", type="primary", use_container_width=True):
            if st.session_state.dataset is not None:
                train_new_model(selected_model, None)
            else:
                st.error("Dataset non disponible. Veuillez d'abord exécuter le pipeline de données.")
    
    with tab2:
        st.subheader("📚 Bibliothèque des Modèles Sauvegardés")
        
        # Lister tous les modèles sauvegardés
        saved_models = model_manager.list_models()
        
        if saved_models:
            # Indicateur du modèle actif
            active_model = model_manager.get_active_model()
            if active_model:
                active_id = model_manager.active_model_id
                active_info = next((m for m in saved_models if m["model_id"] == active_id), None)
                if active_info:
                    st.success(f"🎯 **Modèle Actif**: {active_info['model_name']} ({active_info['performance_summary']['overall_score']} précision)")
            
            st.markdown("---")
            
            # Cartes de modèles
            for model in saved_models:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                    
                    with col1:
                        is_active = model.get("is_active", False)
                        status_icon = "🎯" if is_active else "📦"
                        st.markdown(f"**{status_icon} {model['model_name']}**")
                        st.caption(f"Type: {model['model_type']} | Créé: {model['created_date'][:10]}")
                    
                    with col2:
                        if "performance_summary" in model:
                            perf = model["performance_summary"]
                            st.metric("Chèques", perf["checks_accuracy"])
                            st.metric("Montants", perf["amount_accuracy"])
                    
                    with col3:
                        if "performance_summary" in model:
                            st.metric("Global", perf["overall_score"])
                        
                        if not is_active:
                            if st.button("🎯 Activer", key=f"activate_{model['model_id']}", use_container_width=True):
                                try:
                                    model_manager.set_active_model(model['model_id'])
                                    st.session_state.prediction_model = model_manager.get_active_model()
                                    st.success(f"✅ Activé: {model['model_name']}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Échec de l'activation du modèle: {e}")
                    
                    with col4:
                        if st.button("🗑️ Supprimer", key=f"delete_{model['model_id']}", use_container_width=True):
                            try:
                                model_manager.delete_model(model['model_id'])
                                if is_active:
                                    st.session_state.prediction_model = None
                                st.success(f"🗑️ Supprimé: {model['model_name']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Échec de la suppression du modèle: {e}")
                
                st.markdown("---")
        else:
            st.info("📝 Aucun modèle sauvegardé pour le moment. Entraînez votre premier modèle dans l'onglet 'Entraîner Modèles'!")
    
    with tab3:
        st.subheader("📊 Comparaison des Performances des Modèles")
        
        comparison = model_manager.get_model_comparison()
        
        if comparison["summary"]["total_models"] > 0:
            # Meilleurs performeurs
            st.markdown("### 🏆 Meilleurs Performeurs")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "checks" in comparison["best_performers"]:
                    best = comparison["best_performers"]["checks"]
                    st.metric(
                        "🔢 Meilleur pour Chèques",
                        best["accuracy"],
                        help=f"Modèle: {best['model_name']}"
                    )
            
            with col2:
                if "amounts" in comparison["best_performers"]:
                    best = comparison["best_performers"]["amounts"]
                    st.metric(
                        "💰 Meilleur pour Montants",
                        best["accuracy"],
                        help=f"Modèle: {best['model_name']}"
                    )
            
            with col3:
                if "overall" in comparison["best_performers"]:
                    best = comparison["best_performers"]["overall"]
                    st.metric(
                        "🎯 Meilleur Global",
                        best["accuracy"],
                        help=f"Modèle: {best['model_name']}"
                    )
            
            # Graphique de performance
            if saved_models:
                st.markdown("### 📈 Visualisation des Performances")
                
                chart_data = []
                for model in saved_models:
                    if "performance_summary" in model:
                        metrics = model["metrics"]
                        chart_data.append({
                            "Modèle": model["model_name"],
                            "Type": model["model_type"],
                            "Précision Chèques": metrics.get("nbr_cheques", {}).get("r2", 0) * 100,
                            "Précision Montants": metrics.get("montant_max", {}).get("r2", 0) * 100,
                            "Statut": "🎯 Actif" if model.get("is_active") else "📦 Sauvegardé"
                        })
                
                if chart_data:
                    import plotly.express as px
                    import pandas as pd
                    
                    df = pd.DataFrame(chart_data)
                    
                    fig = px.scatter(
                        df,
                        x="Précision Chèques",
                        y="Précision Montants",
                        color="Type",
                        symbol="Statut",
                        size=[100] * len(df),
                        hover_data=["Modèle"],
                        title="Comparaison des Performances des Modèles",
                        labels={
                            "Précision Chèques": "Précision Prédiction Chèques (%)",
                            "Précision Montants": "Précision Prédiction Montants (%)"
                        }
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Entraînez d'abord quelques modèles pour voir les comparaisons de performances!")
    
    with tab4:
        st.subheader("⚙️ Pipeline de Traitement des Données")
        
        # Statut du pipeline
        pipeline_status = check_pipeline_status()
        
        if pipeline_status["completed"]:
            st.success(f"✅ Pipeline terminé: {pipeline_status['records']:,} enregistrements clients traités")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("📊 Total Clients", f"{pipeline_status['records']:,}")
                st.metric("🔧 Caractéristiques", pipeline_status.get('features', 'N/A'))
            
            with col2:
                st.metric("📁 Fichiers de Données", f"{pipeline_status.get('files', 'N/A')}")
                st.metric("⏱️ Dernière Exécution", pipeline_status.get('last_run', 'N/A'))
        else:
            st.warning("⚠️ Pipeline de données non terminé")
        
        # Contrôles du pipeline
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Exécuter Pipeline de Données", type="primary", use_container_width=True):
                run_data_pipeline()
        
        with col2:
            if pipeline_status["completed"]:
                if st.button("📊 Voir Statistiques des Données", use_container_width=True):
                    show_data_statistics()

def train_new_model(model_type: str, model_name: str = None):
    """Entraîner un nouveau modèle avec le gestionnaire de modèles amélioré."""
    model_manager = st.session_state.model_manager
    
    # Afficher la progression de l'entraînement
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Convertir dataframe en liste de dictionnaires
        status_text.text("📊 Préparation des données d'entraînement...")
        progress_bar.progress(10)
        training_data = st.session_state.dataset.to_dict('records')
        
        # Initialiser le modèle avec le type sélectionné
        status_text.text("🔧 Initialisation du modèle...")
        progress_bar.progress(20)
        model = CheckPredictionModel()
        model.set_model_type(model_type)
        
        # Créer un conteneur de logs en temps réel
        log_container = st.empty()
        terminal_logs = []
        
        # Capture stdout personnalisée pour les mises à jour en temps réel
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
                    # Mettre à jour la progression basée sur les logs
                    if "TRAINING NUMBER OF CHECKS MODEL" in text:
                        self.progress_bar.progress(30)
                        self.status_text.text("🔵 Entraînement du modèle de prédiction des chèques...")
                    elif "TRAINING MAXIMUM AMOUNT MODEL" in text:
                        self.progress_bar.progress(60)
                        self.status_text.text("💰 Entraînement du modèle de prédiction des montants...")
                    elif "RESULTS" in text:
                        self.progress_bar.progress(85)
                        self.status_text.text("📈 Évaluation des performances du modèle...")
                    elif "COMPLETED" in text:
                        self.progress_bar.progress(90)
                        self.status_text.text("✅ Entraînement terminé!")
                    
                    # Afficher les logs récents
                    recent_logs = self.terminal_logs[-8:]
                    log_text = "\n".join(recent_logs)
                    self.log_container.text_area("🖥️ Progression de l'Entraînement", log_text, height=150)
                
                self.original_stdout.write(text)
            
            def flush(self):
                self.original_stdout.flush()
        
        # Entraîner le modèle avec logs en temps réel
        logger = StreamlitLogger(log_container, terminal_logs, progress_bar, status_text)
        
        model_names = {
            'linear': 'Régression Linéaire',
            'neural_network': 'Réseau de Neurones', 
            'gradient_boost': 'Gradient Boosting'
        }
        
        status_text.text(f"🚀 Entraînement {model_names[model_type]}...")
        with contextlib.redirect_stdout(logger):
            model.fit(training_data)
        
        # Sauvegarder le modèle avec le gestionnaire amélioré
        status_text.text("💾 Sauvegarde du modèle...")
        progress_bar.progress(95)
        
        model_id = model_manager.save_model(model, model_name)
        model_manager.set_active_model(model_id)
        st.session_state.prediction_model = model_manager.get_active_model()
        
        progress_bar.progress(100)
        status_text.text("🎉 Entraînement terminé avec succès!")
        
        # Message de succès avec info du modèle
        saved_model_info = model_manager.model_registry["models"][model_id]
        st.success(f"✅ Modèle '{saved_model_info['model_name']}' entraîné et sauvegardé avec succès!")
        
        # Afficher les métriques de performance
        if hasattr(model, 'metrics') and model.metrics:
            st.markdown("### 📊 Résultats de l'Entraînement")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                nbr_r2 = model.metrics.get('nbr_cheques', {}).get('r2', 0)
                st.metric(
                    "🔢 Précision Chèques", 
                    f"{nbr_r2:.1%}",
                    help="Précision de prédiction du nombre de chèques"
                )
            
            with col2:
                amount_r2 = model.metrics.get('montant_max', {}).get('r2', 0)
                st.metric(
                    "💰 Précision Montants", 
                    f"{amount_r2:.1%}",
                    help="Précision de prédiction des montants maximums"
                )
            
            with col3:
                avg_accuracy = (nbr_r2 + amount_r2) / 2
                st.metric(
                    "📈 Score Global", 
                    f"{avg_accuracy:.1%}",
                    help="Précision moyenne de prédiction sur les deux cibles"
                )
        
        # Afficher les logs d'entraînement
        with st.expander("📋 Logs Complets d'Entraînement"):
            all_logs = "\n".join(terminal_logs)
            st.text_area("", all_logs, height=200)
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Échec de l'entraînement: {e}")
        import traceback
        with st.expander("🔍 Détails de l'Erreur"):
            st.text(traceback.format_exc())

def check_pipeline_status():
    """Vérifier le statut du pipeline de traitement des données."""
    try:
        dataset_path = Path("data/processed/dataset_final.csv")
        stats_path = Path("data/processed/dataset_statistics.json")
        
        if dataset_path.exists() and stats_path.exists():
            # Charger les statistiques
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
    """Exécuter le pipeline complet de traitement des données."""
    with st.spinner("Exécution du pipeline complet de traitement des données..."):
        try:
            builder = DatasetBuilder()
            final_dataset = builder.run_complete_pipeline()
            st.session_state.dataset = pd.DataFrame(final_dataset)
            st.success("✅ Pipeline de données terminé avec succès!")
            st.info(f"📊 Le dataset contient {len(final_dataset):,} enregistrements clients")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Échec du pipeline: {e}")

def show_data_statistics():
    """Afficher les statistiques détaillées des données."""
    try:
        stats_path = Path("data/processed/dataset_statistics.json")
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            st.json(stats)
        else:
            st.warning("Fichier de statistiques non trouvé")
    except Exception as e:
        st.error(f"Échec du chargement des statistiques: {e}")

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
        
        # Mode selection
        input_mode = st.radio(
            "Mode de saisie client:",
            ["📋 Client Existant", "✏️ Nouveau Client (Manuel)"],
            help="Choisissez comment vous voulez spécifier le client"
        )
        
        if input_mode == "📋 Client Existant":
            # Client selection from existing dataset
            if st.session_state.dataset is not None:
                client_ids = st.session_state.dataset['CLI'].unique()
                selected_client = st.selectbox(
                    "Sélectionnez un client existant:",
                    options=client_ids,
                    help="Choisissez un client du dataset pour générer des recommandations"
                )
                
                if st.button("🎯 Générer Recommandations", type="primary", key="rec_existing"):
                    with st.spinner("Génération des recommandations..."):
                        try:
                            # Get recommendations for existing client
                            recommendations = st.session_state.recommendation_api.get_client_recommendations(selected_client)
                            display_recommendation_results(recommendations)
                        except Exception as e:
                            st.error(f"Erreur lors de la génération des recommandations: {e}")
            else:
                st.warning("⚠️ Aucun dataset disponible. Veuillez d'abord exécuter le pipeline de données.")
        
        else:  # Nouveau Client (Manuel)
            st.markdown("### ✏️ Saisie Manuelle - Nouveau Client")
            
            # Manual input form (similar to prediction form)
            with st.form("recommendation_manual_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📋 Informations Client")
                    client_id = st.text_input("ID Client", value="nouveau_client_001")
                    marche = st.selectbox("Marché", ["Particuliers", "PME", "TPE", "GEI", "TRE", "PRO"])
                    csp = st.text_input("CSP", value="SALARIE CADRE MOYEN")
                    segment = st.selectbox("Segment", ["S1 Excellence", "S2 Premium", "S3 Essentiel", "S4 Avenir", "S5 Univers", "NON SEGMENTE"])
                    secteur = st.text_input("Secteur d'Activité", value="ADMINISTRATION PUBLIQUE")
                    
                with col2:
                    st.markdown("#### 💰 Informations Financières")
                    revenu = st.number_input("Revenu Estimé (TND)", min_value=0.0, value=50000.0)
                    nbr_2024 = st.number_input("Nombre de Chèques 2024", min_value=0, value=5)
                    montant_2024 = st.number_input("Montant Max Chèque 2024 (TND)", min_value=0.0, value=30000.0)
                    nbr_transactions = st.number_input("Nombre de Transactions 2025", min_value=1, value=20)
                    
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown("#### 📊 Comportement")
                    mobile_banking = st.checkbox("Utilise Mobile Banking")
                    nb_methodes = st.number_input("Nombre de Méthodes de Paiement", min_value=1, value=3)
                    ecart_cheques = st.number_input("Écart Chèques 2024→2025", value=-2)
                    
                with col4:
                    st.markdown("#### 🔧 Autres")
                    demande_derogation = st.checkbox("A Demandé une Dérogation")
                    ecart_montant = st.number_input("Écart Montant Max 2024→2025", value=5000.0)
                    ratio_cheques = st.slider("Ratio Paiements Chèques", 0.0, 1.0, 0.3)
                
                submitted_manual = st.form_submit_button("🎯 Générer Recommandations", use_container_width=True)
                
                if submitted_manual:
                    # Prepare manual client data
                    manual_client_data = {
                        'CLI': client_id,
                        'CLIENT_MARCHE': marche,
                        'CSP': csp,
                        'Segment_NMR': segment,
                        'CLT_SECTEUR_ACTIVITE_LIB': secteur,
                        'Revenu_Estime': revenu,
                        'Nbr_Cheques_2024': nbr_2024,
                        'Montant_Max_2024': montant_2024,
                        'Nbr_Transactions_2025': nbr_transactions,
                        'Ecart_Nbr_Cheques_2024_2025': ecart_cheques,
                        'Ecart_Montant_Max_2024_2025': ecart_montant,
                        'A_Demande_Derogation': int(demande_derogation),
                        'Utilise_Mobile_Banking': int(mobile_banking),
                        'Nombre_Methodes_Paiement': nb_methodes,
                        'Ratio_Cheques_Paiements': ratio_cheques
                    }
                    
                    with st.spinner("Génération des recommandations pour nouveau client..."):
                        try:
                            # Get recommendations for manual client data
                            recommendations = st.session_state.recommendation_api.get_manual_client_recommendations(manual_client_data)
                            display_recommendation_results(recommendations)
                        except Exception as e:
                            st.error(f"Erreur lors de la génération des recommandations: {e}")
    
    with tab2:
        st.subheader("📊 Analyse par Segment Comportemental")
        
        if st.session_state.dataset is not None:
            # Analyze all clients and show segment distribution
            df = st.session_state.dataset
            
            # Calculate behavior segments for all clients
            st.markdown("### 📈 Distribution des Segments")
            
            # Mock segment analysis (since we don't have pre-calculated segments in dataset)
            segment_options = ["TRADITIONNEL_RESISTANT", "TRADITIONNEL_MODERE", "DIGITAL_TRANSITOIRE", 
                             "DIGITAL_ADOPTER", "DIGITAL_NATIF", "EQUILIBRE"]
            
            selected_segment = st.selectbox("Sélectionner un segment à analyser:", segment_options)
            
            if st.button("📊 Analyser ce Segment", type="primary"):
                with st.spinner("Analyse du segment en cours..."):
                    # Mock analysis
                    import random
                    client_count = random.randint(200, 800)
                    avg_checks = random.randint(5, 25)
                    avg_digital = random.randint(20, 80)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Clients dans ce segment", f"{client_count:,}")
                    with col2:
                        st.metric("Chèques moyens/an", avg_checks)
                    with col3:
                        st.metric("Score digital moyen", f"{avg_digital}%")
                    
                    st.markdown("### 🎯 Services Recommandés pour ce Segment")
                    
                    # Segment-specific recommendations
                    segment_recommendations = {
                        "TRADITIONNEL_RESISTANT": ["Formation Services Digitaux", "Accompagnement Personnel", "Carte Bancaire Moderne"],
                        "TRADITIONNEL_MODERE": ["Carte Bancaire Moderne", "Virements Automatiques", "Formation Services Digitaux"],
                        "DIGITAL_TRANSITOIRE": ["Application Mobile Banking", "Paiement Mobile QR Code", "Carte Sans Contact Premium"],
                        "DIGITAL_ADOPTER": ["Pack Services Premium", "Carte Sans Contact Premium", "Paiement Mobile QR Code"],
                        "DIGITAL_NATIF": ["Pack Services Premium", "Application Mobile Banking", "Carte Sans Contact Premium"],
                        "EQUILIBRE": ["Carte Bancaire Moderne", "Application Mobile Banking", "Virements Automatiques"]
                    }
                    
                    for i, service in enumerate(segment_recommendations.get(selected_segment, [])):
                        st.write(f"**{i+1}.** {service}")
        else:
            st.warning("⚠️ Aucun dataset disponible. Veuillez d'abord exécuter le pipeline de données.")
    
    with tab3:
        st.subheader("🔍 Analyse de Profil Détaillé")
        
        if st.session_state.dataset is not None:
            client_ids = st.session_state.dataset['CLI'].unique()
            selected_client = st.selectbox(
                "Sélectionnez un client pour analyse détaillée:",
                options=client_ids,
                key="detailed_profile_client"
            )
            
            if st.button("🔍 Analyser Profil Complet", type="primary"):
                with st.spinner("Analyse détaillée en cours..."):
                    client_data = st.session_state.dataset[st.session_state.dataset['CLI'] == selected_client].iloc[0]
                    
                    # Display comprehensive client profile
                    st.markdown("### 👤 Informations Client")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("ID Client", selected_client)
                        st.metric("Marché", client_data.get('CLIENT_MARCHE', 'N/A'))
                        st.metric("Segment", client_data.get('Segment_NMR', 'N/A'))
                    
                    with col2:
                        st.metric("CSP", client_data.get('CSP', 'N/A'))
                        st.metric("Secteur", client_data.get('CLT_SECTEUR_ACTIVITE_LIB', 'N/A')[:20] + "..." if len(str(client_data.get('CLT_SECTEUR_ACTIVITE_LIB', ''))) > 20 else client_data.get('CLT_SECTEUR_ACTIVITE_LIB', 'N/A'))
                    
                    with col3:
                        st.metric("Mobile Banking", "✅ Oui" if client_data.get('Utilise_Mobile_Banking', 0) else "❌ Non")
                        st.metric("Chèques 2024", f"{client_data.get('Nbr_Cheques_2024', 0)}")
                    
                    # Behavioral analysis
                    st.markdown("### 📊 Analyse Comportementale")
                    
                    # Generate mock behavioral scores
                    try:
                        recommendations = st.session_state.recommendation_api.get_client_recommendations(selected_client)
                        if recommendations.get('status') == 'success':
                            behavior_profile = recommendations['data'].get('behavior_profile', {})
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                check_score = behavior_profile.get('check_dependency_score', 0) * 100
                                st.metric("Dépendance Chèques", f"{check_score:.1f}%")
                            
                            with col2:
                                digital_score = behavior_profile.get('digital_adoption_score', 0) * 100
                                st.metric("Adoption Digitale", f"{digital_score:.1f}%")
                            
                            with col3:
                                evolution_score = behavior_profile.get('payment_evolution_score', 0) * 100
                                st.metric("Évolution Paiements", f"{evolution_score:.1f}%")
                            
                            with col4:
                                segment = behavior_profile.get('behavior_segment', 'N/A')
                                st.metric("Segment Comportemental", segment)
                    
                    except Exception as e:
                        st.info("💡 Analyse comportementale disponible après génération de recommandations")
        else:
            st.warning("⚠️ Aucun dataset disponible. Veuillez d'abord exécuter le pipeline de données.")
    
    with tab4:
        st.subheader("⚙️ Catalogue des Services Bancaires")
        
        # Display service catalog
        st.markdown("### 💼 Services Disponibles")
        
        # Mock service catalog display
        services = {
            "🆓 Services Gratuits": [
                {"nom": "Carte Bancaire Moderne", "cout": "0 TND", "description": "Carte avec technologie sans contact"},
                {"nom": "Application Mobile Banking", "cout": "0 TND", "description": "Gestion complète depuis smartphone"},
                {"nom": "Virements Automatiques", "cout": "0 TND", "description": "Automatisation paiements récurrents"},
                {"nom": "Paiement Mobile QR Code", "cout": "0 TND", "description": "Paiements instantanés par QR"},
                {"nom": "Formation Services Digitaux", "cout": "0 TND", "description": "Accompagnement personnalisé"},
                {"nom": "Accompagnement Personnel", "cout": "0 TND", "description": "Conseiller dédié transition"}
            ],
            "💎 Services Premium": [
                {"nom": "Carte Sans Contact Premium", "cout": "150 TND/an", "description": "Carte avec plafond élevé et assurances"},
                {"nom": "Pack Services Premium", "cout": "600 TND/an", "description": "Ensemble services bancaires avancés"}
            ]
        }
        
        for category, service_list in services.items():
            st.markdown(f"#### {category}")
            
            for service in service_list:
                with st.expander(f"📌 {service['nom']} - {service['cout']}"):
                    st.write(f"**Description:** {service['description']}")
                    st.write(f"**Coût:** {service['cout']}")
                    
                    if service['cout'] == "0 TND":
                        st.success("🎯 Service gratuit - Recommandé pour tous les clients")
                    else:
                        st.info("💼 Service premium - Ciblé clients à hauts revenus")
        
        # Service management
        st.markdown("### 🔧 Gestion des Services")
        st.info("💡 **Note**: Les prix et descriptions des services sont configurés dans le code source. Consultez `src/models/recommendation_engine.py` pour les modifications.")
        
        # Service effectiveness
        if st.button("📈 Voir Efficacité des Services"):
            st.markdown("#### 📊 Statistiques d'Adoption (Simulation)")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Services Gratuits", "73% adoption moyenne")
                st.metric("Mobile Banking", "89% satisfaction")
                st.metric("Carte Bancaire Moderne", "67% adoption")
            
            with col2:
                st.metric("Services Premium", "34% adoption moyenne")
                st.metric("Pack Premium", "45% satisfaction")
                st.metric("Carte Premium", "23% adoption")

def display_recommendation_results(recommendations):
    """Display recommendation results (shared function)."""
    import streamlit as st
    from src.utils.data_utils import format_currency_tnd
    
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
                
                # Note sur l'adoption (pas de bouton dans le contexte des recommandations)
                st.markdown("💡 **Note:** Ce service peut être proposé au client pour adoption")
                if rec.get('service_info', {}).get('cout', 0) == 0:
                    st.markdown("🆓 **Service gratuit** - Facilité d'adoption élevée")
                else:
                    cout = rec.get('service_info', {}).get('cout', 0)
                    st.markdown(f"💰 **Service premium** - {format_currency_tnd(cout, 0)}/an")
        
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
    
    else:
        st.error(f"Erreur: {recommendations.get('error', 'Erreur inconnue')}")

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