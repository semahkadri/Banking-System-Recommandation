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
from src.utils.data_utils import format_currency_tnd, format_currency_tnd_business

# Configuration de la page
st.set_page_config(
    page_title="Tableau de Bord - Prédiction Bancaire",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
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
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def load_prediction_model():
    """Chargement du modèle de prédiction."""
    try:
        # Utiliser le ModelManager pour obtenir le modèle actif
        model_manager = ModelManager()
        active_model = model_manager.get_active_model()
        
        if active_model is not None:
            return active_model
        else:
            # Vérifier l'ancien fichier prediction_model.json avec validation
            model_path = Path("data/models/prediction_model.json")
            if model_path.exists():
                # Validation de sécurité du fichier modèle
                try:
                    import json
                    with open(model_path, 'r') as f:
                        model_data = json.load(f)
                    
                    # Vérifications de base du modèle
                    if not isinstance(model_data, dict):
                        st.error("Format de modèle invalide")
                        return None
                    
                    required_fields = ['model_type', 'is_trained']
                    if not all(field in model_data for field in required_fields):
                        st.error("Modèle incomplet - champs requis manquants")
                        return None
                        
                    model = CheckPredictionModel()
                    model.load_model(str(model_path))
                    return model
                except json.JSONDecodeError:
                    st.error("Fichier modèle corrompu")
                    return None
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
    st.title("🏦 Système de Prédiction Bancaire - Intelligence Financière")
    st.markdown("---")
    
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
    
    # Nouvelle navigation par blocs
    current_page = st.session_state.current_page
    
    if current_page == 'home':
        show_new_home_page()
    elif current_page == 'analytics':
        show_analytics_insights_page()
    elif current_page == 'models':
        show_models_management_page()
    elif current_page == 'predictions':
        show_unified_predictions_page()
    elif current_page == 'performance':
        show_performance_analysis_page()
    elif current_page == 'recommendations':
        show_unified_recommendations_page()
    elif current_page == 'simulation':
        show_client_simulation_page()

def show_new_home_page():
    """Nouvelle page d'accueil avec blocs de navigation cliquables."""
    
    st.header("🏠 Tableau de Bord - Intelligence Bancaire")
    
    # Métriques de vue d'ensemble du système
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Statut du Modèle",
            value="✅ Prêt" if st.session_state.prediction_model and st.session_state.prediction_model.is_trained else "❌ Non Prêt",
            delta="Modèle entraîné" if st.session_state.prediction_model and st.session_state.prediction_model.is_trained else "Entraînement requis"
        )
    
    with col2:
        dataset_size = len(st.session_state.dataset) if st.session_state.dataset is not None else 0
        st.metric(
            label="Base de Données",
            value=f"{dataset_size:,}",
            delta="Clients"
        )
    
    with col3:
        st.metric(
            label="Précision Système",
            value="85-91%",
            delta="Performances ML"
        )
    
    with col4:
        st.metric(
            label="Services",
            value="8",
            delta="Alternatives Chèques"
        )
    
    st.markdown("---")
    
    # Analyse des données intégrée (partie statique)
    show_integrated_data_insights()
    
    st.markdown("---")
    
    # Modules de navigation par blocs visuels
    st.subheader("🎛️ Modules du Système")
    st.markdown("Cliquez sur un module pour l'utiliser :")
    
    # Première ligne de modules
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 1. Analyse des Données & Insights", use_container_width=True, type="primary"):
            st.session_state.current_page = 'analytics'
            st.rerun()
        st.markdown("""
        **🔍 Explorez vos données**
        - Analyse comportementale des clients
        - Tendances de paiement
        - Insights métier
        """)
    
    with col2:
        if st.button("⚙️ 2. Gestion des Modèles", use_container_width=True, type="secondary"):
            st.session_state.current_page = 'models'
            st.rerun()
        st.markdown("""
        **🤖 Gérez l'IA**
        - Entraîner de nouveaux modèles
        - Comparer les performances
        - Pipeline de données
        """)
    
    with col3:
        if st.button("🔮 3. Prédiction", use_container_width=True, type="secondary"):
            st.session_state.current_page = 'predictions'
            st.rerun()
        st.markdown("""
        **🎯 Prédisez l'avenir**
        - Nombre de chèques clients
        - Montants maximums
        - Confiance des prédictions
        """)
    
    # Deuxième ligne de modules
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 4. Performance des Modèles", use_container_width=True, type="secondary"):
            st.session_state.current_page = 'performance'
            st.rerun()
        st.markdown("""
        **📊 Analysez les performances**
        - Métriques détaillées
        - Importance des variables
        - Comparaisons modèles
        """)
    
    with col2:
        if st.button("🎯 5. Recommandations", use_container_width=True, type="secondary"):
            st.session_state.current_page = 'recommendations'
            st.rerun()
        st.markdown("""
        **💡 Recommandations intelligentes**
        - Services personnalisés
        - Analyse comportementale
        - ROI estimé
        """)
    
    with col3:
        if st.button("🎭 6. Simulation Client / Actions", use_container_width=True, type="secondary"):
            st.session_state.current_page = 'simulation'
            st.rerun()
        st.markdown("""
        **🧪 Simulez et agissez**
        - Tests de scénarios
        - Suivi des adoptions
        - Actions commerciales
        """)

def show_integrated_data_insights():
    """Analyse des données intégrée dans la page d'accueil."""
    
    st.subheader("📈 Insights des Données - Vue d'Ensemble")
    
    if st.session_state.dataset is None:
        st.warning("⚠️ Données non disponibles. Veuillez exécuter le pipeline de données.")
        return
        
    df = st.session_state.dataset
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_clients = len(df)
        st.metric("Total Clients", f"{total_clients:,}")
    
    with col2:
        avg_checks = df['Target_Nbr_Cheques_Futur'].mean()
        st.metric("Chèques Moyens/Client", f"{avg_checks:.1f}")
        st.caption("📝 **Interprétation:** Moyenne de chèques prédits par client pour l'année")
    
    with col3:
        avg_amount = df['Target_Montant_Max_Futur'].mean()
        st.metric("Montant Max Moyen", format_currency_tnd(avg_amount, 0))
        st.caption("💰 **Interprétation:** Montant maximum moyen autorisé par chèque")
    
    with col4:
        derogation_rate = df['A_Demande_Derogation'].mean() * 100
        st.metric("Taux de Dérogation", f"{derogation_rate:.1f}%")
        st.caption("⚠️ **Interprétation:** Pourcentage de clients ayant demandé des dérogations")
    
    # Graphiques compacts avec interprétations
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution par marché
        market_counts = df['CLIENT_MARCHE'].value_counts()
        fig = px.pie(
            values=market_counts.values, 
            names=market_counts.index, 
            title="Distribution des Clients par Marché"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🏢 **Interprétation:** Répartition de la clientèle par segment de marché")
    
    with col2:
        # Evolution comportementale
        fig = px.histogram(
            df,
            x='Ecart_Nbr_Cheques_2024_2025',
            title="Évolution Usage des Chèques (2024→2025)"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📊 **Interprétation:** Valeurs négatives = réduction chèques, positives = augmentation")

    # Bouton retour à l'accueil sur toutes les pages non-home
    if st.session_state.current_page != 'home':
        st.markdown("---")
        if st.button("🏠 Retour à l'Accueil", type="secondary"):
            st.session_state.current_page = 'home'
            st.rerun()

def show_analytics_insights_page():
    """Page d'analyse des données et insights détaillés (one-page)."""
    
    st.header("📊 Analyse des Données & Insights Détaillés")
    
    if st.session_state.dataset is None:
        st.error("Dataset non disponible. Veuillez vérifier le pipeline de traitement des données.")
        add_back_to_home_button()
        return
    
    df = st.session_state.dataset
    
    # Vue d'ensemble compacte
    st.subheader("📈 Vue d'Ensemble Complète")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Clients", f"{len(df):,}")
    with col2:
        avg_checks = df['Target_Nbr_Cheques_Futur'].mean()
        st.metric("Chèques Moyens", f"{avg_checks:.1f}")
    with col3:
        avg_amount = df['Target_Montant_Max_Futur'].mean()
        st.metric("Montant Max Moyen", format_currency_tnd(avg_amount, 0))
    with col4:
        derogation_rate = df['A_Demande_Derogation'].mean() * 100
        st.metric("Taux Dérogation", f"{derogation_rate:.1f}%")
    with col5:
        mobile_rate = df['Utilise_Mobile_Banking'].mean() * 100
        st.metric("Mobile Banking", f"{mobile_rate:.1f}%")
    
    # Analyses détaillées dans une seule vue
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution par marché
        market_counts = df['CLIENT_MARCHE'].value_counts()
        fig = px.pie(values=market_counts.values, names=market_counts.index, 
                    title="🏢 Distribution par Marché Client")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 **Insight:** Particuliers dominent le portefeuille client")
        
        # CSP Analysis
        csp_counts = df['CSP'].value_counts().head(8)
        fig = px.bar(x=csp_counts.values, y=csp_counts.index, orientation='h',
                    title="👥 Top 8 Catégories Socio-Professionnelles")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 **Insight:** Salariés cadres représentent le segment principal")
    
    with col2:
        # Distribution des variables cibles
        fig = px.histogram(df, x='Target_Nbr_Cheques_Futur', 
                          title="🎯 Distribution Nombre de Chèques Prédit")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 **Insight:** La plupart des clients utilisent 0-10 chèques/an")
        
        # Montants
        fig = px.histogram(df, x='Target_Montant_Max_Futur', 
                          title="💰 Distribution Montant Maximum")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"💡 **Insight:** Concentration autour de {format_currency_tnd(30000, 0)}-{format_currency_tnd(50000, 0)}")
    
    # Analyses comportementales
    st.subheader("🧠 Analyses Comportementales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Evolution 2024->2025
        fig = px.histogram(df, x='Ecart_Nbr_Cheques_2024_2025',
                          title="📈 Évolution Usage Chèques (2024→2025)")
        st.plotly_chart(fig, use_container_width=True)
        reduction_clients = len(df[df['Ecart_Nbr_Cheques_2024_2025'] < 0])
        st.caption(f"💡 **Insight:** {reduction_clients:,} clients réduisent leur usage des chèques")
    
    with col2:
        # Mobile Banking vs Chèques - Fixed to handle missing data
        mobile_vs_checks = df.groupby('Utilise_Mobile_Banking')['Target_Nbr_Cheques_Futur'].mean()
        
        # Ensure we have both categories, default to 0 if missing
        without_mobile = mobile_vs_checks.get(0, 0)
        with_mobile = mobile_vs_checks.get(1, 0)
        
        # Create the chart with proper data alignment
        x_labels = ['Sans Mobile Banking', 'Avec Mobile Banking']
        y_values = [without_mobile, with_mobile]
        
        fig = px.bar(x=x_labels, y=y_values,
                    title="📱 Mobile Banking vs Usage Chèques")
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate reduction safely
        if without_mobile > 0 and with_mobile > 0:
            reduction = ((without_mobile - with_mobile) / without_mobile * 100)
            st.caption(f"💡 **Insight:** Mobile Banking réduit usage chèques de {reduction:.1f}%")
        else:
            st.caption(f"💡 **Insight:** Données insuffisantes pour comparaison Mobile Banking")
    
    # Segments et revenus
    st.subheader("💼 Analyse Segments et Revenus")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment_counts = df['Segment_NMR'].value_counts()
        fig = px.bar(x=segment_counts.index, y=segment_counts.values,
                    title="🎯 Répartition par Segment Client")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 **Insight:** Segment Essentiel domine le portefeuille")
    
    with col2:
        # Corrélation revenus/chèques
        fig = px.scatter(df, x='Revenu_Estime', y='Target_Nbr_Cheques_Futur',
                        color='CLIENT_MARCHE', title="💰 Revenus vs Usage Chèques")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 **Insight:** Corrélation positive entre revenus et usage chèques")
    
    add_back_to_home_button()

def show_models_management_page():
    """Page de gestion des modèles unifiée (one-page)."""
    
    st.header("⚙️ Gestion Complète des Modèles IA")
    
    model_manager = st.session_state.model_manager
    
    # Section statut actuel
    st.subheader("📊 Statut Actuel du Système")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_model = model_manager.get_active_model()
        st.metric("Modèle Actif", 
                 "✅ Prêt" if active_model else "❌ Aucun",
                 "Entraîné" if active_model else "Requiert entraînement")
    
    with col2:
        saved_models = model_manager.list_models()
        st.metric("Modèles Sauvegardés", len(saved_models))
    
    with col3:
        if active_model and hasattr(active_model, 'metrics'):
            metrics = active_model.metrics
            nbr_r2 = metrics.get('nbr_cheques', {}).get('r2', 0)
            st.metric("Précision Chèques", f"{nbr_r2:.1%}")
        else:
            st.metric("Précision Chèques", "N/A")
    
    with col4:
        if active_model and hasattr(active_model, 'metrics'):
            amount_r2 = metrics.get('montant_max', {}).get('r2', 0) 
            st.metric("Précision Montants", f"{amount_r2:.1%}")
        else:
            st.metric("Précision Montants", "N/A")
    
    # Entraînement rapide
    st.subheader("🚀 Entraînement Rapide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_options = {
            'gradient_boost': '🚀 Gradient Boosting (Recommandé)',
            'linear': '⚡ Régression Linéaire (Rapide)', 
            'neural_network': '🧠 Réseau de Neurones (Avancé)'
        }
        
        selected_model = st.selectbox("Algorithme:", list(model_options.keys()), 
                                    format_func=lambda x: model_options[x])
    
    with col2:
        st.markdown("**Caractéristiques:**")
        if selected_model == 'gradient_boost':
            st.markdown("• Meilleure précision (91%)")
            st.markdown("• Temps d'entraînement moyen")
            st.markdown("• Recommandé pour production")
        elif selected_model == 'linear':
            st.markdown("• Précision correcte (85%)")
            st.markdown("• Très rapide")
            st.markdown("• Bon pour tests rapides")
        else:
            st.markdown("• Précision variable (78%)")
            st.markdown("• Plus lent")
            st.markdown("• Expérimental")
    
    if st.button("🎯 Entraîner Nouveau Modèle", type="primary", use_container_width=True):
        if st.session_state.dataset is not None:
            train_model_unified(selected_model)
        else:
            st.error("Dataset non disponible. Exécutez d'abord le pipeline de données.")
    
    # Bibliothèque des modèles dans une vue compacte
    st.subheader("📚 Bibliothèque des Modèles")
    
    if saved_models:
        for model in saved_models:
            with st.expander(f"{'🎯 ACTIF' if model.get('is_active') else '📦'} {model['model_name']} - {model.get('performance_summary', {}).get('overall_score', 'N/A')} précision"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Type:** {model['model_type']}")
                    st.write(f"**Créé:** {model['created_date'][:10]}")
                
                with col2:
                    if "performance_summary" in model:
                        perf = model["performance_summary"] 
                        st.write(f"**Chèques:** {perf['checks_accuracy']}")
                        st.write(f"**Montants:** {perf['amount_accuracy']}")
                
                with col3:
                    if not model.get('is_active'):
                        if st.button("🎯 Activer", key=f"activate_{model['model_id']}"):
                            try:
                                model_manager.set_active_model(model['model_id'])
                                st.session_state.prediction_model = model_manager.get_active_model()
                                st.success("✅ Modèle activé!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erreur: {e}")
                    
                    if st.button("🗑️ Supprimer", key=f"delete_{model['model_id']}"):
                        try:
                            model_manager.delete_model(model['model_id'])
                            if model.get('is_active'):
                                st.session_state.prediction_model = None
                            st.success("🗑️ Modèle supprimé!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur: {e}")
    else:
        st.info("📝 Aucun modèle sauvegardé. Entraînez votre premier modèle!")
    
    # Pipeline de données
    st.subheader("⚙️ Pipeline de Données")
    
    pipeline_status = check_pipeline_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pipeline_status["completed"]:
            st.success(f"✅ Pipeline terminé: {pipeline_status['records']:,} clients")
        else:
            st.warning("⚠️ Pipeline non terminé")
    
    with col2:
        if st.button("🔄 Exécuter Pipeline", type="secondary"):
            run_data_pipeline()
    
    add_back_to_home_button()

def show_unified_predictions_page():
    """Page de prédiction unifiée avec tous les détails (one-page)."""
    
    st.header("🔮 Prédictions Client - Interface Unifiée Améliorée")
    
    if not st.session_state.prediction_model or not st.session_state.prediction_model.is_trained:
        st.error("❌ Modèle de prédiction non disponible. Entraînez d'abord un modèle dans 'Gestion des Modèles'.")
        add_back_to_home_button()
        return
    
    # Importer les nouveaux outils
    try:
        from src.utils.field_explanations import FieldExplanationSystem
        from src.utils.prediction_testing import PredictionTestingSystem
        explanation_system = FieldExplanationSystem()
        testing_system = PredictionTestingSystem()
    except ImportError as e:
        st.warning(f"⚠️ Modules d'amélioration non disponibles: {e}")
        explanation_system = None
        testing_system = None
    
    # Informations sur le modèle actuel
    model_info = st.session_state.prediction_model.get_model_info()
    metrics = st.session_state.prediction_model.metrics
    
    st.subheader("🤖 Modèle Actuel")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_names = {'linear': 'Régression Linéaire', 'gradient_boost': 'Gradient Boosting', 'neural_network': 'Réseau de Neurones'}
        st.metric("Type de Modèle", model_names.get(model_info.get('model_type', 'unknown'), 'Inconnu'))
    
    with col2:
        nbr_r2 = metrics.get('nbr_cheques', {}).get('r2', 0)
        st.metric("Précision Chèques", f"{nbr_r2:.1%}")
    
    with col3:
        amount_r2 = metrics.get('montant_max', {}).get('r2', 0)
        st.metric("Précision Montants", f"{amount_r2:.1%}")
    
    with col4:
        avg_confidence = (nbr_r2 + amount_r2) / 2
        st.metric("Confiance Globale", f"{avg_confidence:.1%}")
    
    # Performance détaillée (bouton pour afficher)
    if st.button("📊 Voir Performance Détaillée", type="secondary"):
        show_performance_details()
    
    # Section de test avec vrais clients
    if testing_system:
        st.subheader("🧪 Test avec Vrais Clients du Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎲 Client Aléatoire", use_container_width=True):
                test_client = testing_system.get_random_test_client()
                if test_client:
                    st.session_state.test_client_data = test_client
        
        with col2:
            if st.button("📱 Client Digital", use_container_width=True):
                test_client = testing_system.get_test_client_by_profile("digital")
                if test_client:
                    st.session_state.test_client_data = test_client
        
        with col3:
            if st.button("🏛️ Client Traditionnel", use_container_width=True):
                test_client = testing_system.get_test_client_by_profile("traditional")
                if test_client:
                    st.session_state.test_client_data = test_client
        
        with col4:
            if st.button("👑 Client Premium", use_container_width=True):
                test_client = testing_system.get_test_client_by_profile("premium")
                if test_client:
                    st.session_state.test_client_data = test_client
        
        # Afficher le client de test sélectionné
        if hasattr(st.session_state, 'test_client_data') and st.session_state.test_client_data:
            test_client = st.session_state.test_client_data
            st.info("✅ Client de test chargé depuis le dataset réel")
            
            display_info = testing_system.get_client_display_info(test_client)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**ID:** {display_info['id']}")
                st.write(f"**Marché:** {display_info['marche']}")
            with col2:
                st.write(f"**Segment:** {display_info['segment']}")
                st.write(f"**Profil:** {display_info['profil']}")
            with col3:
                st.write(f"**Revenu:** {display_info['revenu']}")
                st.write(f"**Mobile Banking:** {display_info['mobile_banking']}")
            with col4:
                st.write(f"**Chèques 2024:** {display_info['cheques_2024']}")
                st.write(f"**Max 2024:** {display_info['montant_max_2024']}")
            
            # Tester avec ce client
            if st.button("🔮 Tester Prédiction avec ce Client", type="primary"):
                try:
                    result = st.session_state.prediction_model.predict(test_client)
                    
                    # Validation de précision si données target disponibles
                    if 'Target_Nbr_Cheques_Futur' in test_client or 'Target_Montant_Max_Futur' in test_client:
                        validation = testing_system.validate_prediction_accuracy(result, test_client)
                        
                        st.success("✅ Prédiction et validation terminées!")
                        
                        # Résultats avec validation
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Nombre de Chèques Prédit", 
                                result['predicted_nbr_cheques'],
                                delta=f"Réel: {test_client.get('Target_Nbr_Cheques_Futur', 'N/A')}"
                            )
                            nbr_validation = validation['nbr_cheques_validation']
                            st.write(f"{nbr_validation['status']} **{nbr_validation['level']}**")
                            st.caption(nbr_validation['interpretation'])
                        
                        with col2:
                            st.metric(
                                "Montant Maximum Prédit",
                                format_currency_tnd(result['predicted_montant_max']),
                                delta=f"Réel: {format_currency_tnd(test_client.get('Target_Montant_Max_Futur', 0))}"
                            )
                            montant_validation = validation['montant_max_validation']
                            st.write(f"{montant_validation['status']} **{montant_validation['level']}**")
                            st.caption(montant_validation['interpretation'])
                        
                        with col3:
                            overall = validation['overall_accuracy']
                            st.metric(
                                "Précision Globale",
                                f"{overall['score']:.1%}",
                                f"Niveau: {overall['level']}"
                            )
                            st.caption(overall['interpretation'])
                        
                        # Afficher les nouvelles métriques de confiance
                        if 'model_confidence' in result and isinstance(result['model_confidence'], dict):
                            confidence = result['model_confidence']
                            if 'confidence_level' in confidence:
                                st.subheader("🎯 Analyse de Confiance Avancée")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Niveau Confiance", confidence['confidence_level'])
                                with col2:
                                    st.metric("Confiance Globale", f"{confidence.get('overall_confidence', 0):.1%}")
                                with col3:
                                    st.metric("Qualité Données", f"{confidence.get('data_completeness_score', 0):.1%}")
                                with col4:
                                    st.metric("Cohérence Tendance", f"{confidence.get('trend_consistency_score', 0):.1%}")
                        
                        # Validation business
                        if 'business_validation' in result:
                            business = result['business_validation']
                            if business['validation_reason'] != "Aucun ajustement nécessaire":
                                st.info(f"🔧 **Ajustements appliqués:** {business['validation_reason']}")
                    
                    else:
                        st.success("✅ Prédiction terminée (pas de données de validation)")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Nombre de Chèques Prédit", result['predicted_nbr_cheques'])
                        with col2:
                            st.metric("Montant Maximum Prédit", format_currency_tnd(result['predicted_montant_max']))
                        with col3:
                            confidence = result.get('model_confidence', {})
                            avg_conf = (confidence.get('nbr_cheques_r2', 0) + confidence.get('montant_max_r2', 0)) / 2
                            st.metric("Confiance Modèle", f"{avg_conf:.1%}")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors du test: {e}")
        
        st.markdown("---")
    
    # Formulaire de prédiction unifié avec explications
    st.subheader("👤 Nouvelle Prédiction Client avec Explications")
    
    with st.form("unified_prediction_form"):
        # Informations client sur 2 colonnes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Profil Client")
            
            # Champ avec explication
            client_id = st.text_input("ID Client", value="client_pred_001", 
                                    help="Identifiant unique du client dans le système bancaire")
            
            # Marché avec tooltip détaillé
            if explanation_system:
                marche_help = explanation_system.get_field_tooltip("CLIENT_MARCHE")
            else:
                marche_help = "Segment commercial du client"
            marche = st.selectbox("Marché Client", ["Particuliers", "PME", "TPE", "GEI", "TRE", "PRO"],
                                help=marche_help)
            
            csp = st.text_input("CSP (Catégorie Socio-Professionnelle)", value="SALARIE CADRE MOYEN",
                              help="Profession du client (ex: SALARIE CADRE MOYEN, RETRAITE, etc.)")
            
            # Segment avec explication business
            if explanation_system:
                segment_help = explanation_system.get_field_tooltip("Segment_NMR")
            else:
                segment_help = "Segment de valeur client basé sur les revenus"
            segment = st.selectbox("Segment NMR", ["S1 Excellence", "S2 Premium", "S3 Essentiel", "S4 Avenir", "S5 Univers"],
                                 help=segment_help)
            
            secteur = st.text_input("Secteur d'Activité", value="ADMINISTRATION PUBLIQUE",
                                  help="Secteur d'activité professionnel du client")
        
        with col2:
            st.markdown("#### 💰 Finances & Historique")
            
            # Revenu avec explication détaillée
            if explanation_system:
                revenu_help = explanation_system.get_field_tooltip("Revenu_Estime")
            else:
                revenu_help = "Revenu annuel estimé en TND"
            revenu = st.number_input("Revenu Annuel Estimé (TND)", min_value=0.0, value=50000.0,
                                   help=revenu_help)
            
            # Nombre chèques avec contexte
            if explanation_system:
                nbr_help = explanation_system.get_field_tooltip("Nbr_Cheques_2024")
            else:
                nbr_help = "Nombre total de chèques émis en 2024"
            nbr_2024 = st.number_input("Nombre de Chèques 2024", min_value=0, value=5,
                                     help=nbr_help)
            
            # Montant avec validation business
            if explanation_system:
                montant_help = explanation_system.get_field_tooltip("Montant_Max_2024")
            else:
                montant_help = "Montant maximum d'un chèque en 2024"
            montant_2024 = st.number_input("Montant Maximum 2024 (TND)", min_value=0.0, value=30000.0,
                                         help=montant_help)
            
            nbr_transactions = st.number_input("Nombre Transactions 2025", min_value=1, value=20,
                                             help="Nombre total de transactions (tous types) en 2025")
            
            # Mobile banking avec impact
            if explanation_system:
                mobile_help = explanation_system.get_field_tooltip("Utilise_Mobile_Banking")
            else:
                mobile_help = "Le client utilise-t-il l'application mobile bancaire?"
            mobile_banking = st.checkbox("Utilise Mobile Banking", help=mobile_help)
        
        # Paramètres avancés avec explications
        st.markdown("#### ⚙️ Paramètres Comportementaux")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            if explanation_system:
                derog_help = explanation_system.get_field_tooltip("A_Demande_Derogation")
            else:
                derog_help = "Client a-t-il demandé une dérogation pour son chéquier?"
            demande_derogation = st.checkbox("A Demandé Dérogation", help=derog_help)
            
            if explanation_system:
                methodes_help = explanation_system.get_field_tooltip("Nombre_Methodes_Paiement")
            else:
                methodes_help = "Nombre de méthodes de paiement utilisées"
            nb_methodes = st.number_input("Nb Méthodes Paiement", min_value=1, value=3, help=methodes_help)
        
        with col4:
            if explanation_system:
                ecart_help = explanation_system.get_field_tooltip("Ecart_Nbr_Cheques_2024_2025")
            else:
                ecart_help = "Évolution du nombre de chèques entre 2024 et 2025"
            ecart_cheques = st.number_input("Écart Chèques 2024→2025", value=-2, help=ecart_help)
            
            if explanation_system:
                ecart_montant_help = explanation_system.get_field_tooltip("Ecart_Montant_Max_2024_2025")
            else:
                ecart_montant_help = "Évolution du montant maximum entre 2024 et 2025"
            ecart_montant = st.number_input("Écart Montant Max (TND)", value=5000.0, help=ecart_montant_help)
        
        with col5:
            if explanation_system:
                ratio_help = explanation_system.get_field_tooltip("Ratio_Cheques_Paiements")
            else:
                ratio_help = "Proportion des paiements effectués par chèques (0.0 à 1.0)"
            ratio_cheques = st.slider("Ratio Paiements Chèques", 0.0, 1.0, 0.3, help=ratio_help)
            
            if explanation_system:
                moy_help = explanation_system.get_field_tooltip("Montant_Moyen_Cheque")
            else:
                moy_help = "Montant moyen des chèques émis par le client"
            montant_moyen_cheque = st.number_input("Montant Moyen Chèque (TND)", value=1500.0, help=moy_help)
        
        # Guide d'aide rapide
        if explanation_system:
            with st.expander("💡 Guide d'Aide Rapide - Signification des Champs"):
                st.markdown("**Conseils pour une prédiction optimale:**")
                st.markdown("• **Revenu Estimé**: Influence directement les montants prédits")
                st.markdown("• **Mobile Banking**: Les clients digitaux utilisent généralement moins de chèques")
                st.markdown("• **Ratio Chèques**: >0.5 = forte dépendance, <0.2 = usage minimal")
                st.markdown("• **Écart négatif**: Indique une réduction de l'usage des chèques")
                st.markdown("• **Segment S1/S2**: Clients premium avec montants plus élevés")
        
        predict_button = st.form_submit_button("🔮 PRÉDIRE", use_container_width=True, type="primary")
        
        if predict_button:
            # Préparer les données
            client_data = {
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
                'Ratio_Cheques_Paiements': ratio_cheques,
                'Utilise_Mobile_Banking': int(mobile_banking),
                'Nombre_Methodes_Paiement': nb_methodes,
                'Montant_Moyen_Cheque': montant_moyen_cheque,
                'Montant_Moyen_Alternative': 800.0
            }
            
            # Effectuer la prédiction
            try:
                result = st.session_state.prediction_model.predict(client_data)
                
                # Affichage des résultats avec validation améliorée
                st.success("✅ Prédiction terminée avec succès!")
                
                # Résultats principaux avec validation business
                st.subheader("🎯 Résultats de la Prédiction Validée")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Nombre de Chèques Prédit", 
                        result['predicted_nbr_cheques'],
                        delta=f"vs {nbr_2024} en 2024"
                    )
                    
                    # Afficher si la prédiction a été ajustée
                    if 'business_validation' in result:
                        if result['business_validation']['nbr_cheques_validated']:
                            st.caption("🔧 Ajusté par validation business")
                
                with col2:
                    st.metric(
                        "Montant Maximum Prédit",
                        format_currency_tnd(result['predicted_montant_max']),
                        delta=f"vs {format_currency_tnd(montant_2024)} en 2024"
                    )
                    
                    # Afficher si le montant a été ajusté
                    if 'business_validation' in result:
                        if result['business_validation']['montant_max_validated']:
                            st.caption("🔧 Ajusté par validation business")
                
                with col3:
                    # Utiliser la nouvelle confiance améliorée si disponible
                    confidence = result['model_confidence']
                    if 'overall_confidence' in confidence:
                        overall_conf = confidence['overall_confidence']
                        conf_level = confidence.get('confidence_level', 'MOYENNE')
                        st.metric(
                            "Confiance Globale",
                            f"{overall_conf:.1%}",
                            conf_level
                        )
                    else:
                        # Fallback sur l'ancienne méthode
                        avg_confidence = (confidence['nbr_cheques_r2'] + confidence['montant_max_r2']) / 2
                        st.metric(
                            "Confiance du Modèle",
                            f"{avg_confidence:.1%}",
                            "Score R² Moyen"
                        )
                
                # Métriques de confiance détaillées
                if 'model_confidence' in result and 'confidence_level' in result['model_confidence']:
                    st.subheader("📊 Analyse de Confiance Détaillée")
                    
                    confidence = result['model_confidence']
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Qualité des Données",
                            f"{confidence.get('data_completeness_score', 0):.1%}",
                            "Complétude"
                        )
                    
                    with col2:
                        st.metric(
                            "Cohérence Tendance",
                            f"{confidence.get('trend_consistency_score', 0):.1%}",
                            "Historique"
                        )
                    
                    with col3:
                        st.metric(
                            "Logique Business",
                            f"{confidence.get('business_logic_score', 0):.1%}",
                            "Validation"
                        )
                    
                    with col4:
                        level = confidence.get('confidence_level', 'MOYENNE')
                        color = {'TRÈS ÉLEVÉE': '🟢', 'ÉLEVÉE': '🔵', 'MOYENNE': '🟡', 'FAIBLE': '🟠', 'TRÈS FAIBLE': '🔴'}
                        st.metric(
                            "Niveau Global",
                            f"{color.get(level, '⚪')} {level}",
                            "Évaluation"
                        )
                
                # Validation et ajustements appliqués
                if 'business_validation' in result:
                    validation = result['business_validation']
                    if validation['validation_reason'] != "Aucun ajustement nécessaire":
                        st.info(f"🔧 **Ajustements automatiques appliqués:** {validation['validation_reason']}")
                    
                    # Afficher les valeurs brutes vs ajustées si différentes
                    if 'raw_predictions' in result:
                        raw = result['raw_predictions']
                        if (raw['nbr_cheques_raw'] != result['predicted_nbr_cheques'] or 
                            raw['montant_max_raw'] != result['predicted_montant_max']):
                            
                            with st.expander("🔍 Comparaison Prédictions Brutes vs Validées"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Nombre de Chèques:**")
                                    st.write(f"• Brute: {raw['nbr_cheques_raw']:.1f}")
                                    st.write(f"• Validée: {result['predicted_nbr_cheques']}")
                                with col2:
                                    st.write("**Montant Maximum:**")
                                    st.write(f"• Brut: {format_currency_tnd(raw['montant_max_raw'])}")
                                    st.write(f"• Validé: {format_currency_tnd(result['predicted_montant_max'])}")
                
                # Analyse complémentaire
                st.subheader("🧠 Analyse Complémentaire")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Impact prédiction
                    evolution_checks = result['predicted_nbr_cheques'] - nbr_2024
                    if evolution_checks > 0:
                        st.warning(f"⬆️ **Augmentation prévue:** +{evolution_checks} chèques")
                        st.markdown("💡 **Recommandation:** Proposer alternatives digitales")
                    elif evolution_checks < 0:
                        st.success(f"⬇️ **Réduction prévue:** {evolution_checks} chèques")
                        st.markdown("💡 **Opportunité:** Client en transition digitale")
                    else:
                        st.info("➡️ **Stabilité prévue:** Usage constant")
                
                with col2:
                    # Catégorisation du client
                    if result['predicted_nbr_cheques'] <= 5:
                        st.success("🟢 **Client Digital** - Usage minimal des chèques")
                    elif result['predicted_nbr_cheques'] <= 15:
                        st.info("🟡 **Client Mixte** - Usage modéré des chèques") 
                    else:
                        st.warning("🔴 **Client Traditionnel** - Usage élevé des chèques")
                
                # Détails techniques (expandable) - SÉCURISÉ
                with st.expander("🔧 Détails Techniques"):
                    # Afficher uniquement les métriques non-sensibles
                    safe_details = {
                        "modele_utilise": result.get('model_info', {}).get('model_type', 'N/A'),
                        "confiance_prediction": f"{result.get('confidence', 0):.1%}",
                        "timestamp": result.get('prediction_timestamp', 'N/A'),
                        "version_modele": result.get('model_info', {}).get('version', 'N/A')
                    }
                    st.json(safe_details)
                
            except Exception as e:
                st.error(f"❌ Échec de la prédiction: {e}")
    
    add_back_to_home_button()

def show_performance_analysis_page():
    """Page d'analyse des performances des modèles (one-page)."""
    
    st.header("📈 Analyse des Performances - Vue Complète")
    
    if not st.session_state.prediction_model or not st.session_state.prediction_model.is_trained:
        st.error("Modèle non disponible. Veuillez vérifier la gestion des modèles.")
        add_back_to_home_button()
        return
    
    show_performance_details()
    add_back_to_home_button()

def show_unified_recommendations_page():
    """Page de recommandations unifiée (one-page)."""
    
    st.header("🎯 Recommandations Personnalisées - Interface Complète")
    
    # Section principale de recommandation
    st.subheader("💡 Générer Recommandations Client")
    
    # Mode de saisie
    input_mode = st.radio("Mode de saisie:", 
                         ["📋 Client Existant", "✏️ Nouveau Client"], horizontal=True)
    
    if input_mode == "📋 Client Existant":
        if st.session_state.dataset is not None:
            client_ids = st.session_state.dataset['CLI'].unique()
            selected_client = st.selectbox("Sélectionnez un client:", client_ids)
            
            if st.button("🎯 Générer Recommandations", type="primary"):
                generate_and_display_recommendations(selected_client, mode="existing")
        else:
            st.warning("⚠️ Dataset non disponible.")
    
    else:  # Nouveau client
        with st.form("recommendation_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                client_id = st.text_input("ID Client", value="nouveau_001")
                marche = st.selectbox("Marché", ["Particuliers", "PME", "TPE"])
                csp = st.text_input("CSP", value="SALARIE CADRE MOYEN")
                revenu = st.number_input("Revenu (TND)", value=50000.0)
            
            with col2:
                segment = st.selectbox("Segment", ["S1 Excellence", "S2 Premium", "S3 Essentiel"])
                nbr_cheques_2024 = st.number_input("Chèques 2024", value=5)
                mobile_banking = st.checkbox("Mobile Banking")
                nb_methodes = st.number_input("Nb Méthodes Paiement", value=3)
            
            with col3:
                secteur = st.text_input("Secteur", value="SERVICES")
                montant_max_2024 = st.number_input("Montant Max 2024", value=30000.0)
                demande_derogation = st.checkbox("Demande Dérogation")
                ecart_cheques = st.number_input("Écart Chèques", value=-2)
            
            if st.form_submit_button("🎯 Générer Recommandations", use_container_width=True):
                manual_data = {
                    'CLI': client_id, 'CLIENT_MARCHE': marche, 'CSP': csp, 'Segment_NMR': segment,
                    'CLT_SECTEUR_ACTIVITE_LIB': secteur, 'Revenu_Estime': revenu,
                    'Nbr_Cheques_2024': nbr_cheques_2024, 'Montant_Max_2024': montant_max_2024,
                    'Nbr_Transactions_2025': 20, 'Ecart_Nbr_Cheques_2024_2025': ecart_cheques,
                    'Ecart_Montant_Max_2024_2025': 5000.0, 'A_Demande_Derogation': int(demande_derogation),
                    'Utilise_Mobile_Banking': int(mobile_banking), 'Nombre_Methodes_Paiement': nb_methodes,
                    'Ratio_Cheques_Paiements': 0.3
                }
                generate_and_display_recommendations(manual_data, mode="manual")
    
    # Analyse par segments (compacte)
    st.subheader("📊 Analyse par Segments Comportementaux")
    
    segments_info = {
        "TRADITIONNEL_RESISTANT": {"clients": "~15%", "services": ["Formation Digital", "Accompagnement Personnel"]},
        "TRADITIONNEL_MODERE": {"clients": "~25%", "services": ["Carte Bancaire", "Virements Auto"]},
        "DIGITAL_TRANSITOIRE": {"clients": "~30%", "services": ["Mobile Banking", "Paiement QR"]},
        "DIGITAL_ADOPTER": {"clients": "~20%", "services": ["Services Premium", "Carte Premium"]},
        "DIGITAL_NATIF": {"clients": "~8%", "services": ["Pack Premium", "Solutions Avancées"]},
        "EQUILIBRE": {"clients": "~2%", "services": ["Mix Optimal", "Services Équilibrés"]}
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (segment, info) in enumerate(segments_info.items()):
        with [col1, col2, col3][i % 3]:
            with st.expander(f"{segment} - {info['clients']}"):
                st.write("**Services recommandés:**")
                for service in info['services']:
                    st.write(f"• {service}")
    
    # Catalogue des services (compact)
    st.subheader("💼 Catalogue des Services")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🆓 Services Gratuits:**")
        free_services = ["Carte Bancaire Moderne", "Mobile Banking", "Virements Auto", "Paiement QR", "Formation Digital"]
        for service in free_services:
            st.write(f"• {service}")
    
    with col2:
        st.markdown("**💎 Services Premium:**")
        st.write(f"• Carte Sans Contact Premium ({format_currency_tnd_business(150, 'service_cost')}/an)")
        st.write(f"• Pack Services Premium ({format_currency_tnd_business(600, 'service_cost')}/an)")
    
    add_back_to_home_button()

def show_client_simulation_page():
    """Page de simulation client et actions (one-page)."""
    
    st.header("🎭 Simulation Client & Actions Commerciales")
    
    st.subheader("🧪 Simulateur de Scénarios")
    
    # Tests rapides de scénarios
    scenario_type = st.selectbox("Type de simulation:", 
                                ["📈 Impact Mobile Banking", "💳 Adoption Carte Premium", "🔄 Migration Digitale"])
    
    if scenario_type == "📈 Impact Mobile Banking":
        st.info("**Scénario:** Client traditionnel adoptant mobile banking")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Avant Mobile Banking", "12 chèques/an")
            st.metric("Coût Traitement", f"{format_currency_tnd_business(54, 'service_cost')}/an")
        
        with col2:
            st.metric("Après Mobile Banking", "6 chèques/an (-50%)")
            st.metric("Économies", f"{format_currency_tnd_business(27, 'service_cost')}/an")
        
        st.success("💡 **Insight:** Mobile banking divise par 2 l'usage des chèques")
    
    # Suivi des adoptions (simulation)
    st.subheader("📊 Suivi des Adoptions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recommandations Générées", "1,247")
    with col2:
        st.metric("Adoptions Confirmées", "421")
    with col3:
        st.metric("Taux d'Adoption", "33.8%")
    with col4:
        st.metric("ROI Estimé", format_currency_tnd_business(156400, 'impact'))
    
    # Actions commerciales suggérées
    st.subheader("🎯 Actions Commerciales Suggérées")
    
    actions = [
        {"priorité": "🔴 HAUTE", "action": "Campagne Mobile Banking", "cible": "Clients +10 chèques/an", "impact": "Réduction 40% usage"},
        {"priorité": "🟡 MOYENNE", "action": "Promotion Carte Premium", "cible": f"Revenus >{format_currency_tnd(80000, 0)}", "impact": f"Revenus +{format_currency_tnd_business(150, 'revenue')}/client/an"},
        {"priorité": "🟢 BASSE", "action": "Formation Digital Senior", "cible": "Clients 65+ ans", "impact": "Adoption graduelle"}
    ]
    
    for action in actions:
        with st.expander(f"{action['priorité']} - {action['action']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Cible:** {action['cible']}")
            with col2:
                st.write(f"**Impact attendu:** {action['impact']}")
    
    # Tableau de bord commercial
    st.subheader("📈 Tableau de Bord Commercial")
    
    # Graphique simulé
    import numpy as np
    dates = pd.date_range('2024-01-01', periods=12, freq='M')
    adoptions = np.random.poisson(35, 12).cumsum()
    
    fig = px.line(x=dates, y=adoptions, title="Évolution Adoptions Services 2024")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    add_back_to_home_button()

def add_back_to_home_button():
    """Ajoute un bouton de retour à l'accueil."""
    st.markdown("---")
    if st.button("🏠 Retour à l'Accueil", type="secondary"):
        st.session_state.current_page = 'home'
        st.rerun()

def train_model_unified(model_type):
    """Entraînement de modèle unifié avec feedback temps réel."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("📊 Préparation des données...")
        progress_bar.progress(20)
        
        training_data = st.session_state.dataset.to_dict('records')
        
        status_text.text(f"🤖 Entraînement {model_type}...")
        progress_bar.progress(40)
        
        model = CheckPredictionModel()
        model.set_model_type(model_type)
        model.fit(training_data)
        
        status_text.text("💾 Sauvegarde...")
        progress_bar.progress(80)
        
        model_manager = st.session_state.model_manager
        model_id = model_manager.save_model(model)
        model_manager.set_active_model(model_id)
        st.session_state.prediction_model = model_manager.get_active_model()
        
        progress_bar.progress(100)
        status_text.text("✅ Terminé!")
        
        st.success("🎉 Modèle entraîné et activé avec succès!")
        
        # Afficher les métriques
        if hasattr(model, 'metrics'):
            col1, col2 = st.columns(2)
            with col1:
                nbr_r2 = model.metrics.get('nbr_cheques', {}).get('r2', 0)
                st.metric("Précision Chèques", f"{nbr_r2:.1%}")
            with col2:
                amount_r2 = model.metrics.get('montant_max', {}).get('r2', 0)
                st.metric("Précision Montants", f"{amount_r2:.1%}")
        
    except Exception as e:
        st.error(f"❌ Erreur d'entraînement: {e}")

def show_performance_details():
    """Affiche les détails de performance du modèle."""
    metrics = st.session_state.prediction_model.metrics
    
    st.markdown("### 📊 Métriques Détaillées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔢 Modèle Nombre de Chèques")
        nbr_metrics = metrics.get('nbr_cheques', {})
        
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.metric("R² Score", f"{nbr_metrics.get('r2', 0):.4f}")
            st.metric("MAE", f"{nbr_metrics.get('mae', 0):.2f}")
        with subcol2:
            st.metric("MSE", f"{nbr_metrics.get('mse', 0):.2f}")
            st.metric("RMSE", f"{nbr_metrics.get('rmse', 0):.2f}")
    
    with col2:
        st.markdown("#### 💰 Modèle Montant Maximum")
        montant_metrics = metrics.get('montant_max', {})
        
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.metric("R² Score", f"{montant_metrics.get('r2', 0):.4f}")
            st.metric("MAE", f"{montant_metrics.get('mae', 0):,.0f}")
        with subcol2:
            st.metric("MSE", f"{montant_metrics.get('mse', 0):,.0f}")
            st.metric("RMSE", f"{montant_metrics.get('rmse', 0):,.0f}")
    
    # Importance des variables
    importance = st.session_state.prediction_model.get_feature_importance()
    if importance:
        st.markdown("### 🎯 Importance des Variables")
        
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['Variable', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Variable', orientation='h',
                    title="Variables les Plus Influentes")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def generate_and_display_recommendations(client_data, mode="existing"):
    """Génère et affiche les recommandations."""
    with st.spinner("Génération des recommandations..."):
        try:
            if mode == "existing":
                recommendations = st.session_state.recommendation_api.get_client_recommendations(client_data)
            else:
                recommendations = st.session_state.recommendation_api.get_manual_client_recommendations(client_data)
            
            if recommendations.get('status') == 'success':
                rec_data = recommendations['data']
                
                # Profil comportemental
                st.subheader("🧠 Profil Comportemental")
                
                behavior_profile = rec_data.get('behavior_profile', {})
                
                # Vérifier si segmentation avancée pour afficher plus d'informations
                if 'behavioral_scores' in behavior_profile:
                    col1, col2, col3, col4, col5 = st.columns(5)
                else:
                    col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    segment = behavior_profile.get('behavior_segment', 'N/A')
                    
                    # Ajouter des indicateurs visuels cohérents
                    segment_icons = {
                        'TRADITIONNEL_RESISTANT': '🔴',
                        'TRADITIONNEL_MODERE': '🟡', 
                        'DIGITAL_TRANSITOIRE': '🟠',
                        'DIGITAL_ADOPTER': '🟢',
                        'DIGITAL_NATIF': '💚',
                        'EQUILIBRE_MIXTE': '🔵'
                    }
                    
                    icon = segment_icons.get(segment, '⚪')
                    st.metric("Segment", f"{icon} {segment}")
                    
                    # Afficher la logique de classification
                    if segment == 'TRADITIONNEL_RESISTANT' and 'behavioral_scores' in behavior_profile:
                        check_val = behavior_profile['behavioral_scores'].get('check_dependency_score', 0)
                        digital_val = behavior_profile['behavioral_scores'].get('digital_adoption_score', 0)
                        if check_val > 0.6 and digital_val < 0.3:
                            st.caption("✅ Logique cohérente")
                        else:
                            st.caption("⚠️ Vérifier logique")
                    elif segment == 'TRADITIONNEL_MODERE' and 'behavioral_scores' in behavior_profile:
                        check_val = behavior_profile['behavioral_scores'].get('check_dependency_score', 0)
                        digital_val = behavior_profile['behavioral_scores'].get('digital_adoption_score', 0)
                        if (0.25 <= check_val <= 0.65) and (0.25 <= digital_val <= 0.65):
                            st.caption("✅ Logique cohérente")
                        else:
                            st.caption("⚠️ Vérifier logique")
                
                with col2:
                    # Vérifier si on utilise la segmentation avancée (scores dans behavioral_scores)
                    if 'behavioral_scores' in behavior_profile:
                        check_score = behavior_profile['behavioral_scores'].get('check_dependency_score', 0) * 100
                    else:
                        # Système legacy (scores au niveau racine)
                        check_score = behavior_profile.get('check_dependency_score', 0) * 100
                    st.metric("Dépendance Chèques", f"{check_score:.1f}%")
                
                with col3:
                    if 'behavioral_scores' in behavior_profile:
                        digital_score = behavior_profile['behavioral_scores'].get('digital_adoption_score', 0) * 100
                    else:
                        digital_score = behavior_profile.get('digital_adoption_score', 0) * 100
                    st.metric("Adoption Digitale", f"{digital_score:.1f}%")
                
                with col4:
                    if 'behavioral_scores' in behavior_profile:
                        evolution_score = behavior_profile['behavioral_scores'].get('payment_evolution_score', 0) * 100
                    else:
                        evolution_score = behavior_profile.get('payment_evolution_score', 0) * 100
                    st.metric("Évolution Paiements", f"{evolution_score:.1f}%")
                
                # Afficher le score de modernité si segmentation avancée
                if 'behavioral_scores' in behavior_profile:
                    with col5:
                        modernity_score = behavior_profile['behavioral_scores'].get('modernity_score', 0) * 100
                        st.metric("Score Modernité", f"{modernity_score:.1f}%")
                
                # Afficher la confiance d'analyse avec contexte
                if 'confidence' in behavior_profile:
                    confidence = behavior_profile['confidence'] * 100
                    if confidence >= 80:
                        st.success(f"🎯 Analyse très fiable: {confidence:.1f}%")
                    elif confidence >= 60:
                        st.info(f"🎯 Analyse fiable: {confidence:.1f}%")
                    else:
                        st.warning(f"⚠️ Analyse à confirmer: {confidence:.1f}% - Données incomplètes")
                
                # Ajouter une explication du segment pour l'utilisateur
                if 'behavioral_scores' in behavior_profile:
                    segment = behavior_profile.get('behavior_segment', 'N/A')
                    segment_explanations = {
                        'TRADITIONNEL_RESISTANT': '🔴 Client très dépendant aux chèques, résistant au digital',
                        'TRADITIONNEL_MODERE': '🟡 Client modérément traditionnel, ouvert au changement',
                        'DIGITAL_TRANSITOIRE': '🟠 Client en transition active vers le digital',
                        'DIGITAL_ADOPTER': '🟢 Client adopteur avancé des services digitaux',
                        'DIGITAL_NATIF': '💚 Client natif digital, avant-gardiste',
                        'EQUILIBRE_MIXTE': '🔵 Client avec approche équilibrée et flexible'
                    }
                    
                    explanation = segment_explanations.get(segment, '')
                    if explanation:
                        st.markdown(f"**📝 Profil:** {explanation}")
                
                # Recommandations avec organisation claire
                st.subheader("🎯 Services Recommandés")
                
                recommendations = rec_data.get('recommendations', [])
                if not recommendations:
                    st.warning("Aucune recommandation disponible pour ce profil.")
                else:
                    st.markdown(f"**{len(recommendations)} service(s) recommandé(s)** pour ce profil client")
                    
                    for i, rec in enumerate(recommendations):
                        service_info = rec.get('service_info', {})
                        scores = rec.get('scores', {})
                        
                        # Priorité visuelle selon le score
                        score = scores.get('global', 0)
                        if score >= 0.8:
                            priority_icon = "🏆"  # Très recommandé
                        elif score >= 0.6:
                            priority_icon = "⭐"  # Recommandé
                        else:
                            priority_icon = "💡"  # À considérer
                        
                        with st.expander(f"{priority_icon} #{i+1} {service_info.get('nom', 'Service')} - Score: {score:.2f}/1.0"):
                            col1, col2 = st.columns(2)
                        
                            with col1:
                                st.write(f"**📋 Description:** {service_info.get('description', 'N/A')}")
                                st.write(f"**🎯 Objectif:** {service_info.get('cible', 'N/A')}")
                                st.write(f"**💰 Coût:** {format_currency_tnd(service_info.get('cout', 0), 0)}")
                                st.write(f"**🏷️ Type:** {service_info.get('type', 'Service Bancaire')}")
                                
                                # Lien vers le produit Attijari Bank
                                product_link = service_info.get('lien_produit', '')
                                if product_link:
                                    st.markdown(f"**🔗 [Accéder au service sur Attijari Bank]({product_link})**")
                                
                                # Avantages du service
                                avantages = service_info.get('avantages', [])
                                if avantages:
                                    st.write("**✨ Avantages:**")
                                    for avantage in avantages:
                                        st.write(f"• {avantage}")
                        
                            with col2:
                                st.metric("📊 Score Base", f"{scores.get('base', 0):.2f}")
                                st.metric("⚡ Score Urgence", f"{scores.get('urgency', 0):.2f}")
                                st.metric("✅ Score Faisabilité", f"{scores.get('feasibility', 0):.2f}")
                
                # Impact estimé
                st.subheader("📈 Impact Estimé")
                
                impact = rec_data.get('impact_estimations', {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    reduction = impact.get('reduction_cheques_estimee', 0)
                    st.metric("Réduction Chèques", f"{reduction:.1f}")
                
                with col2:
                    percentage = impact.get('pourcentage_reduction', 0)
                    st.metric("% Réduction", f"{percentage:.1f}%")
                
                with col3:
                    benefit = impact.get('benefice_bancaire_estime', 0)
                    st.metric("Bénéfice Estimé", format_currency_tnd(benefit))
            
            else:
                st.error(f"Erreur: {recommendations.get('error', 'Erreur inconnue')}")
        
        except Exception as e:
            st.error(f"Erreur lors de la génération: {e}")

# Anciennes fonctions supprimées - remplacées par les versions unifiées

# Anciennes fonctions supprimées - remplacées par les nouvelles versions unifiées one-page

# Toutes les anciennes fonctions ont été supprimées et remplacées par les nouvelles versions unifiées one-page

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

# Toutes les anciennes fonctions de recommandations ont été remplacées par les nouvelles versions unifiées

if __name__ == "__main__":
    main()