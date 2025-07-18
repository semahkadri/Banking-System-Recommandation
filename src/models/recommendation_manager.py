# -*- coding: utf-8 -*-
"""
Gestionnaire du Système de Recommandation Personnalisée

Ce module coordonne l'ensemble du système de recommandation et fournit
une interface unifiée pour les recommandations bancaires personnalisées.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .recommendation_engine import RecommendationEngine, RecommendationTracker
from .eligibility_rules import integrate_eligibility_with_recommendations
from ..utils.data_utils import clean_numeric_data


class RecommendationManager:
    """Gestionnaire principal du système de recommandation."""
    
    def __init__(self, data_path: str = "data/processed"):
        self.data_path = Path(data_path)
        self.models_path = Path("data/models")
        self.models_path.mkdir(exist_ok=True)
        
        # Initialiser les composants
        self.recommendation_engine = RecommendationEngine(data_path)
        self.tracker = RecommendationTracker(data_path)
        
        # Chargement des données client
        self.client_data = None
        self.load_client_data()
        
        print("[RECOMMENDATION MANAGER] Système de recommandation initialisé")
    
    def load_client_data(self):
        """Charge les données client pour les recommandations."""
        
        final_dataset_path = self.data_path / "dataset_final.csv"
        
        if final_dataset_path.exists():
            try:
                self.client_data = pd.read_csv(final_dataset_path)
                print(f"[RECOMMENDATION MANAGER] Données client chargées: {len(self.client_data)} clients")
            except Exception as e:
                print(f"[RECOMMENDATION MANAGER] Erreur chargement données: {e}")
                self.client_data = None
        else:
            print("[RECOMMENDATION MANAGER] Fichier de données client non trouvé")
    
    def get_client_recommendations(self, client_id: str) -> Dict[str, Any]:
        """Obtient les recommandations pour un client spécifique avec insights avancés."""
        
        if self.client_data is None:
            return {"error": "Données client non disponibles"}
        
        # Rechercher le client
        client_row = self.client_data[self.client_data['CLI'] == client_id]
        
        if client_row.empty:
            return {"error": f"Client {client_id} non trouvé"}
        
        # Convertir en dictionnaire
        client_dict = client_row.iloc[0].to_dict()
        
        # Nettoyer les données
        cleaned_data = self._clean_client_data(client_dict)
        
        # Générer les recommandations
        recommendations = self.recommendation_engine.generate_recommendations(cleaned_data)
        
        # Ajouter des insights avancés
        recommendations['advanced_insights'] = self._generate_advanced_insights(cleaned_data, recommendations)
        
        # Ajouter l'analyse d'éligibilité bancaire
        recommendations['eligibility_analysis'] = integrate_eligibility_with_recommendations(cleaned_data)
        
        # Enregistrer pour le suivi
        self.tracker.record_recommendation(client_id, recommendations)
        
        return recommendations
    
    def get_batch_recommendations(self, client_ids: List[str] = None, 
                                 limit: int = 100) -> Dict[str, Any]:
        """Génère des recommandations pour plusieurs clients."""
        
        if self.client_data is None:
            return {"error": "Données client non disponibles"}
        
        # Sélectionner les clients
        if client_ids:
            selected_clients = self.client_data[self.client_data['CLI'].isin(client_ids)]
        else:
            selected_clients = self.client_data.head(limit)
        
        batch_recommendations = {}
        
        for _, client_row in selected_clients.iterrows():
            client_id = client_row['CLI']
            client_dict = client_row.to_dict()
            
            try:
                # Nettoyer les données
                cleaned_data = self._clean_client_data(client_dict)
                
                # Générer les recommandations
                recommendations = self.recommendation_engine.generate_recommendations(cleaned_data)
                
                batch_recommendations[client_id] = recommendations
                
                # Enregistrer pour le suivi
                self.tracker.record_recommendation(client_id, recommendations)
                
            except Exception as e:
                batch_recommendations[client_id] = {"error": str(e)}
        
        return {
            "total_clients": len(batch_recommendations),
            "clients": batch_recommendations,
            "generation_date": datetime.now().isoformat()
        }
    
    def get_segment_recommendations(self, segment: str = None, 
                                  market: str = None) -> Dict[str, Any]:
        """Génère des recommandations par segment ou marché."""
        
        try:
            if self.client_data is None:
                return {"error": "Données client non disponibles"}
            
            # Filtrer les données
            filtered_data = self.client_data.copy()
            
            # Debug info
            print(f"[DEBUG] Total clients: {len(filtered_data)}")
            print(f"[DEBUG] Segment filter: {segment}")
            print(f"[DEBUG] Market filter: {market}")
            
            if segment and segment.strip():
                print(f"[DEBUG] Unique segments available: {filtered_data['Segment_NMR_2025'].unique()}")
                filtered_data = filtered_data[filtered_data['Segment_NMR_2025'] == segment]
                print(f"[DEBUG] After segment filter: {len(filtered_data)}")
            
            if market and market.strip():
                print(f"[DEBUG] Unique markets available: {filtered_data['CLIENT_MARCHE'].unique()}")
                filtered_data = filtered_data[filtered_data['CLIENT_MARCHE'] == market]
                print(f"[DEBUG] After market filter: {len(filtered_data)}")
            
            if filtered_data.empty:
                return {"error": f"Aucun client trouvé pour les critères spécifiés. Segment: {segment}, Market: {market}"}
        
        except Exception as e:
            return {"error": f"Erreur lors du filtrage des données: {str(e)}"}
        
        # Générer des recommandations pour un échantillon
        sample_size = min(50, len(filtered_data))
        try:
            sample_clients = filtered_data.sample(n=sample_size)
            
            clients_analyzed = len(sample_clients)
            segment_recommendations = {}
            recommendation_summary = {}
            
            for _, client_row in sample_clients.iterrows():
                client_id = client_row['CLI']
                client_dict = client_row.to_dict()
                
                try:
                    # Nettoyer les données
                    cleaned_data = self._clean_client_data(client_dict)
                    
                    # Générer les recommandations
                    recommendations = self.recommendation_engine.generate_recommendations(cleaned_data)
                    
                    segment_recommendations[client_id] = recommendations
                    
                    # Agrégation pour le résumé
                    behavior_segment = recommendations.get('behavior_profile', {}).get('behavior_segment', 'UNKNOWN')
                    if behavior_segment not in recommendation_summary:
                        recommendation_summary[behavior_segment] = {
                            'count': 0,
                            'common_services': {},
                            'avg_impact': 0
                        }
                    
                    recommendation_summary[behavior_segment]['count'] += 1
                    
                    # Services recommandés
                    for rec in recommendations.get('recommendations', []):
                        service_id = rec['service_id']
                        if service_id not in recommendation_summary[behavior_segment]['common_services']:
                            recommendation_summary[behavior_segment]['common_services'][service_id] = 0
                        recommendation_summary[behavior_segment]['common_services'][service_id] += 1
                    
                    # Impact moyen
                    impact = recommendations.get('impact_estimations', {}).get('pourcentage_reduction', 0)
                    recommendation_summary[behavior_segment]['avg_impact'] += impact
                    
                except Exception as e:
                    segment_recommendations[client_id] = {"error": str(e)}
            
            # Calculer les moyennes
            for segment_data in recommendation_summary.values():
                if segment_data['count'] > 0:
                    segment_data['avg_impact'] /= segment_data['count']
            
            return {
                "filter_criteria": {"segment": segment, "market": market},
                "total_clients_analyzed": clients_analyzed,
                "segment_summary": recommendation_summary,
                "individual_recommendations": segment_recommendations,
                "analysis_date": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Erreur lors de l'analyse du segment: {str(e)}"}
    
    def record_service_adoption(self, client_id: str, service_id: str, 
                               adoption_date: str = None) -> Dict[str, Any]:
        """Enregistre l'adoption d'un service par un client."""
        
        try:
            self.tracker.record_adoption(client_id, service_id, adoption_date)
            return {
                "status": "success",
                "message": f"Adoption du service {service_id} enregistrée pour le client {client_id}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_adoption_statistics(self, period_days: int = 30) -> Dict[str, Any]:
        """Obtient les statistiques d'adoption des services."""
        
        return self.tracker.calculate_adoption_rate(period_days)
    
    def get_effectiveness_report(self) -> Dict[str, Any]:
        """Génère un rapport d'efficacité complet."""
        
        return self.tracker.generate_effectiveness_report()
    
    def get_client_profile_analysis(self, client_id: str) -> Dict[str, Any]:
        """Analyse détaillée du profil d'un client."""
        
        if self.client_data is None:
            return {"error": "Données client non disponibles"}
        
        # Rechercher le client
        client_row = self.client_data[self.client_data['CLI'] == client_id]
        
        if client_row.empty:
            return {"error": f"Client {client_id} non trouvé"}
        
        client_dict = client_row.iloc[0].to_dict()
        cleaned_data = self._clean_client_data(client_dict)
        
        # Analyse comportementale
        behavior_profile = self.recommendation_engine.behavior_analyzer.analyze_client_behavior(cleaned_data)
        
        # Comparaison avec les pairs
        peer_comparison = self._compare_with_peers(cleaned_data)
        
        # Évolution potentielle
        evolution_analysis = self._analyze_evolution_potential(cleaned_data)
        
        return {
            "client_id": client_id,
            "basic_info": {
                "marche": cleaned_data.get('CLIENT_MARCHE', 'Unknown'),
                "segment": cleaned_data.get('Segment_NMR', 'Unknown'),
                "csp": cleaned_data.get('CSP', 'Unknown'),
                "revenu_estime": cleaned_data.get('Revenu_Estime', 0),
                "mobile_banking": cleaned_data.get('Utilise_Mobile_Banking', 0)
            },
            "behavior_profile": behavior_profile,
            "peer_comparison": peer_comparison,
            "evolution_analysis": evolution_analysis,
            "analysis_date": datetime.now().isoformat()
        }
    
    def _clean_client_data(self, client_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Nettoie et standardise les données client."""
        
        cleaned_data = {}
        
        # Mappage des colonnes
        column_mapping = {
            'CLI': 'CLI',
            'CLIENT_MARCHE': 'CLIENT_MARCHE',
            'CSP': 'CSP',
            'Segment_NMR_2025': 'Segment_NMR',  # Utiliser la colonne correcte
            'Revenu_Estime': 'Revenu_Estime',
            'Nbr_Cheques_2024': 'Nbr_Cheques_2024',
            'Montant_Max_2024': 'Montant_Max_2024',
            'Ecart_Nbr_Cheques_2024_2025': 'Ecart_Nbr_Cheques_2024_2025',
            'Ecart_Montant_Max_2024_2025': 'Ecart_Montant_Max_2024_2025',
            'A_Demande_Derogation': 'A_Demande_Derogation',
            'Ratio_Cheques_Paiements_2025': 'Ratio_Cheques_Paiements',
            'Utilise_Mobile_Banking': 'Utilise_Mobile_Banking',
            'Nombre_Methodes_Paiement': 'Nombre_Methodes_Paiement',
            'Montant_Moyen_Cheque': 'Montant_Moyen_Cheque',
            'Montant_Moyen_Alternative': 'Montant_Moyen_Alternative',
            'Nbr_Transactions_2025': 'Nbr_Transactions_2025'
        }
        
        for original_key, clean_key in column_mapping.items():
            if original_key in client_dict:
                if clean_key in ['Revenu_Estime', 'Nbr_Cheques_2024', 'Montant_Max_2024', 
                               'Ecart_Nbr_Cheques_2024_2025', 'Ecart_Montant_Max_2024_2025',
                               'Nombre_Methodes_Paiement', 'Montant_Moyen_Cheque', 
                               'Montant_Moyen_Alternative', 'Nbr_Transactions_2025']:
                    cleaned_data[clean_key] = clean_numeric_data(client_dict[original_key])
                else:
                    cleaned_data[clean_key] = client_dict[original_key]
        
        # Valeurs par défaut
        defaults = {
            'CLI': 'unknown',
            'CLIENT_MARCHE': 'Particuliers',
            'CSP': 'Unknown',
            'Segment_NMR': 'S3 Essentiel',
            'Revenu_Estime': 30000,
            'Nbr_Cheques_2024': 0,
            'Montant_Max_2024': 0,
            'Ecart_Nbr_Cheques_2024_2025': 0,
            'Ecart_Montant_Max_2024_2025': 0,
            'A_Demande_Derogation': 0,
            'Ratio_Cheques_Paiements': 0,
            'Utilise_Mobile_Banking': 0,
            'Nombre_Methodes_Paiement': 1,
            'Montant_Moyen_Cheque': 0,
            'Montant_Moyen_Alternative': 0,
            'Nbr_Transactions_2025': 1
        }
        
        for key, default_value in defaults.items():
            if key not in cleaned_data:
                cleaned_data[key] = default_value
        
        return cleaned_data
    
    def _compare_with_peers(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare un client avec ses pairs (même segment/marché)."""
        
        if self.client_data is None:
            return {"error": "Données non disponibles"}
        
        client_segment = client_data.get('Segment_NMR', 'Unknown')
        client_market = client_data.get('CLIENT_MARCHE', 'Unknown')
        
        # Filtrer les pairs
        peers = self.client_data[
            (self.client_data['Segment_NMR_2025'] == client_segment) &
            (self.client_data['CLIENT_MARCHE'] == client_market)
        ]
        
        if len(peers) < 2:
            return {"error": "Pas assez de pairs pour la comparaison"}
        
        # Métriques de comparaison
        metrics = {
            'Nbr_Cheques_2024': client_data.get('Nbr_Cheques_2024', 0),
            'Revenu_Estime': client_data.get('Revenu_Estime', 0),
            'Montant_Max_2024': client_data.get('Montant_Max_2024', 0),
            'Utilise_Mobile_Banking': client_data.get('Utilise_Mobile_Banking', 0)
        }
        
        peer_stats = {}
        for metric, client_value in metrics.items():
            if metric in peers.columns:
                peer_values = peers[metric].dropna()
                if len(peer_values) > 0:
                    peer_stats[metric] = {
                        'client_value': client_value,
                        'peer_mean': float(peer_values.mean()),
                        'peer_median': float(peer_values.median()),
                        'percentile_rank': float((peer_values < client_value).sum() / len(peer_values) * 100)
                    }
        
        return {
            'segment': client_segment,
            'market': client_market,
            'peer_count': len(peers),
            'metric_comparisons': peer_stats
        }
    
    def _analyze_evolution_potential(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse le potentiel d'évolution du client."""
        
        current_checks = client_data.get('Nbr_Cheques_2024', 0)
        mobile_banking = client_data.get('Utilise_Mobile_Banking', 0)
        check_trend = client_data.get('Ecart_Nbr_Cheques_2024_2025', 0)
        
        # Potentiel de réduction des chèques
        if current_checks > 10:
            reduction_potential = "ÉLEVÉ"
        elif current_checks > 5:
            reduction_potential = "MOYEN"
        else:
            reduction_potential = "FAIBLE"
        
        # Potentiel d'adoption digitale
        if mobile_banking:
            digital_potential = "EXPANSION"
        else:
            digital_potential = "ADOPTION"
        
        # Tendance actuelle
        if check_trend < -2:
            trend = "RÉDUCTION_RAPIDE"
        elif check_trend < 0:
            trend = "RÉDUCTION_PROGRESSIVE"
        elif check_trend > 2:
            trend = "AUGMENTATION"
        else:
            trend = "STABLE"
        
        return {
            'reduction_potential': reduction_potential,
            'digital_potential': digital_potential,
            'current_trend': trend,
            'recommendation_priority': self._calculate_priority(reduction_potential, digital_potential, trend)
        }
    
    def _calculate_priority(self, reduction_potential: str, digital_potential: str, trend: str) -> str:
        """Calcule la priorité d'intervention."""
        
        # Logique de priorisation
        if reduction_potential == "ÉLEVÉ" and trend == "AUGMENTATION":
            return "URGENT"
        elif reduction_potential == "ÉLEVÉ" and digital_potential == "ADOPTION":
            return "HAUTE"
        elif reduction_potential == "MOYEN" and trend != "RÉDUCTION_RAPIDE":
            return "MOYENNE"
        else:
            return "BASSE"
    
    def export_recommendations(self, client_ids: List[str] = None, 
                             format: str = "json") -> str:
        """Exporte les recommandations dans différents formats."""
        
        # Générer les recommandations
        if client_ids:
            recommendations = self.get_batch_recommendations(client_ids)
        else:
            recommendations = self.get_batch_recommendations(limit=100)
        
        # Timestamp pour le fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            filename = f"recommendations_export_{timestamp}.json"
            filepath = self.data_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(recommendations, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "csv":
            filename = f"recommendations_export_{timestamp}.csv"
            filepath = self.data_path / filename
            
            # Aplatir les données pour CSV
            flattened_data = []
            for client_id, rec_data in recommendations.get('recommendations', {}).items():
                if 'error' not in rec_data:
                    base_row = {
                        'client_id': client_id,
                        'behavior_segment': rec_data.get('behavior_profile', {}).get('behavior_segment', ''),
                        'total_recommendations': len(rec_data.get('recommendations', [])),
                        'estimated_reduction': rec_data.get('impact_estimations', {}).get('pourcentage_reduction', 0)
                    }
                    
                    # Ajouter les recommandations
                    for i, rec in enumerate(rec_data.get('recommendations', [])[:3]):  # Top 3
                        base_row[f'recommendation_{i+1}'] = rec.get('service_id', '')
                        base_row[f'score_{i+1}'] = rec.get('scores', {}).get('global', 0)
                    
                    flattened_data.append(base_row)
            
            # Créer le DataFrame et sauvegarder
            df = pd.DataFrame(flattened_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
        
        return str(filepath)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtient le statut du système de recommandation."""
        
        return {
            "system_status": "OPERATIONAL",
            "client_data_loaded": self.client_data is not None,
            "total_clients": len(self.client_data) if self.client_data is not None else 0,
            "recommendation_engine_status": "READY",
            "tracker_status": "READY",
            "last_update": datetime.now().isoformat()
        }
    
    def _generate_advanced_insights(self, client_data: Dict[str, Any], 
                                  recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des insights avancés pour un client."""
        
        # Analyse de la trajectoire du client
        current_checks = client_data.get('Nbr_Cheques_2024', 0)
        check_evolution = client_data.get('Ecart_Nbr_Cheques_2024_2025', 0)
        digital_adoption = client_data.get('Utilise_Mobile_Banking', 0)
        revenue = client_data.get('Revenu_Estime', 0)
        segment = client_data.get('Segment_NMR', 'Unknown')
        
        # Insights comportementaux
        behavioral_insights = []
        
        if current_checks > 10:
            behavioral_insights.append("Client avec usage intensif des chèques - Potentiel de réduction élevé")
        elif current_checks > 5:
            behavioral_insights.append("Usage modéré des chèques - Transition progressive recommandée")
        else:
            behavioral_insights.append("Faible usage des chèques - Maintenir les bonnes pratiques")
        
        if check_evolution > 2:
            behavioral_insights.append("⚠️ Augmentation des chèques détectée - Intervention prioritaire")
        elif check_evolution < -2:
            behavioral_insights.append("✅ Réduction des chèques en cours - Accompagner la transition")
        
        if digital_adoption:
            behavioral_insights.append("📱 Utilisateur mobile banking - Prêt pour services digitaux avancés")
        else:
            behavioral_insights.append("📋 Non-utilisateur mobile banking - Formation et accompagnement nécessaires")
        
        # Prédictions d'évolution
        evolution_predictions = self._predict_client_evolution(client_data)
        
        # Recommandations spécifiques par segment
        segment_specific_insights = self._get_segment_specific_insights(segment, client_data)
        
        # Calcul du potentiel de valeur
        value_potential = self._calculate_value_potential(client_data, recommendations)
        
        return {
            'behavioral_insights': behavioral_insights,
            'evolution_predictions': evolution_predictions,
            'segment_specific_insights': segment_specific_insights,
            'value_potential': value_potential,
            'recommended_actions': self._generate_action_plan(client_data, recommendations),
            'success_indicators': self._define_success_indicators(client_data, recommendations)
        }
    
    def _predict_client_evolution(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prédit l'évolution du client dans les 6-12 mois."""
        
        current_checks = client_data.get('Nbr_Cheques_2024', 0)
        check_evolution = client_data.get('Ecart_Nbr_Cheques_2024_2025', 0)
        digital_adoption = client_data.get('Utilise_Mobile_Banking', 0)
        
        # Prédictions basées sur les tendances
        predictions = {
            'trajectory': 'stable',
            'check_usage_6m': current_checks,
            'check_usage_12m': current_checks,
            'digital_readiness': 'medium',
            'intervention_urgency': 'low'
        }
        
        # Ajustements basés sur l'évolution
        if check_evolution > 3:
            predictions['trajectory'] = 'deteriorating'
            predictions['check_usage_6m'] = current_checks + 2
            predictions['check_usage_12m'] = current_checks + 4
            predictions['intervention_urgency'] = 'high'
        elif check_evolution < -3:
            predictions['trajectory'] = 'improving'
            predictions['check_usage_6m'] = max(0, current_checks - 2)
            predictions['check_usage_12m'] = max(0, current_checks - 3)
            predictions['intervention_urgency'] = 'low'
        
        # Ajustements basés sur l'adoption digitale
        if digital_adoption:
            predictions['digital_readiness'] = 'high'
            predictions['check_usage_6m'] = max(0, predictions['check_usage_6m'] - 1)
            predictions['check_usage_12m'] = max(0, predictions['check_usage_12m'] - 2)
        
        return predictions
    
    def _get_segment_specific_insights(self, segment: str, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fournit des insights spécifiques au segment du client."""
        
        segment_insights = {
            'S1 Excellence': {
                'description': 'Client premium avec fort potentiel de revenus',
                'priority_services': ['services_premium', 'carte_sans_contact'],
                'approach': 'Approche consultative avec services haut de gamme'
            },
            'S2 Premium': {
                'description': 'Client à fort potentiel avec besoins sophistiqués',
                'priority_services': ['mobile_banking', 'services_premium'],
                'approach': 'Accompagnement personnalisé vers le digital'
            },
            'S3 Essentiel': {
                'description': 'Client core avec besoins standards',
                'priority_services': ['carte_bancaire', 'mobile_banking'],
                'approach': 'Transition progressive vers les services modernes'
            },
            'S4 Avenir': {
                'description': 'Client jeune avec potentiel de croissance',
                'priority_services': ['mobile_banking', 'paiement_mobile'],
                'approach': 'Éducation financière et services digitaux'
            },
            'S5 Univers': {
                'description': 'Client avec besoins spécifiques',
                'priority_services': ['formation_digital', 'accompagnement_personnel'],
                'approach': 'Accompagnement adapté et formation'
            }
        }
        
        return segment_insights.get(segment, {
            'description': 'Segment non identifié',
            'priority_services': ['carte_bancaire', 'mobile_banking'],
            'approach': 'Approche standard avec évaluation approfondie'
        })
    
    def _calculate_value_potential(self, client_data: Dict[str, Any], 
                                 recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule le potentiel de valeur du client."""
        
        revenue = client_data.get('Revenu_Estime', 0)
        current_checks = client_data.get('Nbr_Cheques_2024', 0)
        impact_estimations = recommendations.get('impact_estimations', {})
        
        # Calcul du potentiel de valeur
        check_cost_savings = impact_estimations.get('economies_operationnelles', 0)
        service_revenues = impact_estimations.get('revenus_additionnels', 0)
        
        # Potentiel de cross-selling basé sur le profil
        cross_sell_potential = min(revenue / 100000, 1.0) * 500  # TND
        
        # Potentiel de rétention
        retention_value = min(revenue / 50000, 2.0) * 200  # TND
        
        total_value = check_cost_savings + service_revenues + cross_sell_potential + retention_value
        
        return {
            'total_value_potential': total_value,
            'cost_savings': check_cost_savings,
            'service_revenues': service_revenues,
            'cross_sell_potential': cross_sell_potential,
            'retention_value': retention_value,
            'value_category': self._categorize_value_potential(total_value)
        }
    
    def _categorize_value_potential(self, value: float) -> str:
        """Catégorise le potentiel de valeur."""
        if value > 1000:
            return 'HIGH'
        elif value > 500:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_action_plan(self, client_data: Dict[str, Any], 
                            recommendations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Génère un plan d'action détaillé."""
        
        actions = []
        behavior_segment = recommendations.get('behavior_profile', {}).get('behavior_segment', 'EQUILIBRE')
        
        # Actions basées sur le segment
        if behavior_segment == 'TRADITIONNEL_RESISTANT':
            actions.extend([
                {
                    'action': 'Rencontrer le client pour évaluer ses préoccupations',
                    'timeline': '1-2 semaines',
                    'priority': 'HIGH'
                },
                {
                    'action': 'Proposer une formation personnalisée aux alternatives',
                    'timeline': '2-4 semaines',
                    'priority': 'MEDIUM'
                }
            ])
        elif behavior_segment == 'DIGITAL_TRANSITOIRE':
            actions.extend([
                {
                    'action': 'Proposer des services digitaux complémentaires',
                    'timeline': '1 semaine',
                    'priority': 'HIGH'
                },
                {
                    'action': 'Suivre l\'adoption et optimiser l\'expérience',
                    'timeline': '1-3 mois',
                    'priority': 'MEDIUM'
                }
            ])
        
        # Actions génériques
        actions.append({
            'action': 'Suivre l\'évolution mensuelle des habitudes de paiement',
            'timeline': 'Continue',
            'priority': 'LOW'
        })
        
        return actions
    
    def _define_success_indicators(self, client_data: Dict[str, Any], 
                                 recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Définit les indicateurs de succès."""
        
        current_checks = client_data.get('Nbr_Cheques_2024', 0)
        
        return {
            'primary_kpis': {
                'check_reduction_target': f"Réduction de {current_checks * 0.3:.0f} chèques (30%)",
                'service_adoption_target': "Adoption d'au moins 1 service recommandé",
                'timeline': "6 mois"
            },
            'secondary_kpis': {
                'digital_engagement': "Utilisation active des services digitaux",
                'satisfaction_score': "Score de satisfaction > 8/10",
                'retention_rate': "Maintien de la relation client"
            },
            'measurement_plan': {
                'frequency': 'Mensuelle',
                'metrics': ['Nombre de chèques', 'Adoption services', 'Utilisation digitale'],
                'review_points': ['1 mois', '3 mois', '6 mois']
            }
        }