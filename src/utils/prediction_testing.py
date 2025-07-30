# -*- coding: utf-8 -*-
"""
Système de test et validation des prédictions

Ce module permet de tester les prédictions avec de vrais clients du dataset
et de valider la précision des modèles.
"""

import pandas as pd
import random
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json

class PredictionTestingSystem:
    """Système de test des prédictions avec vrais clients."""
    
    def __init__(self, dataset_path: str = "data/processed/dataset_final.csv"):
        """Initialise le système de test."""
        self.dataset_path = Path(dataset_path)
        self.test_clients = []
        self.validation_results = {}
        
        # Seuils d'acceptabilité pour validation
        self.acceptability_thresholds = {
            'nbr_cheques': {
                'excellent': 0.1,    # ±10% = excellent
                'bon': 0.25,         # ±25% = bon  
                'acceptable': 0.5,   # ±50% = acceptable
                'mediocre': 1.0      # ±100% = médiocre
            },
            'montant_max': {
                'excellent': 0.15,   # ±15% = excellent
                'bon': 0.3,          # ±30% = bon
                'acceptable': 0.6,   # ±60% = acceptable
                'mediocre': 1.2      # ±120% = médiocre
            }
        }
        
        # Charger les clients de test
        self._load_test_clients()
    
    def _load_test_clients(self):
        """Charge les clients du dataset pour les tests."""
        try:
            if self.dataset_path.exists():
                df = pd.read_csv(self.dataset_path)
                
                # Sélectionner un échantillon représentatif
                if len(df) > 50:
                    # Stratification par segment et marché
                    sample_clients = []
                    
                    # Par segment NMR
                    for segment in df['Segment_NMR'].unique():
                        segment_data = df[df['Segment_NMR'] == segment]
                        n_samples = min(5, len(segment_data))
                        if n_samples > 0:
                            segment_sample = segment_data.sample(n=n_samples, random_state=42)
                            sample_clients.append(segment_sample)
                    
                    # Par marché client
                    for marche in df['CLIENT_MARCHE'].unique():
                        marche_data = df[df['CLIENT_MARCHE'] == marche]
                        n_samples = min(3, len(marche_data))
                        if n_samples > 0:
                            marche_sample = marche_data.sample(n=n_samples, random_state=42)
                            sample_clients.append(marche_sample)
                    
                    # Combiner les échantillons
                    if sample_clients:
                        combined_df = pd.concat(sample_clients).drop_duplicates(subset=['CLI'])
                        self.test_clients = combined_df.to_dict('records')
                    else:
                        # Fallback: échantillon aléatoire
                        sample_df = df.sample(n=min(30, len(df)), random_state=42)
                        self.test_clients = sample_df.to_dict('records')
                else:
                    # Dataset petit, prendre tous les clients
                    self.test_clients = df.to_dict('records')
                
                print(f"[TEST SYSTEM] Chargé {len(self.test_clients)} clients de test depuis le dataset")
                
            else:
                print(f"[TEST SYSTEM] Dataset non trouvé: {self.dataset_path}")
                # Créer des clients de test fictifs pour démonstration
                self._create_demo_clients()
                
        except Exception as e:
            print(f"[TEST SYSTEM] Erreur lors du chargement: {e}")
            self._create_demo_clients()
    
    def _create_demo_clients(self):
        """Crée des clients de test fictifs pour démonstration."""
        self.test_clients = [
            {
                'CLI': 'TEST_001',
                'CLIENT_MARCHE': 'Particuliers',
                'Segment_NMR': 'S3 Essentiel',
                'Revenu_Estime': 45000,
                'Nbr_Cheques_2024': 8,
                'Montant_Max_2024': 12000,
                'Target_Nbr_Cheques_Futur': 6,
                'Target_Montant_Max_Futur': 15000,
                'Utilise_Mobile_Banking': 0,
                'A_Demande_Derogation': 0,
                'Nombre_Methodes_Paiement': 3,
                'Montant_Moyen_Cheque': 2800,
                'Montant_Moyen_Alternative': 450,
                'Ratio_Cheques_Paiements': 0.35,
                'Ecart_Nbr_Cheques_2024_2025': -2,
                'Ecart_Montant_Max_2024_2025': 3000
            },
            {
                'CLI': 'TEST_002',
                'CLIENT_MARCHE': 'PME',
                'Segment_NMR': 'S2 Premium',
                'Revenu_Estime': 120000,
                'Nbr_Cheques_2024': 25,
                'Montant_Max_2024': 45000,
                'Target_Nbr_Cheques_Futur': 22,
                'Target_Montant_Max_Futur': 50000,
                'Utilise_Mobile_Banking': 1,
                'A_Demande_Derogation': 1,
                'Nombre_Methodes_Paiement': 5,
                'Montant_Moyen_Cheque': 8500,
                'Montant_Moyen_Alternative': 1200,
                'Ratio_Cheques_Paiements': 0.6,
                'Ecart_Nbr_Cheques_2024_2025': -3,
                'Ecart_Montant_Max_2024_2025': 5000
            },
            {
                'CLI': 'TEST_003',
                'CLIENT_MARCHE': 'TPE',
                'Segment_NMR': 'S4 Avenir',
                'Revenu_Estime': 32000,
                'Nbr_Cheques_2024': 3,
                'Montant_Max_2024': 8500,
                'Target_Nbr_Cheques_Futur': 4,
                'Target_Montant_Max_Futur': 9000,
                'Utilise_Mobile_Banking': 1,
                'A_Demande_Derogation': 0,
                'Nombre_Methodes_Paiement': 4,
                'Montant_Moyen_Cheque': 3200,
                'Montant_Moyen_Alternative': 280,
                'Ratio_Cheques_Paiements': 0.15,
                'Ecart_Nbr_Cheques_2024_2025': 1,
                'Ecart_Montant_Max_2024_2025': 500
            }
        ]
        print(f"[TEST SYSTEM] Créé {len(self.test_clients)} clients de test fictifs")
    
    def get_random_test_client(self) -> Dict[str, Any]:
        """Récupère un client de test aléatoire."""
        if self.test_clients:
            return random.choice(self.test_clients)
        return {}
    
    def get_test_client_by_profile(self, profile_type: str = "mixed") -> Dict[str, Any]:
        """Récupère un client de test selon un profil spécifique."""
        if not self.test_clients:
            return {}
        
        if profile_type == "digital":
            # Chercher client avec mobile banking
            digital_clients = [c for c in self.test_clients if c.get('Utilise_Mobile_Banking', 0) == 1]
            if digital_clients:
                return random.choice(digital_clients)
        
        elif profile_type == "traditional":
            # Chercher client sans mobile banking, beaucoup de chèques
            traditional_clients = [c for c in self.test_clients 
                                 if c.get('Utilise_Mobile_Banking', 0) == 0 and 
                                    c.get('Nbr_Cheques_2024', 0) > 10]
            if traditional_clients:
                return random.choice(traditional_clients)
        
        elif profile_type == "premium":
            # Chercher client segment premium
            premium_clients = [c for c in self.test_clients 
                             if c.get('Segment_NMR', '') in ['S1 Excellence', 'S2 Premium']]
            if premium_clients:
                return random.choice(premium_clients)
        
        elif profile_type == "enterprise":
            # Chercher client entreprise
            enterprise_clients = [c for c in self.test_clients 
                                if c.get('CLIENT_MARCHE', '') in ['PME', 'GEI', 'TRE']]
            if enterprise_clients:
                return random.choice(enterprise_clients)
        
        # Retour par défaut: client aléatoire
        return random.choice(self.test_clients)
    
    def get_all_test_clients(self) -> List[Dict[str, Any]]:
        """Récupère tous les clients de test disponibles."""
        return self.test_clients.copy()
    
    def validate_prediction_accuracy(self, predicted: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
        """Valide la précision d'une prédiction par rapport aux valeurs réelles."""
        
        # Extraire les valeurs
        pred_nbr = predicted.get('predicted_nbr_cheques', 0)
        pred_montant = predicted.get('predicted_montant_max', 0)
        
        actual_nbr = actual.get('Target_Nbr_Cheques_Futur', actual.get('Nbr_Cheques_2024', 0))
        actual_montant = actual.get('Target_Montant_Max_Futur', actual.get('Montant_Max_2024', 0))
        
        # Calculer les écarts
        nbr_accuracy = self._calculate_accuracy_metrics(pred_nbr, actual_nbr, 'nbr_cheques')
        montant_accuracy = self._calculate_accuracy_metrics(pred_montant, actual_montant, 'montant_max')
        
        # Évaluation globale
        overall_score = (nbr_accuracy['score'] + montant_accuracy['score']) / 2
        overall_level = self._get_accuracy_level(overall_score)
        
        return {
            'nbr_cheques_validation': nbr_accuracy,
            'montant_max_validation': montant_accuracy,
            'overall_accuracy': {
                'score': overall_score,
                'level': overall_level,
                'interpretation': self._interpret_overall_accuracy(overall_score)
            },
            'detailed_comparison': {
                'nbr_cheques': {
                    'predicted': pred_nbr,
                    'actual': actual_nbr,
                    'difference': pred_nbr - actual_nbr,
                    'percentage_error': nbr_accuracy.get('percentage_error', 0)
                },
                'montant_max': {
                    'predicted': pred_montant,
                    'actual': actual_montant,
                    'difference': pred_montant - actual_montant,
                    'percentage_error': montant_accuracy.get('percentage_error', 0)
                }
            }
        }
    
    def _calculate_accuracy_metrics(self, predicted: float, actual: float, metric_type: str) -> Dict[str, Any]:
        """Calcule les métriques de précision pour une prédiction."""
        
        # Éviter division par zéro
        if actual == 0:
            if predicted == 0:
                return {
                    'score': 1.0,
                    'level': 'EXCELLENT',
                    'status': '✅',
                    'percentage_error': 0,
                    'interpretation': 'Prédiction parfaite (0 = 0)'
                }
            else:
                return {
                    'score': 0.0,
                    'level': 'MÉDIOCRE',
                    'status': '❌',
                    'percentage_error': float('inf'),
                    'interpretation': f'Prédit {predicted} mais réel était 0'
                }
        
        # Calculer l'erreur relative
        percentage_error = abs(predicted - actual) / actual
        
        # Déterminer le niveau selon les seuils
        thresholds = self.acceptability_thresholds[metric_type]
        
        if percentage_error <= thresholds['excellent']:
            level = 'EXCELLENT'
            status = '✅'
            score = 1.0
        elif percentage_error <= thresholds['bon']:
            level = 'BON'
            status = '✅'
            score = 0.8
        elif percentage_error <= thresholds['acceptable']:
            level = 'ACCEPTABLE'
            status = '⚠️'
            score = 0.6
        elif percentage_error <= thresholds['mediocre']:
            level = 'MÉDIOCRE'
            status = '❌'
            score = 0.3
        else:
            level = 'INACCEPTABLE'
            status = '❌'
            score = 0.0
        
        return {
            'score': score,
            'level': level,
            'status': status,
            'percentage_error': percentage_error,
            'interpretation': self._interpret_accuracy(percentage_error, level, predicted, actual)
        }
    
    def _interpret_accuracy(self, percentage_error: float, level: str, predicted: float, actual: float) -> str:
        """Génère une interprétation textuelle de la précision."""
        error_pct = percentage_error * 100
        
        if level == 'EXCELLENT':
            return f"Prédiction très précise (écart: {error_pct:.1f}%)"
        elif level == 'BON':
            return f"Bonne prédiction (écart: {error_pct:.1f}%)"
        elif level == 'ACCEPTABLE':
            return f"Prédiction acceptable (écart: {error_pct:.1f}%)"
        elif level == 'MÉDIOCRE':
            return f"Prédiction imprécise (écart: {error_pct:.1f}%)"
        else:
            return f"Prédiction très imprécise (écart: {error_pct:.1f}%)"
    
    def _get_accuracy_level(self, score: float) -> str:
        """Convertit un score en niveau d'exactitude."""
        if score >= 0.9:
            return "EXCELLENT"
        elif score >= 0.7:
            return "BON"
        elif score >= 0.5:
            return "ACCEPTABLE"
        elif score >= 0.3:
            return "MÉDIOCRE"
        else:
            return "INACCEPTABLE"
    
    def _interpret_overall_accuracy(self, score: float) -> str:
        """Interprète le score global de précision."""
        if score >= 0.9:
            return "🎯 Prédictions très fiables, modèle performant"
        elif score >= 0.7:
            return "✅ Bonnes prédictions, modèle fiable"
        elif score >= 0.5:
            return "⚠️ Prédictions acceptables, amélioration possible"
        elif score >= 0.3:
            return "❌ Prédictions imprécises, révision nécessaire"
        else:
            return "🚫 Prédictions très imprécises, modèle à revoir"
    
    def generate_test_summary(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Génère un résumé des tests effectués."""
        if not test_results:
            return {'status': 'error', 'message': 'Aucun test effectué'}
        
        # Statistiques globales
        total_tests = len(test_results)
        nbr_scores = [r['nbr_cheques_validation']['score'] for r in test_results]
        montant_scores = [r['montant_max_validation']['score'] for r in test_results]
        overall_scores = [r['overall_accuracy']['score'] for r in test_results]
        
        # Moyennes
        avg_nbr_score = sum(nbr_scores) / len(nbr_scores)
        avg_montant_score = sum(montant_scores) / len(montant_scores)
        avg_overall_score = sum(overall_scores) / len(overall_scores)
        
        # Distribution des niveaux
        level_distribution = {}
        for result in test_results:
            level = result['overall_accuracy']['level']
            level_distribution[level] = level_distribution.get(level, 0) + 1
        
        # Pourcentages de réussite
        excellent_count = sum(1 for s in overall_scores if s >= 0.9)
        good_count = sum(1 for s in overall_scores if s >= 0.7)
        acceptable_count = sum(1 for s in overall_scores if s >= 0.5)
        
        return {
            'total_tests': total_tests,
            'average_scores': {
                'nbr_cheques': avg_nbr_score,
                'montant_max': avg_montant_score,
                'overall': avg_overall_score
            },
            'success_rates': {
                'excellent': (excellent_count / total_tests) * 100,
                'good_or_better': (good_count / total_tests) * 100,
                'acceptable_or_better': (acceptable_count / total_tests) * 100
            },
            'level_distribution': level_distribution,
            'model_assessment': self._assess_model_performance(avg_overall_score),
            'recommendations': self._generate_improvement_recommendations(test_results)
        }
    
    def _assess_model_performance(self, avg_score: float) -> str:
        """Évalue la performance globale du modèle."""
        if avg_score >= 0.8:
            return "🏆 MODÈLE PERFORMANT - Prédictions très fiables"
        elif avg_score >= 0.65:
            return "✅ MODÈLE CORRECT - Bonnes prédictions générales"
        elif avg_score >= 0.45:
            return "⚠️ MODÈLE MOYEN - Améliorations recommandées"
        else:
            return "❌ MODÈLE FAIBLE - Révision complète nécessaire"
    
    def _generate_improvement_recommendations(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """Génère des recommandations d'amélioration basées sur les tests."""
        recommendations = []
        
        # Analyser les erreurs communes
        nbr_errors = [r['detailed_comparison']['nbr_cheques']['percentage_error'] for r in test_results]
        montant_errors = [r['detailed_comparison']['montant_max']['percentage_error'] for r in test_results]
        
        avg_nbr_error = sum(nbr_errors) / len(nbr_errors)
        avg_montant_error = sum(montant_errors) / len(montant_errors)
        
        if avg_nbr_error > 0.4:
            recommendations.append("📊 Améliorer la prédiction du nombre de chèques (erreur moyenne >40%)")
        
        if avg_montant_error > 0.5:
            recommendations.append("💰 Améliorer la prédiction des montants (erreur moyenne >50%)")
        
        # Analyser les patterns d'erreur
        high_error_cases = [r for r in test_results if r['overall_accuracy']['score'] < 0.3]
        
        if len(high_error_cases) > len(test_results) * 0.2:
            recommendations.append("🎯 Plus de 20% des cas ont de mauvaises prédictions - revoir les features")
        
        # Recommandations spécifiques
        digital_clients_errors = [r for r in test_results 
                                if r.get('client_profile', {}).get('Utilise_Mobile_Banking', 0) == 1 
                                and r['overall_accuracy']['score'] < 0.5]
        
        if len(digital_clients_errors) > 0:
            recommendations.append("📱 Améliorer les prédictions pour les clients digitaux")
        
        if not recommendations:
            recommendations.append("🎉 Modèle performant - continuer le monitoring")
        
        return recommendations
    
    def get_client_display_info(self, client_data: Dict[str, Any]) -> Dict[str, str]:
        """Génère les informations d'affichage d'un client."""
        return {
            'id': str(client_data.get('CLI', 'N/A')),
            'marche': str(client_data.get('CLIENT_MARCHE', 'N/A')),
            'segment': str(client_data.get('Segment_NMR', 'N/A')),
            'revenu': f"{client_data.get('Revenu_Estime', 0):,.0f} TND",
            'cheques_2024': str(client_data.get('Nbr_Cheques_2024', 0)),
            'montant_max_2024': f"{client_data.get('Montant_Max_2024', 0):,.0f} TND",
            'mobile_banking': "Oui" if client_data.get('Utilise_Mobile_Banking', 0) else "Non",
            'profil': self._determine_client_profile(client_data)
        }
    
    def _determine_client_profile(self, client_data: Dict[str, Any]) -> str:
        """Détermine le profil du client."""
        nbr_cheques = client_data.get('Nbr_Cheques_2024', 0)
        mobile_banking = client_data.get('Utilise_Mobile_Banking', 0)
        segment = client_data.get('Segment_NMR', '')
        
        if mobile_banking and nbr_cheques <= 5:
            return "🔵 Digital"
        elif nbr_cheques > 20:
            return "🔴 Traditionnel"
        elif segment in ['S1 Excellence', 'S2 Premium']:
            return "👑 Premium"
        else:
            return "🟡 Standard"