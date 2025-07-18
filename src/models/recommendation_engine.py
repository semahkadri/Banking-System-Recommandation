# -*- coding: utf-8 -*-
"""
Système de Recommandation Personnalisée pour Services Bancaires

Ce module implémente un système de recommandation personnalisé pour proposer
des solutions adaptées au profil et comportement de chaque client bancaire.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ClientBehaviorAnalyzer:
    """Analyse le comportement des clients pour la segmentation."""
    
    def __init__(self):
        self.behavior_segments = {}
        self.migration_patterns = {}
        self.payment_preferences = {}
        
    def analyze_client_behavior(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse le comportement d'un client spécifique."""
        
        # Calcul des métriques comportementales
        check_dependency = self._calculate_check_dependency(client_data)
        digital_adoption = self._calculate_digital_adoption(client_data)
        payment_evolution = self._calculate_payment_evolution(client_data)
        risk_profile = self._calculate_risk_profile(client_data)
        
        behavior_profile = {
            'check_dependency_score': check_dependency,
            'digital_adoption_score': digital_adoption,
            'payment_evolution_score': payment_evolution,
            'risk_profile_score': risk_profile,
            'behavior_segment': self._determine_behavior_segment(
                check_dependency, digital_adoption, payment_evolution, risk_profile
            )
        }
        
        return behavior_profile
    
    def _calculate_check_dependency(self, client_data: Dict[str, Any]) -> float:
        """Calcule le niveau de dépendance aux chèques."""
        nbr_cheques_2024 = client_data.get('Nbr_Cheques_2024', 0)
        total_transactions = client_data.get('Nbr_Transactions_2025', 1)
        
        # Ratio de dépendance aux chèques
        if total_transactions > 0:
            dependency_ratio = nbr_cheques_2024 / total_transactions
        else:
            dependency_ratio = 0
        
        # Normalisation (0-1)
        dependency_score = min(dependency_ratio * 2, 1.0)
        
        return dependency_score
    
    def _calculate_digital_adoption(self, client_data: Dict[str, Any]) -> float:
        """Calcule le niveau d'adoption des services digitaux."""
        mobile_banking = client_data.get('Utilise_Mobile_Banking', 0)
        payment_methods = client_data.get('Nombre_Methodes_Paiement', 1)
        
        # Score basé sur l'utilisation mobile banking et diversité des méthodes
        mobile_score = 0.6 if mobile_banking else 0.0
        diversity_score = min(payment_methods / 5, 0.4)  # Max 0.4 pour la diversité
        
        digital_score = mobile_score + diversity_score
        
        return min(digital_score, 1.0)
    
    def _calculate_payment_evolution(self, client_data: Dict[str, Any]) -> float:
        """Calcule l'évolution des habitudes de paiement."""
        ecart_cheques = client_data.get('Ecart_Nbr_Cheques_2024_2025', 0)
        ecart_montant = client_data.get('Ecart_Montant_Max_2024_2025', 0)
        
        # Evolution positive si réduction des chèques
        if ecart_cheques < 0:  # Réduction des chèques
            evolution_score = min(abs(ecart_cheques) / 10, 0.7)
        else:  # Augmentation des chèques
            evolution_score = max(0.3 - (ecart_cheques / 10), 0)
        
        # Bonus pour évolution du montant
        if ecart_montant > 0:
            evolution_score += 0.2
        
        return min(evolution_score, 1.0)
    
    def _calculate_risk_profile(self, client_data: Dict[str, Any]) -> float:
        """Calcule le profil de risque du client."""
        demande_derogation = client_data.get('A_Demande_Derogation', 0)
        revenu_estime = client_data.get('Revenu_Estime', 30000)
        segment = client_data.get('Segment_NMR', 'S3 Essentiel')
        
        # Score de base selon le segment
        segment_risk = {
            'S1 Excellence': 0.9,
            'S2 Premium': 0.8,
            'S3 Essentiel': 0.6,
            'S4 Avenir': 0.5,
            'S5 Univers': 0.4,
            'NON SEGMENTE': 0.3
        }
        
        risk_score = segment_risk.get(segment, 0.5)
        
        # Ajustement pour les dérogations
        if demande_derogation:
            risk_score *= 0.8
        
        # Ajustement pour le revenu
        if revenu_estime > 100000:
            risk_score += 0.1
        elif revenu_estime < 20000:
            risk_score -= 0.1
        
        return max(0, min(risk_score, 1.0))
    
    def _determine_behavior_segment(self, check_dep: float, digital_adop: float, 
                                  payment_evol: float, risk_prof: float) -> str:
        """Détermine le segment comportemental du client avec logique améliorée."""
        
        # Segmentation basée sur les scores avec des seuils plus réalistes
        if check_dep > 0.6 and digital_adop < 0.3:
            return "TRADITIONNEL_RESISTANT"
        elif check_dep > 0.4 and digital_adop < 0.5 and payment_evol < 0.5:
            return "TRADITIONNEL_MODERE"
        elif digital_adop > 0.7 and payment_evol > 0.6:
            return "DIGITAL_ADOPTER"
        elif digital_adop >= 0.5 and payment_evol >= 0.5:
            return "DIGITAL_TRANSITOIRE"
        elif check_dep < 0.2 and digital_adop > 0.6:
            return "DIGITAL_NATIF"
        else:
            return "EQUILIBRE"


class RecommendationEngine:
    """Moteur de recommandations personnalisées."""
    
    def __init__(self, data_path: str = "data/processed"):
        self.data_path = Path(data_path)
        self.behavior_analyzer = ClientBehaviorAnalyzer()
        
        # Règles de recommandation par segment
        self.recommendation_rules = {
            "TRADITIONNEL_RESISTANT": {
                "priority": ["formation_digital", "accompagnement_personnel", "carte_bancaire"],
                "messaging": "Accompagnement progressif vers les alternatives",
                "urgency": "BASSE"
            },
            "TRADITIONNEL_MODERE": {
                "priority": ["carte_bancaire", "virement_automatique", "formation_digital"],
                "messaging": "Transition douce vers les services modernes",
                "urgency": "MOYENNE"
            },
            "DIGITAL_TRANSITOIRE": {
                "priority": ["mobile_banking", "paiement_mobile", "carte_sans_contact"],
                "messaging": "Optimisation de votre expérience digitale",
                "urgency": "MOYENNE"
            },
            "DIGITAL_ADOPTER": {
                "priority": ["services_premium", "carte_sans_contact", "paiement_mobile"],
                "messaging": "Services avancés pour utilisateurs digitaux",
                "urgency": "HAUTE"
            },
            "DIGITAL_NATIF": {
                "priority": ["services_premium", "mobile_banking", "carte_sans_contact"],
                "messaging": "Solutions innovantes et avancées",
                "urgency": "HAUTE"
            },
            "EQUILIBRE": {
                "priority": ["carte_bancaire", "mobile_banking", "virement_automatique"],
                "messaging": "Équilibre optimal entre services traditionnels et modernes",
                "urgency": "MOYENNE"
            }
        }
        
        # Catalogue des services bancaires (coûts en TND)
        self.services_catalog = {
            "carte_bancaire": {
                "nom": "Carte Bancaire Moderne",
                "description": "Carte avec technologie sans contact et contrôle mobile",
                "avantages": ["Paiements rapides", "Sécurité renforcée", "Contrôle temps réel"],
                "cible": "Remplace progressivement les chèques",
                "cout": 0  # TND
            },
            "mobile_banking": {
                "nom": "Application Mobile Banking",
                "description": "Gestion complète de vos comptes depuis votre smartphone",
                "avantages": ["Virements instantanés", "Suivi temps réel", "Notifications"],
                "cible": "Réduction significative des chèques",
                "cout": 0
            },
            "virement_automatique": {
                "nom": "Virements Automatiques",
                "description": "Automatisation des paiements récurrents",
                "avantages": ["Pas d'oubli", "Économie de temps", "Réduction des frais"],
                "cible": "Élimination des chèques récurrents",
                "cout": 0
            },
            "paiement_mobile": {
                "nom": "Paiement Mobile (QR Code)",
                "description": "Paiements instantanés par QR Code",
                "avantages": ["Instantané", "Sécurisé", "Pratique"],
                "cible": "Alternative moderne aux chèques",
                "cout": 0
            },
            "carte_sans_contact": {
                "nom": "Carte Sans Contact Premium",
                "description": "Carte avec plafond élevé et fonctionnalités avancées",
                "avantages": ["Plafond élevé", "Assurances incluses", "Cashback"],
                "cible": "Remplace les chèques de gros montants",
                "cout": 150  # TND/an
            },
            "services_premium": {
                "nom": "Pack Services Premium",
                "description": "Ensemble de services bancaires avancés",
                "avantages": ["Conseiller dédié", "Frais réduits", "Services prioritaires"],
                "cible": "Optimisation complète des services",
                "cout": 600  # TND/an
            },
            "formation_digital": {
                "nom": "Formation Services Digitaux",
                "description": "Accompagnement personnalisé vers le digital",
                "avantages": ["Formation gratuite", "Support dédié", "Transition accompagnée"],
                "cible": "Adoption progressive des alternatives",
                "cout": 0
            },
            "accompagnement_personnel": {
                "nom": "Accompagnement Personnel",
                "description": "Conseiller dédié pour la transition",
                "avantages": ["Conseils personnalisés", "Suivi régulier", "Adaptation graduelle"],
                "cible": "Changement en douceur des habitudes",
                "cout": 0
            }
        }
        
        # VALIDATION SYSTÈME: Vérifier que toutes les recommandations existent dans le catalogue
        self._validate_recommendation_rules()
        
        print("[RECOMMENDATION] Système de recommandation initialisé")
    
    def _validate_recommendation_rules(self):
        """Valide que tous les services dans les règles existent dans le catalogue."""
        all_errors = []
        
        for segment, rules in self.recommendation_rules.items():
            priority_services = rules.get('priority', [])
            for service in priority_services:
                if service not in self.services_catalog:
                    error = f"Service non-existant '{service}' dans segment '{segment}'"
                    all_errors.append(error)
                    print(f"[ERREUR VALIDATION] {error}")
        
        if all_errors:
            raise ValueError(f"Services non-existants détectés: {all_errors}")
        else:
            print("[VALIDATION] Tous les services recommandés existent dans le catalogue ✓")
    
    def generate_recommendations(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des recommandations personnalisées pour un client."""
        
        # Analyse du comportement
        behavior_profile = self.behavior_analyzer.analyze_client_behavior(client_data)
        
        # Sélection des recommandations
        segment = behavior_profile['behavior_segment']
        recommendations = self._select_recommendations(client_data, behavior_profile, segment)
        
        # Calcul des scores de pertinence
        scored_recommendations = self._score_recommendations(client_data, recommendations)
        
        # Priorisation
        prioritized_recommendations = self._prioritize_recommendations(scored_recommendations)
        
        return {
            'client_id': client_data.get('CLI', 'unknown'),
            'behavior_profile': behavior_profile,
            'recommendations': prioritized_recommendations,
            'impact_estimations': self._estimate_impact(client_data, prioritized_recommendations),
            'generation_timestamp': datetime.now().isoformat()
        }
    
    def _select_recommendations(self, client_data: Dict[str, Any], 
                              behavior_profile: Dict[str, Any], segment: str) -> List[str]:
        """Sélectionne les recommandations appropriées selon le segment."""
        
        base_recommendations = self.recommendation_rules.get(segment, {}).get('priority', [])
        
        # Ajustements selon le profil spécifique
        adjusted_recommendations = base_recommendations.copy()
        
        # Si le client utilise déjà mobile banking, ne pas le recommander
        if client_data.get('Utilise_Mobile_Banking', 0):
            adjusted_recommendations = [r for r in adjusted_recommendations if r != 'mobile_banking']
        
        # Si le client a un revenu élevé, ajouter des services premium
        if client_data.get('Revenu_Estime', 0) > 80000:
            if 'services_premium' not in adjusted_recommendations:
                adjusted_recommendations.append('services_premium')
        
        # Si le client a beaucoup de chèques, prioriser les alternatives directes
        if client_data.get('Nbr_Cheques_2024', 0) > 10:
            priority_alternatives = ['carte_bancaire', 'virement_automatique', 'paiement_mobile']
            adjusted_recommendations = priority_alternatives + adjusted_recommendations
        
        # VALIDATION CRITIQUE: Supprimer les services qui n'existent pas dans le catalogue
        validated_recommendations = [r for r in adjusted_recommendations if r in self.services_catalog]
        
        # Log des services invalides pour debugging
        invalid_services = [r for r in adjusted_recommendations if r not in self.services_catalog]
        if invalid_services:
            print(f"[ERREUR] Services non-existants supprimés: {invalid_services}")
        
        return list(dict.fromkeys(validated_recommendations))  # Supprime les doublons
    
    def _score_recommendations(self, client_data: Dict[str, Any], 
                             recommendations: List[str]) -> List[Dict[str, Any]]:
        """Calcule les scores de pertinence pour chaque recommandation."""
        
        scored_recommendations = []
        
        for rec_id in recommendations:
            if rec_id in self.services_catalog:
                service = self.services_catalog[rec_id]
                
                # Score de base selon le profil
                base_score = self._calculate_base_score(client_data, rec_id)
                
                # Score d'urgence
                urgency_score = self._calculate_urgency_score(client_data, rec_id)
                
                # Score de faisabilité
                feasibility_score = self._calculate_feasibility_score(client_data, rec_id)
                
                # Score global
                global_score = (base_score * 0.5 + urgency_score * 0.3 + feasibility_score * 0.2)
                
                scored_recommendations.append({
                    'service_id': rec_id,
                    'service_info': service,
                    'scores': {
                        'base': base_score,
                        'urgency': urgency_score,
                        'feasibility': feasibility_score,
                        'global': global_score
                    }
                })
        
        return scored_recommendations
    
    def _calculate_base_score(self, client_data: Dict[str, Any], service_id: str) -> float:
        """Calcule le score de base selon l'adéquation client-service."""
        
        nbr_cheques = client_data.get('Nbr_Cheques_2024', 0)
        revenu = client_data.get('Revenu_Estime', 30000)
        digital_adoption = client_data.get('Utilise_Mobile_Banking', 0)
        
        # Scores spécifiques par service
        service_scores = {
            'carte_bancaire': min(0.8 + (nbr_cheques * 0.1), 1.0),
            'mobile_banking': 0.9 if not digital_adoption else 0.2,
            'virement_automatique': min(0.7 + (nbr_cheques * 0.05), 1.0),
            'paiement_mobile': 0.8 if digital_adoption else 0.4,
            'carte_sans_contact': min(0.6 + (revenu / 100000), 1.0),
            'services_premium': min(revenu / 150000, 1.0),
            'formation_digital': 0.9 if not digital_adoption else 0.1,
            'accompagnement_personnel': 0.8 if nbr_cheques > 5 else 0.4
        }
        
        return service_scores.get(service_id, 0.5)
    
    def _calculate_urgency_score(self, client_data: Dict[str, Any], service_id: str) -> float:
        """Calcule le score d'urgence pour une recommandation."""
        
        ecart_cheques = client_data.get('Ecart_Nbr_Cheques_2024_2025', 0)
        demande_derogation = client_data.get('A_Demande_Derogation', 0)
        
        # Urgence élevée si augmentation des chèques ou demande de dérogation
        if ecart_cheques > 0 or demande_derogation:
            return 0.9
        elif ecart_cheques < -5:  # Forte réduction des chèques
            return 0.7
        else:
            return 0.5
    
    def _calculate_feasibility_score(self, client_data: Dict[str, Any], service_id: str) -> float:
        """Calcule le score de faisabilité (capacité du client à adopter le service)."""
        
        age_estimate = self._estimate_age(client_data)
        revenu = client_data.get('Revenu_Estime', 30000)
        segment = client_data.get('Segment_NMR', 'S3 Essentiel')
        
        # Score de base selon l'âge (estimation)
        if age_estimate < 30:
            age_score = 0.9
        elif age_estimate < 50:
            age_score = 0.8
        elif age_estimate < 65:
            age_score = 0.6
        else:
            age_score = 0.4
        
        # Score selon le segment
        segment_scores = {
            'S1 Excellence': 0.9,
            'S2 Premium': 0.8,
            'S3 Essentiel': 0.7,
            'S4 Avenir': 0.8,
            'S5 Univers': 0.6
        }
        
        segment_score = segment_scores.get(segment, 0.6)
        
        # Score selon le revenu pour les services payants
        service_cost = self.services_catalog.get(service_id, {}).get('cout', 0)
        if service_cost > 0:
            cost_score = min(revenu / (service_cost * 1000), 1.0)
        else:
            cost_score = 1.0
        
        return (age_score * 0.4 + segment_score * 0.3 + cost_score * 0.3)
    
    def _estimate_age(self, client_data: Dict[str, Any]) -> int:
        """Estime l'âge du client basé sur CSP et autres indicateurs."""
        
        csp = str(client_data.get('CSP', '')).upper()
        
        if 'JEUNE' in csp:
            return 25
        elif 'RETRAITE' in csp:
            return 65
        elif 'CADRE' in csp:
            return 45
        elif 'SALARIE' in csp:
            return 40
        else:
            return 45  # Âge par défaut
    
    def _prioritize_recommendations(self, scored_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Priorise les recommandations selon leurs scores."""
        
        # Tri par score global décroissant
        prioritized = sorted(scored_recommendations, 
                           key=lambda x: x['scores']['global'], 
                           reverse=True)
        
        # Limitation à 5 recommandations maximum
        return prioritized[:5]
    
    def _estimate_impact(self, client_data: Dict[str, Any], 
                        recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estime l'impact des recommandations sur l'utilisation des services."""
        
        current_checks = client_data.get('Nbr_Cheques_2024', 0)
        digital_adoption = client_data.get('Utilise_Mobile_Banking', 0)
        check_evolution = client_data.get('Ecart_Nbr_Cheques_2024_2025', 0)
        
        # Estimation de réduction des chèques par service (ajustée selon le profil)
        base_impact_rates = {
            'carte_bancaire': 0.25,
            'mobile_banking': 0.35,
            'virement_automatique': 0.20,
            'paiement_mobile': 0.30,
            'carte_sans_contact': 0.15,
            'services_premium': 0.10,
            'formation_digital': 0.25,
            'accompagnement_personnel': 0.20
        }
        
        total_impact = 0
        service_impacts = {}
        
        for rec in recommendations:
            service_id = rec['service_id']
            score = rec['scores']['global']
            
            # Ajustement de l'impact selon le profil client
            base_rate = base_impact_rates.get(service_id, 0.15)
            
            # Facteur d'ajustement basé sur le profil
            adjustment_factor = 1.0
            
            # Si le client utilise déjà mobile banking, impact plus élevé pour services digitaux
            if digital_adoption and service_id in ['paiement_mobile', 'carte_sans_contact']:
                adjustment_factor = 1.3
            
            # Si le client a déjà réduit ses chèques, impact moindre
            if check_evolution < -2:
                adjustment_factor *= 0.8
            elif check_evolution > 2:
                adjustment_factor *= 1.2
            
            # Impact pondéré et ajusté
            impact = base_rate * score * adjustment_factor
            total_impact += impact
            
            service_impacts[service_id] = {
                'reduction_estimee': current_checks * impact,
                'score_adoption': score,
                'impact_financier': self._calculate_financial_impact(client_data, service_id, impact)
            }
        
        # Plafonnement de l'impact total (plus réaliste)
        total_impact = min(total_impact, 0.65)
        
        # Calcul du bénéfice bancaire plus réaliste
        cost_per_check = 4.5  # TND par chèque traité
        checks_reduced = current_checks * total_impact
        operational_savings = checks_reduced * cost_per_check
        
        # Revenus additionnels des nouveaux services
        additional_revenues = sum(
            impact_data['impact_financier']['revenus_service'] 
            for impact_data in service_impacts.values()
        )
        
        return {
            'reduction_cheques_estimee': checks_reduced,
            'pourcentage_reduction': total_impact * 100,
            'impacts_par_service': service_impacts,
            'benefice_bancaire_estime': operational_savings + additional_revenues,
            'economies_operationnelles': operational_savings,
            'revenus_additionnels': additional_revenues
        }
    
    def _calculate_financial_impact(self, client_data: Dict[str, Any], 
                                  service_id: str, impact_rate: float) -> Dict[str, float]:
        """Calcule l'impact financier d'une recommandation."""
        
        current_checks = client_data.get('Nbr_Cheques_2024', 0)
        avg_check_amount = client_data.get('Montant_Moyen_Cheque', 1000)
        
        # Coût estimé des chèques évités (en TND)
        checks_avoided = current_checks * impact_rate
        cost_savings = checks_avoided * 4.5  # Coût moyen de traitement d'un chèque en TND
        
        # Revenus estimés du nouveau service
        service_revenue = self._estimate_service_revenue(service_id, checks_avoided, avg_check_amount)
        
        return {
            'economies_frais': cost_savings,
            'revenus_service': service_revenue,
            'impact_net': service_revenue - cost_savings
        }
    
    def _estimate_service_revenue(self, service_id: str, usage_frequency: float, avg_amount: float) -> float:
        """Estime les revenus générés par un service."""
        
        # Revenus estimés par service (par usage en TND)
        revenue_rates = {
            'carte_bancaire': 0.9,  # TND par usage
            'mobile_banking': 0.3,  # TND par usage
            'virement_automatique': 0.6,  # TND par usage
            'paiement_mobile': 0.45,  # TND par usage
            'carte_sans_contact': 1.05,  # TND par usage
            'services_premium': 15.0,  # TND par usage
            'formation_digital': 0.0,  # TND par usage
            'accompagnement_personnel': 0.0  # TND par usage
        }
        
        rate = revenue_rates.get(service_id, 0.2)
        return usage_frequency * rate * 12  # Revenus annuels estimés


class RecommendationTracker:
    """Suivi et mesure de l'efficacité des recommandations."""
    
    def __init__(self, data_path: str = "data/processed"):
        self.data_path = Path(data_path)
        self.recommendations_file = self.data_path / "recommendations_history.json"
        self.adoptions_file = self.data_path / "service_adoptions.json"
        
        # Initialiser les fichiers s'ils n'existent pas
        self._initialize_tracking_files()
    
    def _initialize_tracking_files(self):
        """Initialise les fichiers de suivi."""
        if not self.recommendations_file.exists():
            with open(self.recommendations_file, 'w') as f:
                json.dump([], f)
        
        if not self.adoptions_file.exists():
            with open(self.adoptions_file, 'w') as f:
                json.dump({}, f)
    
    def record_recommendation(self, client_id: str, recommendations: Dict[str, Any]):
        """Enregistre une recommandation générée."""
        
        # Charger l'historique existant
        with open(self.recommendations_file, 'r') as f:
            history = json.load(f)
        
        # Ajouter la nouvelle recommandation
        record = {
            'client_id': client_id,
            'timestamp': datetime.now().isoformat(),
            'recommendations': recommendations,
            'status': 'GENERATED'
        }
        
        history.append(record)
        
        # Sauvegarder
        with open(self.recommendations_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def record_adoption(self, client_id: str, service_id: str, adoption_date: str = None):
        """Enregistre l'adoption d'un service par un client."""
        
        if adoption_date is None:
            adoption_date = datetime.now().isoformat()
        
        # Charger les adoptions existantes
        with open(self.adoptions_file, 'r') as f:
            adoptions = json.load(f)
        
        # Ajouter l'adoption
        if client_id not in adoptions:
            adoptions[client_id] = []
        
        adoptions[client_id].append({
            'service_id': service_id,
            'adoption_date': adoption_date,
            'source': 'RECOMMENDATION'
        })
        
        # Sauvegarder
        with open(self.adoptions_file, 'w') as f:
            json.dump(adoptions, f, indent=2)
    
    def calculate_adoption_rate(self, period_days: int = 30) -> Dict[str, float]:
        """Calcule le taux d'adoption des recommandations."""
        
        # Charger les données
        with open(self.recommendations_file, 'r') as f:
            history = json.load(f)
        
        with open(self.adoptions_file, 'r') as f:
            adoptions = json.load(f)
        
        # Période de référence
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        # Compter les recommandations et adoptions (sans duplication)
        unique_clients_with_recommendations = set()
        unique_clients_with_adoptions = set()
        service_stats = {}
        
        # Compter les clients avec recommandations dans la période
        for record in history:
            try:
                record_date = datetime.fromisoformat(record['timestamp'])
                if record_date >= cutoff_date:
                    client_id = record['client_id']
                    unique_clients_with_recommendations.add(client_id)
                    
                    # Compter les services recommandés (éviter les doublons par client)
                    for rec in record.get('recommendations', {}).get('recommendations', []):
                        service_id = rec.get('service_id')
                        if service_id:
                            if service_id not in service_stats:
                                service_stats[service_id] = {'recommended': 0, 'adopted': 0}
                            service_stats[service_id]['recommended'] += 1
            except:
                continue
        
        # Compter les adoptions dans la période
        for client_id, client_adoptions in adoptions.items():
            for adoption in client_adoptions:
                try:
                    adoption_date = datetime.fromisoformat(adoption['adoption_date'])
                    if adoption_date >= cutoff_date:
                        unique_clients_with_adoptions.add(client_id)
                        
                        # Compter les services adoptés
                        service_id = adoption['service_id']
                        if service_id not in service_stats:
                            service_stats[service_id] = {'recommended': 0, 'adopted': 0}
                        service_stats[service_id]['adopted'] += 1
                except:
                    continue
        
        # Calculer les taux
        recommendations_count = len(unique_clients_with_recommendations)
        adoptions_count = len(unique_clients_with_adoptions)
        overall_rate = (adoptions_count / recommendations_count * 100) if recommendations_count > 0 else 0
        
        # Debug info
        print(f"[DEBUG ADOPTION] Period: {period_days} days")
        print(f"[DEBUG ADOPTION] Unique clients with recommendations: {recommendations_count}")
        print(f"[DEBUG ADOPTION] Unique clients with adoptions: {adoptions_count}")
        print(f"[DEBUG ADOPTION] Overall rate: {overall_rate}%")
        
        service_rates = {}
        for service_id, stats in service_stats.items():
            if stats['recommended'] > 0:
                rate = (stats['adopted'] / stats['recommended'] * 100)
                service_rates[service_id] = rate
                print(f"[DEBUG ADOPTION] Service {service_id}: {stats['adopted']}/{stats['recommended']} = {rate}%")
        
        return {
            'overall_adoption_rate': overall_rate,
            'total_recommendations': recommendations_count,
            'total_adoptions': adoptions_count,
            'service_adoption_rates': service_rates,
            'period_days': period_days
        }
    
    def generate_effectiveness_report(self) -> Dict[str, Any]:
        """Génère un rapport d'efficacité des recommandations."""
        
        # Taux d'adoption sur différentes périodes
        adoption_30d = self.calculate_adoption_rate(30)
        adoption_90d = self.calculate_adoption_rate(90)
        
        # Charger toutes les données pour analyses plus poussées
        with open(self.recommendations_file, 'r') as f:
            history = json.load(f)
        
        with open(self.adoptions_file, 'r') as f:
            adoptions = json.load(f)
        
        # Analyse des segments les plus réceptifs
        segment_analysis = self._analyze_segment_receptivity(history, adoptions)
        
        # Services les plus adoptés
        popular_services = self._identify_popular_services(adoptions)
        
        # Impact financier estimé
        financial_impact = self._estimate_financial_impact_global(adoptions)
        
        # Tendances d'adoption par semaine
        weekly_trends = self._calculate_weekly_trends(history, adoptions)
        
        return {
            'adoption_rates': {
                '30_days': adoption_30d,
                '90_days': adoption_90d
            },
            'segment_analysis': segment_analysis,
            'popular_services': popular_services,
            'financial_impact': financial_impact,
            'weekly_trends': weekly_trends,
            'total_clients_served': len(set(r['client_id'] for r in history)),
            'report_date': datetime.now().isoformat()
        }
    
    def _analyze_segment_receptivity(self, history: List[Dict], adoptions: Dict[str, List]) -> Dict[str, Any]:
        """Analyse la réceptivité par segment comportemental."""
        
        segment_stats = {}
        
        for record in history:
            behavior_profile = record.get('recommendations', {}).get('behavior_profile', {})
            segment = behavior_profile.get('behavior_segment', 'UNKNOWN')
            
            if segment not in segment_stats:
                segment_stats[segment] = {'recommended': 0, 'adopted': 0}
            
            segment_stats[segment]['recommended'] += 1
            
            # Vérifier adoptions
            client_id = record['client_id']
            if client_id in adoptions:
                segment_stats[segment]['adopted'] += len(adoptions[client_id])
        
        # Calculer les taux
        for segment, stats in segment_stats.items():
            if stats['recommended'] > 0:
                stats['adoption_rate'] = (stats['adopted'] / stats['recommended'] * 100)
            else:
                stats['adoption_rate'] = 0
        
        return segment_stats
    
    def _identify_popular_services(self, adoptions: Dict[str, List]) -> Dict[str, int]:
        """Identifie les services les plus adoptés."""
        
        service_counts = {}
        
        for client_adoptions in adoptions.values():
            for adoption in client_adoptions:
                service_id = adoption['service_id']
                service_counts[service_id] = service_counts.get(service_id, 0) + 1
        
        # Trier par popularité
        return dict(sorted(service_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _estimate_financial_impact_global(self, adoptions: Dict[str, List]) -> Dict[str, float]:
        """Estime l'impact financier global des adoptions."""
        
        # Revenus estimés par service (annuels en TND)
        service_revenues = {
            'carte_bancaire': 72,  # TND/an
            'mobile_banking': 36,  # TND/an
            'virement_automatique': 54,  # TND/an
            'paiement_mobile': 45,  # TND/an
            'carte_sans_contact': 108,  # TND/an
            'services_premium': 600,  # TND/an
            'formation_digital': 0,  # TND/an
            'accompagnement_personnel': 0  # TND/an
        }
        
        total_revenue = 0
        service_breakdown = {}
        
        for client_adoptions in adoptions.values():
            for adoption in client_adoptions:
                service_id = adoption['service_id']
                revenue = service_revenues.get(service_id, 10)
                
                total_revenue += revenue
                service_breakdown[service_id] = service_breakdown.get(service_id, 0) + revenue
        
        return {
            'total_annual_revenue': total_revenue,
            'revenue_by_service': service_breakdown,
            'average_revenue_per_client': total_revenue / len(adoptions) if adoptions else 0
        }
    
    def _calculate_weekly_trends(self, history: List[Dict], adoptions: Dict[str, List]) -> Dict[str, Any]:
        """Calcule les tendances d'adoption par semaine."""
        
        from collections import defaultdict
        
        # Grouper par semaine
        weekly_recommendations = defaultdict(int)
        weekly_adoptions = defaultdict(int)
        
        # Analyser les recommandations
        for record in history:
            try:
                record_date = datetime.fromisoformat(record['timestamp'])
                week_start = record_date - timedelta(days=record_date.weekday())
                week_key = week_start.strftime('%Y-%m-%d')
                weekly_recommendations[week_key] += 1
            except:
                continue
        
        # Analyser les adoptions
        for client_adoptions in adoptions.values():
            for adoption in client_adoptions:
                try:
                    adoption_date = datetime.fromisoformat(adoption['adoption_date'])
                    week_start = adoption_date - timedelta(days=adoption_date.weekday())
                    week_key = week_start.strftime('%Y-%m-%d')
                    weekly_adoptions[week_key] += 1
                except:
                    continue
        
        # Créer les données de tendance
        all_weeks = set(weekly_recommendations.keys()) | set(weekly_adoptions.keys())
        trends = []
        
        for week in sorted(all_weeks):
            recommendations = weekly_recommendations.get(week, 0)
            adoptions_count = weekly_adoptions.get(week, 0)
            adoption_rate = (adoptions_count / recommendations * 100) if recommendations > 0 else 0
            
            trends.append({
                'week': week,
                'recommendations': recommendations,
                'adoptions': adoptions_count,
                'adoption_rate': adoption_rate
            })
        
        return {
            'trends': trends[-8:],  # Dernières 8 semaines
            'total_weeks': len(trends),
            'avg_weekly_recommendations': sum(weekly_recommendations.values()) / len(weekly_recommendations) if weekly_recommendations else 0,
            'avg_weekly_adoptions': sum(weekly_adoptions.values()) / len(weekly_adoptions) if weekly_adoptions else 0
        }