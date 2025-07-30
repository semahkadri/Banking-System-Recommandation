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

# Import client ID utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.client_id_utils import extract_client_id

# Import enhanced behavioral segmentation (optional)
try:
    from utils.behavioral_segmentation import BehavioralSegmentationEngine
    ENHANCED_SEGMENTATION_AVAILABLE = True
except ImportError:
    ENHANCED_SEGMENTATION_AVAILABLE = False
    print("[WARNING] Enhanced behavioral segmentation not available, using legacy system")

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
    
    def __init__(self, data_path: str = "data/processed", use_enhanced_segmentation: bool = True):
        self.data_path = Path(data_path)
        self.behavior_analyzer = ClientBehaviorAnalyzer()
        
        # Configuration de la segmentation comportementale
        self.use_enhanced_segmentation = use_enhanced_segmentation and ENHANCED_SEGMENTATION_AVAILABLE
        
        if self.use_enhanced_segmentation:
            self.enhanced_segmentation_engine = BehavioralSegmentationEngine()
            print("[RECOMMENDATION] Utilisation de la segmentation comportementale avancée")
        else:
            self.enhanced_segmentation_engine = None
            print("[RECOMMENDATION] Utilisation de la segmentation comportementale standard")
        
        # Règles de recommandation par segment - PRODUITS RÉELS ATTIJARI BANK
        self.recommendation_rules = {
            "TRADITIONNEL_RESISTANT": {
                "priority": ["pack_senior_plus", "travel_card", "attijari_realtime"],
                "messaging": "Accompagnement progressif vers les alternatives digitales Attijari",
                "urgency": "BASSE"
            },
            "TRADITIONNEL_MODERE": {
                "priority": ["travel_card", "attijari_realtime", "pack_senior_plus"],
                "messaging": "Transition douce vers les services modernes d'Attijari Bank",
                "urgency": "MOYENNE"
            },
            "DIGITAL_TRANSITOIRE": {
                "priority": ["attijari_mobile", "flouci_payment", "webank_account"],
                "messaging": "Optimisation de votre expérience digitale avec Attijari",
                "urgency": "MOYENNE"
            },
            "DIGITAL_ADOPTER": {
                "priority": ["pack_exclusif", "flouci_payment", "credit_conso"],
                "messaging": "Services avancés Attijari pour utilisateurs digitaux",
                "urgency": "HAUTE"
            },
            "DIGITAL_NATIF": {
                "priority": ["webank_account", "attijari_mobile", "pack_exclusif"],
                "messaging": "Solutions innovantes et avancées d'Attijari Bank",
                "urgency": "HAUTE"
            },
            "EQUILIBRE_MIXTE": {
                "priority": ["attijari_mobile", "attijari_realtime", "travel_card"],
                "messaging": "Équilibre optimal entre services traditionnels et modernes Attijari",
                "urgency": "MOYENNE"
            },
            "EQUILIBRE": {  # Backward compatibility
                "priority": ["attijari_mobile", "attijari_realtime", "travel_card"],
                "messaging": "Équilibre optimal entre services traditionnels et modernes Attijari",
                "urgency": "MOYENNE"
            }
        }
        
        # Catalogue des services bancaires RÉELS d'Attijari Bank Tunisia (coûts en TND)
        self.services_catalog = {
            "attijari_mobile": {
                "nom": "Attijari Mobile Tunisia",
                "description": "Application mobile officielle pour gérer vos comptes 24h/24, 7j/7",
                "avantages": ["Consultation soldes en temps réel", "Historique 6 mois", "Virements gratuits", "Contrôle chéquier"],
                "cible": "Réduction significative des chèques par virements mobiles",
                "cout": 0,  # TND - Service gratuit
                "lien_produit": "https://play.google.com/store/apps/details?id=tn.com.attijarirealtime.mobile",
                "type": "Mobile Banking"
            },
            "flouci_payment": {
                "nom": "Flouci - Paiement Mobile",
                "description": "Solution de paiement mobile rapide et sécurisé d'Attijari Bank",
                "avantages": ["Paiements instantanés", "Transferts rapides", "Marchands partenaires", "Sécurité avancée"],
                "cible": "Alternative moderne aux chèques pour paiements",
                "cout": 0,  # TND - Frais par transaction
                "lien_produit": "https://www.attijaribank.com.tn/fr",
                "type": "Paiement Digital"
            },
            "attijari_realtime": {
                "nom": "Attijari Real Time",
                "description": "Plateforme bancaire en ligne pour gestion complète 24h/24",
                "avantages": ["Virements permanents", "Consultation crédits", "Tableaux amortissement", "Services en ligne"],
                "cible": "Élimination des chèques récurrents par automatisation",
                "cout": 0,  # TND - Inclus dans les packs
                "lien_produit": "https://www.attijarirealtime.com.tn/",
                "type": "Banque en Ligne"
            },
            "webank_account": {
                "nom": "WeBank - Compte Digital",
                "description": "Compte bancaire 100% digital, ouverture directe sur téléphone",
                "avantages": ["Ouverture rapide", "Gestion mobile", "Frais réduits", "Services digitaux inclus"],
                "cible": "Transition complète vers le digital",
                "cout": 0,  # TND - Selon pack choisi
                "lien_produit": "https://www.attijaribank.com.tn/fr",
                "type": "Compte Digital"
            },
            "travel_card": {
                "nom": "Travel Card Attijari",
                "description": "Carte prépayée rechargeable pour tous vos paiements",
                "avantages": ["Rechargeable 24h/24", "Paiements sécurisés", "Contrôle budget", "Sans découvert"],
                "cible": "Remplace les chèques par carte prépayée",
                "cout": 50,  # TND/an
                "lien_produit": "https://www.attijaribank.com.tn/fr",
                "type": "Carte Bancaire"
            },
            "pack_senior_plus": {
                "nom": "Pack Senior Plus",
                "description": "Pack spécialement conçu pour les clients seniors",
                "avantages": ["Services adaptés", "Accompagnement personnalisé", "Tarifs préférentiels", "Formation digitale"],
                "cible": "Transition progressive des seniors vers le digital",
                "cout": 120,  # TND/an
                "lien_produit": "https://www.attijaribank.com.tn/fr",
                "type": "Pack Services"
            },
            "credit_conso": {
                "nom": "Crédit Consommation 100% en ligne",
                "description": "Crédit personnel entièrement digital, simulation et demande en ligne",
                "avantages": ["Traitement rapide", "Simulation gratuite", "Dossier digital", "Taux attractifs"],
                "cible": "Financement sans chèques de garantie",
                "cout": 0,  # TND - Frais de dossier selon montant
                "lien_produit": "https://www.attijaribank.com.tn/fr",
                "type": "Crédit"
            },
            "pack_exclusif": {
                "nom": "Pack Compte Exclusif",
                "description": "Package premium avec services bancaires avancés",
                "avantages": ["Conseiller dédié", "Frais réduits", "Services prioritaires", "Carte Premium incluse"],
                "cible": "Optimisation complète des services bancaires",
                "cout": 600,  # TND/an
                "lien_produit": "https://www.attijaribank.com.tn/fr",
                "type": "Pack Premium"
            }
        }
        
        # VALIDATION SYSTÈME: Vérifier que toutes les recommandations existent dans le catalogue
        self._validate_recommendation_rules()
        
        print("[RECOMMENDATION] Système de recommandation initialisé")
    
    def get_segmentation_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le système de segmentation utilisé."""
        info = {
            "enhanced_available": ENHANCED_SEGMENTATION_AVAILABLE,
            "enhanced_used": self.use_enhanced_segmentation,
            "supported_segments": list(self.recommendation_rules.keys())
        }
        
        if self.use_enhanced_segmentation:
            segmentation_summary = self.enhanced_segmentation_engine.get_segmentation_summary()
            info["enhanced_details"] = segmentation_summary
        
        return info
    
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
            print("[VALIDATION] Tous les services recommandes existent dans le catalogue [OK]")
    
    def generate_recommendations(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des recommandations personnalisées pour un client."""
        
        # Analyse du comportement avec segmentation avancée si disponible
        if self.use_enhanced_segmentation:
            # Utilisation de la segmentation comportementale avancée
            enhanced_analysis = self.enhanced_segmentation_engine.analyze_client_behavior(client_data)
            behavior_profile = {
                'behavior_segment': enhanced_analysis['behavior_segment'],
                'behavioral_scores': enhanced_analysis['behavioral_scores'],
                'confidence': enhanced_analysis['analysis_metadata']['analysis_confidence'],
                'enhanced': True,
                'legacy_compatibility': self.behavior_analyzer.analyze_client_behavior(client_data)
            }
            segment = enhanced_analysis['behavior_segment']
        else:
            # Fallback vers l'analyse standard
            behavior_profile = self.behavior_analyzer.analyze_client_behavior(client_data)
            behavior_profile['enhanced'] = False
            segment = behavior_profile['behavior_segment']
        
        # Sélection des recommandations
        recommendations = self._select_recommendations(client_data, behavior_profile, segment)
        
        # Calcul des scores de pertinence
        scored_recommendations = self._score_recommendations(client_data, recommendations)
        
        # Priorisation
        prioritized_recommendations = self._prioritize_recommendations(scored_recommendations)
        
        # Extract client ID using standardized utility (same as prediction system)
        client_id = extract_client_id(client_data)
        
        return {
            'client_id': client_id,
            'behavior_profile': behavior_profile,
            'recommendations': prioritized_recommendations,
            'impact_estimations': self._estimate_impact(client_data, prioritized_recommendations),
            'generation_timestamp': datetime.now().isoformat()
        }
    
    def _select_recommendations(self, client_data: Dict[str, Any], 
                              behavior_profile: Dict[str, Any], segment: str) -> List[str]:
        """Sélectionne les recommandations appropriées selon le segment."""
        
        # VALIDATION SÉCURISÉE des données client
        if not isinstance(client_data, dict):
            raise ValueError("client_data doit être un dictionnaire")
        
        # Validation et nettoyage des données numériques sensibles
        mobile_banking = int(client_data.get('Utilise_Mobile_Banking', 0)) if client_data.get('Utilise_Mobile_Banking') in [0, 1] else 0
        revenu_estime = max(0, min(float(client_data.get('Revenu_Estime', 0)), 1000000))  # Plafond réaliste
        nbr_cheques_2024 = max(0, min(int(client_data.get('Nbr_Cheques_2024', 0)), 500))  # Plafond réaliste
        
        base_recommendations = self.recommendation_rules.get(segment, {}).get('priority', [])
        
        # Ajustements selon le profil spécifique (avec données validées)
        adjusted_recommendations = base_recommendations.copy()
        
        # Si le client utilise déjà mobile banking, ne pas recommander Attijari Mobile
        if mobile_banking:
            adjusted_recommendations = [r for r in adjusted_recommendations if r != 'attijari_mobile']
        
        # Si le client a un revenu élevé, ajouter des services premium Attijari
        if revenu_estime > 80000:
            if 'pack_exclusif' not in adjusted_recommendations:
                adjusted_recommendations.append('pack_exclusif')
        
        # Si le client a beaucoup de chèques, prioriser les alternatives directes Attijari
        if nbr_cheques_2024 > 10:
            priority_alternatives = ['travel_card', 'attijari_realtime', 'flouci_payment']
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
        
        # Scores spécifiques par service RÉEL Attijari Bank
        service_scores = {
            'attijari_mobile': 0.9 if not digital_adoption else 0.2,
            'flouci_payment': 0.8 if digital_adoption else 0.4,
            'attijari_realtime': min(0.7 + (nbr_cheques * 0.05), 1.0),
            'webank_account': 0.85 if digital_adoption else 0.3,
            'travel_card': min(0.8 + (nbr_cheques * 0.1), 1.0),
            'pack_senior_plus': 0.8 if nbr_cheques > 5 else 0.4,
            'credit_conso': min(revenu / 80000, 1.0),
            'pack_exclusif': min(revenu / 150000, 1.0)
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
        
        # Estimation de réduction des chèques par service RÉEL Attijari Bank
        base_impact_rates = {
            'attijari_mobile': 0.35,  # Impact élevé - virements mobiles remplacent chèques
            'flouci_payment': 0.30,   # Paiements instantanés vs chèques
            'attijari_realtime': 0.20, # Virements permanents automatisés
            'webank_account': 0.25,   # Compte digital complet
            'travel_card': 0.25,      # Carte prépayée vs chèques
            'pack_senior_plus': 0.20, # Transition progressive
            'credit_conso': 0.10,     # Impact indirect
            'pack_exclusif': 0.15     # Services premium
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
            
            # Si le client utilise déjà mobile banking, impact plus élevé pour services digitaux Attijari
            if digital_adoption and service_id in ['flouci_payment', 'webank_account']:
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
        
        # Revenus estimés par service RÉEL Attijari Bank (par usage en TND)
        revenue_rates = {
            'attijari_mobile': 0.3,       # TND par usage - frais virements
            'flouci_payment': 0.45,       # TND par usage - commission paiements
            'attijari_realtime': 0.6,     # TND par usage - frais virements web
            'webank_account': 0.5,        # TND par usage - frais compte digital
            'travel_card': 0.9,           # TND par usage - frais carte prépayée
            'pack_senior_plus': 10.0,     # TND par usage - frais pack mensuel
            'credit_conso': 25.0,         # TND par usage - frais crédit
            'pack_exclusif': 50.0         # TND par usage - frais pack premium
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
        
        # Revenus estimés par service RÉEL Attijari Bank (annuels en TND)
        service_revenues = {
            'attijari_mobile': 36,        # TND/an - frais virements mobiles
            'flouci_payment': 54,         # TND/an - commissions paiements
            'attijari_realtime': 72,      # TND/an - frais virements web
            'webank_account': 60,         # TND/an - frais compte digital
            'travel_card': 108,           # TND/an - frais carte + recharges
            'pack_senior_plus': 120,      # TND/an - coût pack
            'credit_conso': 300,          # TND/an - frais crédit moyen
            'pack_exclusif': 600          # TND/an - coût pack premium
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