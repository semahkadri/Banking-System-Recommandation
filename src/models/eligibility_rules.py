# -*- coding: utf-8 -*-
"""
Règles d'Éligibilité Bancaire - Système de Recommandation

Ce module implémente les règles internes de la banque pour:
- L'éligibilité d'un client à un chéquier
- Le calcul des montants seuils par chèque
- Le nombre de chèques octroyés

Conforme aux exigences de l'énoncé pour l'intégration des règles existantes.
"""

from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class BankEligibilityRules:
    """Gestionnaire des règles d'éligibilité bancaire internes."""
    
    def __init__(self):
        # Règles de base par segment client
        self.segment_rules = {
            'S1 Excellence': {
                'eligible': True,
                'base_limit': 50000,  # TND
                'base_checks': 50,
                'risk_multiplier': 1.5
            },
            'S2 Premium': {
                'eligible': True,
                'base_limit': 30000,  # TND
                'base_checks': 40,
                'risk_multiplier': 1.3
            },
            'S3 Essentiel': {
                'eligible': True,
                'base_limit': 15000,  # TND
                'base_checks': 25,
                'risk_multiplier': 1.0
            },
            'S4 Avenir': {
                'eligible': True,
                'base_limit': 10000,  # TND
                'base_checks': 20,
                'risk_multiplier': 0.8
            },
            'S5 Univers': {
                'eligible': True,
                'base_limit': 5000,   # TND
                'base_checks': 15,
                'risk_multiplier': 0.6
            },
            'NON SEGMENTE': {
                'eligible': False,
                'base_limit': 0,
                'base_checks': 0,
                'risk_multiplier': 0.0
            }
        }
        
        # Ajustements par CSP (Catégorie Socio-Professionnelle)
        self.csp_adjustments = {
            'CADRE SUPERIEUR': {'limit_factor': 1.5, 'checks_factor': 1.3},
            'CADRE MOYEN': {'limit_factor': 1.2, 'checks_factor': 1.1},
            'EMPLOYE': {'limit_factor': 1.0, 'checks_factor': 1.0},
            'OUVRIER': {'limit_factor': 0.8, 'checks_factor': 0.9},
            'RETRAITE': {'limit_factor': 0.9, 'checks_factor': 0.8},
            'SANS EMPLOI': {'limit_factor': 0.3, 'checks_factor': 0.5},
            'ETUDIANT': {'limit_factor': 0.4, 'checks_factor': 0.6},
            'LIBERAL': {'limit_factor': 1.4, 'checks_factor': 1.2},
            'COMMERCANT': {'limit_factor': 1.3, 'checks_factor': 1.2},
            'AGRICULTEUR': {'limit_factor': 1.1, 'checks_factor': 1.0}
        }
        
        # Règles spéciales par marché
        self.market_rules = {
            'Particuliers': {
                'max_limit': 100000,  # TND
                'max_checks': 100,
                'requires_income_proof': True
            },
            'Entreprises': {
                'max_limit': 500000,  # TND
                'max_checks': 200,
                'requires_income_proof': False
            },
            'Associations': {
                'max_limit': 50000,   # TND
                'max_checks': 50,
                'requires_income_proof': False
            }
        }
    
    def evaluate_checkbook_eligibility(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Évalue l'éligibilité d'un client à un chéquier selon les règles internes.
        
        Args:
            client_data: Données du client (segment, CSP, marché, revenus, etc.)
            
        Returns:
            Dict contenant l'éligibilité et les conditions
        """
        
        segment = client_data.get('Segment_NMR', 'NON SEGMENTE')
        csp = str(client_data.get('CSP', '')).upper()
        market = client_data.get('CLIENT_MARCHE', 'Particuliers')
        revenue = client_data.get('Revenu_Estime', 0)
        has_derogation = client_data.get('A_Demande_Derogation', 0)
        
        # Vérification éligibilité de base
        base_rules = self.segment_rules.get(segment, self.segment_rules['NON SEGMENTE'])
        market_rules = self.market_rules.get(market, self.market_rules['Particuliers'])
        
        # Éligibilité de base
        is_eligible = base_rules['eligible']
        reasons = []
        
        # Conditions d'exclusion
        if segment == 'NON SEGMENTE':
            is_eligible = False
            reasons.append("Client non segmenté - éligibilité refusée")
        
        if revenue < 5000 and market == 'Particuliers':
            is_eligible = False
            reasons.append("Revenus insuffisants (< 5000 TND)")
        
        if has_derogation:
            is_eligible = False
            reasons.append("Demande de dérogation en cours - éligibilité suspendue")
        
        # Calcul des limites si éligible
        recommended_limit = 0
        recommended_checks = 0
        
        if is_eligible:
            recommended_limit, recommended_checks = self._calculate_limits(
                client_data, base_rules, market_rules
            )
            reasons.append("Client éligible selon les critères bancaires")
        
        return {
            'eligible': is_eligible,
            'recommended_check_limit': recommended_limit,
            'recommended_check_count': recommended_checks,
            'eligibility_reasons': reasons,
            'risk_level': self._assess_risk_level(client_data),
            'special_conditions': self._get_special_conditions(client_data)
        }
    
    def _calculate_limits(self, client_data: Dict[str, Any], 
                         base_rules: Dict[str, Any], 
                         market_rules: Dict[str, Any]) -> Tuple[float, int]:
        """Calcule les limites recommandées pour le client."""
        
        csp = str(client_data.get('CSP', '')).upper()
        revenue = client_data.get('Revenu_Estime', 0)
        
        # Limites de base
        base_limit = base_rules['base_limit']
        base_checks = base_rules['base_checks']
        
        # Ajustements CSP
        csp_adj = self.csp_adjustments.get(csp, {'limit_factor': 1.0, 'checks_factor': 1.0})
        
        # Calcul des limites ajustées
        adjusted_limit = base_limit * csp_adj['limit_factor']
        adjusted_checks = int(base_checks * csp_adj['checks_factor'])
        
        # Ajustement par revenus
        if revenue > 0:
            revenue_factor = min(revenue / 30000, 2.0)  # Max 2x pour revenus élevés
            adjusted_limit *= revenue_factor
            adjusted_checks = int(adjusted_checks * min(revenue_factor, 1.5))
        
        # Respect des limites maximales du marché
        final_limit = min(adjusted_limit, market_rules['max_limit'])
        final_checks = min(adjusted_checks, market_rules['max_checks'])
        
        return final_limit, final_checks
    
    def _assess_risk_level(self, client_data: Dict[str, Any]) -> str:
        """Évalue le niveau de risque du client."""
        
        segment = client_data.get('Segment_NMR', 'NON SEGMENTE')
        revenue = client_data.get('Revenu_Estime', 0)
        has_derogation = client_data.get('A_Demande_Derogation', 0)
        check_usage = client_data.get('Nbr_Cheques_2024', 0)
        
        risk_score = 0
        
        # Risque par segment
        segment_risk = {
            'S1 Excellence': 1,
            'S2 Premium': 2,
            'S3 Essentiel': 3,
            'S4 Avenir': 4,
            'S5 Univers': 5,
            'NON SEGMENTE': 10
        }
        risk_score += segment_risk.get(segment, 10)
        
        # Risque par revenus
        if revenue < 10000:
            risk_score += 3
        elif revenue < 30000:
            risk_score += 1
        
        # Risque par usage chèques
        if check_usage > 50:
            risk_score += 2
        elif check_usage > 20:
            risk_score += 1
        
        # Pénalité dérogation
        if has_derogation:
            risk_score += 5
        
        # Classification finale
        if risk_score <= 3:
            return "FAIBLE"
        elif risk_score <= 6:
            return "MODÉRÉ"
        elif risk_score <= 10:
            return "ÉLEVÉ"
        else:
            return "TRÈS ÉLEVÉ"
    
    def _get_special_conditions(self, client_data: Dict[str, Any]) -> list:
        """Détermine les conditions spéciales à appliquer."""
        
        conditions = []
        
        segment = client_data.get('Segment_NMR', 'NON SEGMENTE')
        market = client_data.get('CLIENT_MARCHE', 'Particuliers')
        revenue = client_data.get('Revenu_Estime', 0)
        
        # Conditions par segment
        if segment in ['S1 Excellence', 'S2 Premium']:
            conditions.append("Suivi privilégié - conseiller dédié")
        
        # Conditions par marché
        if market == 'Particuliers' and revenue > 100000:
            conditions.append("Justificatif de revenus requis")
        
        if market == 'Entreprises':
            conditions.append("Validation comptabilité entreprise")
        
        # Conditions par revenus
        if revenue < 20000:
            conditions.append("Révision trimestrielle des limites")
        
        return conditions
    
    def generate_eligibility_report(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère un rapport complet d'éligibilité pour un client.
        Utilisé par le système de recommandation pour l'intégration.
        """
        
        eligibility = self.evaluate_checkbook_eligibility(client_data)
        
        # Ajout d'informations contextuelles
        client_id = client_data.get('CLI', 'Unknown')
        
        report = {
            'client_id': client_id,
            'evaluation_date': "2025-07-11",
            'eligibility_status': eligibility['eligible'],
            'financial_conditions': {
                'recommended_check_limit': eligibility['recommended_check_limit'],
                'recommended_check_count': eligibility['recommended_check_count'],
                'current_usage': client_data.get('Nbr_Cheques_2024', 0),
                'usage_efficiency': self._calculate_usage_efficiency(client_data)
            },
            'risk_assessment': {
                'risk_level': eligibility['risk_level'],
                'risk_factors': self._identify_risk_factors(client_data)
            },
            'recommendations': {
                'eligibility_reasons': eligibility['eligibility_reasons'],
                'special_conditions': eligibility['special_conditions'],
                'next_review_date': "2025-10-11"  # Révision dans 3 mois
            }
        }
        
        return report
    
    def _calculate_usage_efficiency(self, client_data: Dict[str, Any]) -> str:
        """Calcule l'efficacité d'usage des chèques."""
        
        current_usage = client_data.get('Nbr_Cheques_2024', 0)
        segment = client_data.get('Segment_NMR', 'NON SEGMENTE')
        
        # Usage recommandé par segment
        recommended_usage = self.segment_rules.get(segment, {}).get('base_checks', 0)
        
        if recommended_usage == 0:
            return "NON APPLICABLE"
        
        efficiency = current_usage / recommended_usage
        
        if efficiency < 0.3:
            return "SOUS-UTILISÉ"
        elif efficiency < 0.7:
            return "USAGE MODÉRÉ"
        elif efficiency < 1.2:
            return "USAGE OPTIMAL"
        else:
            return "SUR-UTILISÉ"
    
    def _identify_risk_factors(self, client_data: Dict[str, Any]) -> list:
        """Identifie les facteurs de risque spécifiques."""
        
        risk_factors = []
        
        if client_data.get('A_Demande_Derogation', 0):
            risk_factors.append("Demande de dérogation active")
        
        if client_data.get('Revenu_Estime', 0) < 10000:
            risk_factors.append("Revenus faibles")
        
        if client_data.get('Nbr_Cheques_2024', 0) > 50:
            risk_factors.append("Usage intensif des chèques")
        
        check_evolution = client_data.get('Ecart_Nbr_Cheques_2024_2025', 0)
        if check_evolution > 10:
            risk_factors.append("Augmentation significative usage chèques")
        
        if not client_data.get('Utilise_Mobile_Banking', 0):
            risk_factors.append("Non-adoption services digitaux")
        
        return risk_factors if risk_factors else ["Aucun facteur de risque identifié"]


# Intégration avec le système de recommandation
def integrate_eligibility_with_recommendations(client_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fonction d'intégration pour enrichir les recommandations avec les règles d'éligibilité.
    """
    
    eligibility_engine = BankEligibilityRules()
    eligibility_report = eligibility_engine.generate_eligibility_report(client_data)
    
    return {
        'eligibility_analysis': eligibility_report,
        'integration_status': 'COMPLETE',
        'recommendations_enhancement': {
            'checkbook_alternatives_priority': 'HIGH' if not eligibility_report['eligibility_status'] else 'MEDIUM',
            'digital_services_priority': 'HIGH' if eligibility_report['risk_assessment']['risk_level'] in ['ÉLEVÉ', 'TRÈS ÉLEVÉ'] else 'MEDIUM'
        }
    }