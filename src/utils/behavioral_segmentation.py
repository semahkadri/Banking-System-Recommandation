# -*- coding: utf-8 -*-
"""
Système de Segmentation Comportementale Bancaire

Ce module implémente une segmentation comportementale avancée des clients bancaires
basée sur leurs habitudes de paiement, adoption digitale et profil de risque.
"""

from typing import Dict, Any, Tuple, List
import json
from pathlib import Path

class BehavioralSegmentationEngine:
    """Moteur de segmentation comportementale des clients bancaires."""
    
    def __init__(self):
        """Initialise le moteur de segmentation avec les critères définis."""
        
        # Critères de segmentation détaillés
        self.segmentation_criteria = {
            "dimensions": {
                "check_dependency": {
                    "description": "Niveau de dépendance aux chèques comme méthode de paiement",
                    "calculation_method": "Ratio chèques / total transactions + pondération historique",
                    "scale": "0.0 (aucune dépendance) à 1.0 (très forte dépendance)",
                    "business_impact": "Plus le score est élevé, plus le client résiste au digital"
                },
                "digital_adoption": {
                    "description": "Niveau d'adoption des services bancaires digitaux",
                    "calculation_method": "Mobile banking + diversité méthodes paiement + usage services en ligne",
                    "scale": "0.0 (aucune adoption) à 1.0 (adoption complète)",
                    "business_impact": "Plus le score est élevé, plus le client est réceptif aux nouveaux services"
                },
                "payment_evolution": {
                    "description": "Évolution des habitudes de paiement (tendance)",
                    "calculation_method": "Analyse de l'évolution 2024→2025 + projection tendance",
                    "scale": "0.0 (régression) à 1.0 (forte progression digitale)",
                    "business_impact": "Indique la direction d'évolution du client"
                },
                "financial_sophistication": {
                    "description": "Niveau de sophistication financière et bancaire",
                    "calculation_method": "Segment NMR + revenus + diversité produits + historique",
                    "scale": "0.0 (basique) à 1.0 (très sophistiqué)",
                    "business_impact": "Influence la complexité des produits proposables"
                }
            },
            
            "segments": {
                "TRADITIONNEL_RESISTANT": {
                    "description": "Clients fortement dépendants aux chèques, très résistants au digital",
                    "characteristics": [
                        "Usage intensif des chèques (>60-75% des paiements)",
                        "Très faible adoption mobile banking (<30%)",
                        "Évolution négative ou nulle vers le digital",
                        "Forte préférence pour services traditionnels"
                    ],
                    "typical_profile": "Senior, revenus stables, habitudes très ancrées",
                    "business_strategy": "Accompagnement très progressif, formation intensive, services hybrides",
                    "estimated_population": "15-20%",
                    "criteria": {
                        "check_dependency": ">0.6 (ou >0.75 quelle que soit la situation)",
                        "digital_adoption": "<0.3",
                        "payment_evolution": "<0.4",
                        "financial_sophistication": "0.2-0.7"
                    }
                },
                
                "TRADITIONNEL_MODERE": {
                    "description": "Clients avec usage modéré des chèques, ouverts au changement progressif",
                    "characteristics": [
                        "Usage modéré des chèques (30-60% des paiements)",
                        "Adoption digitale limitée mais supérieure aux résistants (30-60%)",
                        "Évolution lente mais positive vers le digital",
                        "Sensible aux avantages pratiques et économiques"
                    ],
                    "typical_profile": "Adulte actif, revenus moyens, pragmatique et adaptable",
                    "business_strategy": "Incitation douce, démonstration bénéfices concrets, transition graduelle",
                    "estimated_population": "25-30%",
                    "criteria": {
                        "check_dependency": "0.25-0.65 (ranges modérés)",
                        "digital_adoption": "0.25-0.65 (supérieur aux résistants)",
                        "payment_evolution": "≥0.3 (évolution positive)",
                        "financial_sophistication": "0.4-0.8"
                    }
                },
                
                "DIGITAL_TRANSITOIRE": {
                    "description": "Clients en transition active vers le digital",
                    "characteristics": [
                        "Usage décroissant des chèques",
                        "Adoption progressive du mobile banking",
                        "Évolution positive claire",
                        "Expérimente de nouveaux services"
                    ],
                    "typical_profile": "Professionnel, revenus moyens-élevés, adaptable",
                    "business_strategy": "Accélération transition, nouveaux services, support technique",
                    "estimated_population": "25-30%",
                    "criteria": {
                        "check_dependency": "0.2-0.5",
                        "digital_adoption": "0.5-0.7",
                        "payment_evolution": "0.5-0.8",
                        "financial_sophistication": "0.5-0.9"
                    }
                },
                
                "DIGITAL_ADOPTER": {
                    "description": "Clients adopteurs avancés des services digitaux",
                    "characteristics": [
                        "Usage minimal des chèques (<20% des paiements)",
                        "Forte adoption mobile banking (>70%)",
                        "Évolution continue vers le digital",
                        "Utilise services bancaires avancés"
                    ],
                    "typical_profile": "Cadre, revenus élevés, technophile",
                    "business_strategy": "Services premium, innovations, fidélisation par la technologie",
                    "estimated_population": "15-20%",
                    "criteria": {
                        "check_dependency": "<0.2",
                        "digital_adoption": ">0.7",
                        "payment_evolution": ">0.6",
                        "financial_sophistication": "0.7-1.0"
                    }
                },
                
                "DIGITAL_NATIF": {
                    "description": "Clients natifs digitaux, avant-gardistes",
                    "characteristics": [
                        "Usage quasi-nul des chèques (<10%)",
                        "Maîtrise complète des outils digitaux",
                        "Demandeur d'innovations",
                        "Influence les autres clients"
                    ],
                    "typical_profile": "Jeune professionnel, revenus variables, early adopter",
                    "business_strategy": "Partenariat innovation, tests bêta, services exclusifs",
                    "estimated_population": "8-12%",
                    "criteria": {
                        "check_dependency": "<0.1",
                        "digital_adoption": ">0.8",
                        "payment_evolution": ">0.7",
                        "financial_sophistication": "0.6-1.0"
                    }
                },
                
                "EQUILIBRE_MIXTE": {
                    "description": "Clients avec approche équilibrée et flexible",
                    "characteristics": [
                        "Usage adaptatif selon le contexte",
                        "Adoption sélective du digital",
                        "Évolution stable et mesurée",
                        "Privilégie l'efficacité"
                    ],
                    "typical_profile": "Profil mixte, revenus stables, rationnel",
                    "business_strategy": "Solutions sur-mesure, choix multiples, conseil personnalisé",
                    "estimated_population": "7-10%",
                    "criteria": {
                        "check_dependency": "0.2-0.4",
                        "digital_adoption": "0.4-0.7",
                        "payment_evolution": "0.4-0.7",
                        "financial_sophistication": "0.5-0.8"
                    }
                }
            }
        }
        
        # Pondérations pour le calcul des scores
        self.score_weights = {
            "check_dependency": 0.3,      # 30% - Historique usage chèques
            "digital_adoption": 0.25,     # 25% - Adoption actuelle digital
            "payment_evolution": 0.25,    # 25% - Tendance d'évolution
            "financial_sophistication": 0.2  # 20% - Sophistication financière
        }
        
        print("[SEGMENTATION] Moteur de segmentation comportementale initialisé")
    
    def analyze_client_behavior(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse comportementale complète d'un client."""
        
        # Calcul des 4 dimensions comportementales
        check_dependency = self._calculate_check_dependency_enhanced(client_data)
        digital_adoption = self._calculate_digital_adoption_enhanced(client_data)
        payment_evolution = self._calculate_payment_evolution_enhanced(client_data)
        financial_sophistication = self._calculate_financial_sophistication(client_data)
        
        # Détermination du segment comportemental
        behavior_segment = self._determine_behavior_segment_enhanced(
            check_dependency, digital_adoption, payment_evolution, financial_sophistication
        )
        
        # Score composite de modernité bancaire
        modernity_score = self._calculate_modernity_score(
            check_dependency, digital_adoption, payment_evolution, financial_sophistication
        )
        
        # Profil de recommandation
        recommendation_profile = self._generate_recommendation_profile(behavior_segment, client_data)
        
        return {
            'behavioral_scores': {
                'check_dependency_score': check_dependency,
                'digital_adoption_score': digital_adoption,
                'payment_evolution_score': payment_evolution,
                'financial_sophistication_score': financial_sophistication,
                'modernity_score': modernity_score
            },
            'behavior_segment': behavior_segment,
            'segment_details': self.segmentation_criteria['segments'][behavior_segment],
            'recommendation_profile': recommendation_profile,
            'analysis_metadata': {
                'client_id': client_data.get('CLI', 'N/A'),
                'analysis_confidence': self._calculate_analysis_confidence(client_data),
                'segment_probability': self._calculate_segment_probability(
                    check_dependency, digital_adoption, payment_evolution, financial_sophistication
                )
            }
        }
    
    def _calculate_check_dependency_enhanced(self, client_data: Dict[str, Any]) -> float:
        """Calcul amélioré de la dépendance aux chèques."""
        
        # Données de base
        nbr_cheques_2024 = client_data.get('Nbr_Cheques_2024', 0)
        total_transactions = max(client_data.get('Nbr_Transactions_2025', 1), 1)
        ratio_cheques = client_data.get('Ratio_Cheques_Paiements', 0.0)
        
        # Score principal basé sur le ratio réel
        if ratio_cheques > 0:
            primary_score = min(ratio_cheques, 1.0)
        else:
            # Fallback sur le calcul historique
            primary_score = min(nbr_cheques_2024 / max(total_transactions, 1), 1.0)
        
        # Ajustements contextuels
        montant_moyen_cheque = client_data.get('Montant_Moyen_Cheque', 0)
        montant_moyen_alt = client_data.get('Montant_Moyen_Alternative', 1)
        
        # Bonus si les chèques sont pour des gros montants (dépendance structurelle)
        if montant_moyen_cheque > montant_moyen_alt * 3:
            primary_score += 0.1
        
        # Pénalité si évolution négative (réduction dépendance)
        ecart_cheques = client_data.get('Ecart_Nbr_Cheques_2024_2025', 0)
        if ecart_cheques < -2:
            primary_score *= 0.8
        
        return max(0.0, min(primary_score, 1.0))
    
    def _calculate_digital_adoption_enhanced(self, client_data: Dict[str, Any]) -> float:
        """Calcul amélioré de l'adoption digitale."""
        
        # Composantes principales
        mobile_banking = client_data.get('Utilise_Mobile_Banking', 0)
        nb_methodes = client_data.get('Nombre_Methodes_Paiement', 1)
        
        # Score de base mobile banking (pondération réduite pour gradation)
        mobile_score = 0.5 if mobile_banking else 0.0  # Réduit de 0.6 à 0.5
        
        # Score diversité des méthodes (plus de méthodes = plus digital)
        diversity_score = min(nb_methodes / 6, 0.3)  # Augmenté à 30% pour compenser
        
        # Bonus pour les segments technologiques
        segment = client_data.get('Segment_NMR', '')
        if segment in ['S1 Excellence', 'S2 Premium']:
            tech_bonus = 0.1  # Segments premium plus enclins au digital
        elif segment == 'S4 Avenir':
            tech_bonus = 0.15  # Segment jeune très digital
        else:
            tech_bonus = 0.0
        
        # Bonus évolution positive
        ecart_cheques = client_data.get('Ecart_Nbr_Cheques_2024_2025', 0)
        if ecart_cheques < -3:  # Forte réduction chèques = adoption digital
            evolution_bonus = 0.1
        else:
            evolution_bonus = 0.0
        
        total_score = mobile_score + diversity_score + tech_bonus + evolution_bonus
        
        return max(0.0, min(total_score, 1.0))
    
    def _calculate_payment_evolution_enhanced(self, client_data: Dict[str, Any]) -> float:
        """Calcul amélioré de l'évolution des paiements."""
        
        ecart_cheques = client_data.get('Ecart_Nbr_Cheques_2024_2025', 0)
        ecart_montant = client_data.get('Ecart_Montant_Max_2024_2025', 0)
        
        # Score principal basé sur l'évolution du nombre de chèques
        if ecart_cheques < -5:  # Forte réduction
            evolution_score = 0.9
        elif ecart_cheques < -2:  # Réduction modérée
            evolution_score = 0.7
        elif ecart_cheques < 0:  # Légère réduction
            evolution_score = 0.6
        elif ecart_cheques == 0:  # Stabilité
            evolution_score = 0.5
        elif ecart_cheques < 3:  # Légère augmentation
            evolution_score = 0.3
        else:  # Forte augmentation
            evolution_score = 0.1
        
        # Ajustement selon l'évolution des montants
        if ecart_montant > 0 and ecart_cheques <= 0:
            # Montants en hausse mais chèques stables/baisse = digitalisation
            evolution_score += 0.1
        elif ecart_montant < 0 and ecart_cheques < 0:
            # Montants et chèques en baisse = vraie transition
            evolution_score += 0.15
        
        # Prise en compte mobile banking comme indicateur d'évolution
        mobile_banking = client_data.get('Utilise_Mobile_Banking', 0)
        if mobile_banking:
            evolution_score += 0.1
        
        return max(0.0, min(evolution_score, 1.0))
    
    def _calculate_financial_sophistication(self, client_data: Dict[str, Any]) -> float:
        """Calcul de la sophistication financière."""
        
        # Base: segment NMR
        segment = client_data.get('Segment_NMR', 'S3 Essentiel')
        segment_scores = {
            'S1 Excellence': 0.9,
            'S2 Premium': 0.8,
            'S3 Essentiel': 0.6,
            'S4 Avenir': 0.7,  # Jeunes = potentiel élevé
            'S5 Univers': 0.4,
            'NON SEGMENTE': 0.3
        }
        
        base_score = segment_scores.get(segment, 0.5)
        
        # Ajustement revenu
        revenu = client_data.get('Revenu_Estime', 30000)
        if revenu > 100000:
            revenue_bonus = 0.1
        elif revenu > 60000:
            revenue_bonus = 0.05
        elif revenu < 25000:
            revenue_bonus = -0.1
        else:
            revenue_bonus = 0.0
        
        # Diversité des méthodes de paiement (sophistication comportementale)
        nb_methodes = client_data.get('Nombre_Methodes_Paiement', 1)
        method_score = min(nb_methodes / 8, 0.1)  # Max 10% pour diversité
        
        # Dérogations demandées = sophistication (connaissance des produits)
        derogation = client_data.get('A_Demande_Derogation', 0)
        sophistication_bonus = 0.05 if derogation else 0.0
        
        total_score = base_score + revenue_bonus + method_score + sophistication_bonus
        
        return max(0.0, min(total_score, 1.0))
    
    def _determine_behavior_segment_enhanced(self, check_dep: float, digital_adop: float, 
                                           payment_evol: float, financial_soph: float) -> str:
        """Détermine le segment comportemental avec logique corrigée et cohérente."""
        
        # LOGIQUE CORRIGÉE : Cohérence entre noms de segments et scores
        
        # 1. DIGITAL_NATIF (critères très stricts - clients 100% digitaux)
        if (check_dep < 0.15 and digital_adop > 0.8 and 
            payment_evol > 0.7 and financial_soph > 0.6):
            return "DIGITAL_NATIF"
        
        # 2. DIGITAL_ADOPTER (forte adoption digitale)
        if (check_dep < 0.25 and digital_adop > 0.7 and payment_evol > 0.6):
            return "DIGITAL_ADOPTER"
        
        # 3. DIGITAL_TRANSITOIRE (transition active vers digital)
        if (digital_adop >= 0.5 and payment_evol >= 0.5 and check_dep < 0.6):
            return "DIGITAL_TRANSITOIRE"
        
        # 4. TRADITIONNEL_RESISTANT (CORRIGÉ - très résistant au digital)
        # Logique: Forte dépendance chèques ET faible adoption digitale
        if (check_dep > 0.6 and digital_adop < 0.3 and payment_evol < 0.4):
            return "TRADITIONNEL_RESISTANT"
        
        # 5. Cas supplémentaires RESISTANT (forte dépendance chèques)
        if check_dep > 0.75:  # Très forte dépendance = résistant même avec autres scores
            return "TRADITIONNEL_RESISTANT"
        
        # 6. TRADITIONNEL_MODERE (CORRIGÉ - usage modéré dans ranges définis)
        # Logique: Scores dans ranges modérés (ni résistant ni digital)
        if (0.3 <= check_dep <= 0.6 and 0.3 <= digital_adop <= 0.6 and 0.3 <= payment_evol <= 0.6):
            return "TRADITIONNEL_MODERE"
        
        # 7. Cas supplémentaires MODERE (entre traditionnel et digital)
        if (0.25 < check_dep <= 0.65 and 0.25 < digital_adop < 0.65 and payment_evol >= 0.3):
            return "TRADITIONNEL_MODERE"
        
        # 8. EQUILIBRE_MIXTE (approche équilibrée - reste des cas)
        return "EQUILIBRE_MIXTE"
    
    def _calculate_modernity_score(self, check_dep: float, digital_adop: float, 
                                 payment_evol: float, financial_soph: float) -> float:
        """Calcule un score composite de modernité bancaire."""
        
        # Score inversé pour check_dependency (moins de chèques = plus moderne)
        modern_check_score = 1.0 - check_dep
        
        # Score pondéré
        modernity = (
            modern_check_score * self.score_weights['check_dependency'] +
            digital_adop * self.score_weights['digital_adoption'] +
            payment_evol * self.score_weights['payment_evolution'] +
            financial_soph * self.score_weights['financial_sophistication']
        )
        
        return max(0.0, min(modernity, 1.0))
    
    def _generate_recommendation_profile(self, segment: str, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Génère un profil de recommandation basé sur le segment."""
        
        segment_details = self.segmentation_criteria['segments'][segment]
        
        # Stratégies spécifiques par segment
        strategy_mapping = {
            "TRADITIONNEL_RESISTANT": {
                "approach": "Accompagnement progressif",
                "priority_services": ["formation_digital", "accompagnement_personnel", "services_hybrides"],
                "communication_style": "Rassurante, éducative, graduelle",
                "success_factors": ["Confiance", "Simplicité", "Support humain"]
            },
            "TRADITIONNEL_MODERE": {
                "approach": "Incitation douce",
                "priority_services": ["mobile_banking_basic", "carte_contactless", "formation_practical"],
                "communication_style": "Pragmatique, bénéfices concrets",
                "success_factors": ["Utilité pratique", "Économies", "Facilité"]
            },
            "DIGITAL_TRANSITOIRE": {
                "approach": "Accélération transition",
                "priority_services": ["mobile_banking_advanced", "paiements_digitaux", "services_online"],
                "communication_style": "Encourageante, technique modérée",
                "success_factors": ["Efficacité", "Innovation", "Support technique"]
            },
            "DIGITAL_ADOPTER": {
                "approach": "Services premium",
                "priority_services": ["services_premium", "innovations", "apis_ouvertes"],
                "communication_style": "Technique, avantages avancés",
                "success_factors": ["Performance", "Exclusivité", "Technologie"]
            },
            "DIGITAL_NATIF": {
                "approach": "Partenariat innovation",
                "priority_services": ["beta_testing", "services_exclusifs", "personnalisation_avancee"],
                "communication_style": "Collaborative, avant-gardiste",
                "success_factors": ["Innovation", "Personnalisation", "Reconnaissance"]
            },
            "EQUILIBRE_MIXTE": {
                "approach": "Solutions sur-mesure",
                "priority_services": ["options_multiples", "conseil_personnalise", "flexibilite"],
                "communication_style": "Consultative, rationnelle",
                "success_factors": ["Choix", "Rationalité", "Efficience"]
            }
        }
        
        return {
            "segment_strategy": strategy_mapping.get(segment, {}),
            "estimated_conversion_rate": self._estimate_conversion_rate(segment),
            "recommended_contact_frequency": self._get_contact_frequency(segment),
            "priority_metrics": self._get_priority_metrics(segment)
        }
    
    def _calculate_analysis_confidence(self, client_data: Dict[str, Any]) -> float:
        """Calcule la confiance dans l'analyse comportementale."""
        
        required_fields = [
            'Nbr_Cheques_2024', 'Utilise_Mobile_Banking', 'Nombre_Methodes_Paiement',
            'Ecart_Nbr_Cheques_2024_2025', 'Segment_NMR', 'Revenu_Estime'
        ]
        
        available_fields = sum(1 for field in required_fields if client_data.get(field) is not None)
        data_completeness = available_fields / len(required_fields)
        
        # Bonus pour données de qualité
        if client_data.get('Ratio_Cheques_Paiements', 0) > 0:
            data_completeness += 0.1
        
        return min(data_completeness, 1.0)
    
    def _calculate_segment_probability(self, check_dep: float, digital_adop: float, 
                                     payment_evol: float, financial_soph: float) -> Dict[str, float]:
        """Calcule la probabilité d'appartenance à chaque segment."""
        
        probabilities = {}
        
        # Calcul des distances aux centres des segments (approximation)
        segment_centers = {
            "TRADITIONNEL_RESISTANT": (0.8, 0.2, 0.2, 0.5),
            "TRADITIONNEL_MODERE": (0.5, 0.4, 0.4, 0.6),
            "DIGITAL_TRANSITOIRE": (0.3, 0.6, 0.6, 0.7),
            "DIGITAL_ADOPTER": (0.1, 0.8, 0.8, 0.8),
            "DIGITAL_NATIF": (0.05, 0.9, 0.9, 0.7),
            "EQUILIBRE_MIXTE": (0.3, 0.5, 0.5, 0.6)
        }
        
        scores = (check_dep, digital_adop, payment_evol, financial_soph)
        
        for segment, center in segment_centers.items():
            # Distance euclidienne
            distance = sum((s - c) ** 2 for s, c in zip(scores, center)) ** 0.5
            # Conversion en probabilité (plus proche = plus probable)
            probability = max(0, 1 - (distance / 2))  # Normalisation
            probabilities[segment] = probability
        
        # Normalisation pour que la somme = 1
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        return probabilities
    
    def _estimate_conversion_rate(self, segment: str) -> float:
        """Estime le taux de conversion par segment."""
        rates = {
            "TRADITIONNEL_RESISTANT": 0.15,
            "TRADITIONNEL_MODERE": 0.35,
            "DIGITAL_TRANSITOIRE": 0.65,
            "DIGITAL_ADOPTER": 0.80,
            "DIGITAL_NATIF": 0.90,
            "EQUILIBRE_MIXTE": 0.50
        }
        return rates.get(segment, 0.50)
    
    def _get_contact_frequency(self, segment: str) -> str:
        """Recommande la fréquence de contact par segment."""
        frequencies = {
            "TRADITIONNEL_RESISTANT": "Mensuelle - approche douce",
            "TRADITIONNEL_MODERE": "Bi-mensuelle - opportunités ciblées",
            "DIGITAL_TRANSITOIRE": "Hebdomadaire - support transition",
            "DIGITAL_ADOPTER": "Bi-hebdomadaire - nouveautés",
            "DIGITAL_NATIF": "Continue - partenariat innovation",
            "EQUILIBRE_MIXTE": "Mensuelle - solutions personnalisées"
        }
        return frequencies.get(segment, "Mensuelle")
    
    def _get_priority_metrics(self, segment: str) -> List[str]:
        """Définit les métriques prioritaires par segment."""
        metrics = {
            "TRADITIONNEL_RESISTANT": ["Taux formation", "Satisfaction support", "Réduction progressive chèques"],
            "TRADITIONNEL_MODERE": ["Adoption services", "Économies réalisées", "Temps gagné"],
            "DIGITAL_TRANSITOIRE": ["Vitesse transition", "Utilisation mobile", "Abandon chèques"],
            "DIGITAL_ADOPTER": ["Usage services premium", "NPS", "Cross-selling"],
            "DIGITAL_NATIF": ["Innovation adoptée", "Feedback produits", "Influence réseau"],
            "EQUILIBRE_MIXTE": ["Satisfaction globale", "Équilibre usage", "Efficience"]
        }
        return metrics.get(segment, ["Satisfaction client"])
    
    def get_segmentation_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de la segmentation comportementale."""
        return {
            "methodology": "Segmentation comportementale multi-dimensionnelle",
            "dimensions": len(self.segmentation_criteria["dimensions"]),
            "segments": len(self.segmentation_criteria["segments"]),
            "criteria_summary": {
                name: details["description"] 
                for name, details in self.segmentation_criteria["dimensions"].items()
            },
            "segment_summary": {
                name: {
                    "description": details["description"],
                    "population": details["estimated_population"],
                    "strategy": details["business_strategy"]
                }
                for name, details in self.segmentation_criteria["segments"].items()
            }
        }