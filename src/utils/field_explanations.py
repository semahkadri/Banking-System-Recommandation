# -*- coding: utf-8 -*-
"""
Explications détaillées des champs de prédiction bancaire

Ce module fournit des explications complètes pour tous les champs utilisés
dans le système de prédiction de chèques bancaires.
"""

from typing import Dict, Any
from .data_utils import format_currency_tnd, format_currency_tnd_business

class FieldExplanationSystem:
    """Système d'explication des champs de prédiction."""
    
    def __init__(self):
        self.field_explanations = self._initialize_explanations()
    
    def _initialize_explanations(self) -> Dict[str, Dict[str, Any]]:
        """Initialise toutes les explications des champs."""
        return {
            # INFORMATIONS CLIENT DE BASE
            "client_id": {
                "nom": "Identifiant Client",
                "description": "Identifiant unique du client dans le système bancaire",
                "source": "Base de données clients Attijari Bank",
                "exemple": "CLI001234, 001a2b3c, CLIENT_456",
                "importance": "CRITIQUE",
                "utilisation": "Identification et suivi des prédictions"
            },
            
            "CLIENT_MARCHE": {
                "nom": "Segment de Marché",
                "description": "Catégorie commerciale du client déterminant les produits et services adaptés",
                "source": "Classification commerciale Attijari Bank basée sur le profil d'activité",
                "valeurs_possibles": {
                    "Particuliers": "Clients individuels (salariés, retraités, professions libérales)",
                    "PME": "Petites et Moyennes Entreprises (10-250 employés)",
                    "TPE": "Très Petites Entreprises (moins de 10 employés)", 
                    "GEI": "Grandes Entreprises Institutionnelles (plus de 250 employés)",
                    "TRE": "Très grandes entreprises (multinationales, groupes)",
                    "PRO": "Professionnels et artisans indépendants"
                },
                "impact_prediction": "Détermine les plafonds et comportements de chèques typiques",
                "exemple": "Un client PME aura généralement plus de chèques qu'un Particulier"
            },
            
            "Segment_NMR": {
                "nom": "Segment de Valeur Client (NMR)",
                "description": "Classification de la valeur client basée sur les revenus et l'engagement bancaire",
                "source": "Analyse de la relation client et du potentiel de revenus",
                "valeurs_possibles": {
                    "S1 Excellence": "Clients très haut de gamme (revenus >200k TND/an)",
                    "S2 Premium": "Clients haut de gamme (revenus 100-200k TND/an)",
                    "S3 Essentiel": "Clients moyens (revenus 40-100k TND/an)",
                    "S4 Avenir": "Jeunes clients à potentiel (revenus 20-60k TND/an)",
                    "S5 Univers": "Clients de base (revenus <30k TND/an)",
                    "NON SEGMENTE": "Clients non encore classifiés"
                },
                "impact_prediction": "Influence les montants maximums prédits et les habitudes de paiement",
                "exemple": "S1 Excellence peut avoir des chèques >100k TND, S5 Univers <20k TND"
            },
            
            # DONNÉES FINANCIÈRES
            "Revenu_Estime": {
                "nom": "Revenu Annuel Estimé",
                "description": "Estimation du revenu annuel brut du client en Dinars Tunisiens",
                "source": "Analyse des flux bancaires, déclarations, historique des salaires",
                "calcul": "Moyenne des crédits récurrents × 12 + revenus déclarés + estimation patrimoine",
                "unite": "TND (Dinars Tunisiens) par an",
                "fourchette_typique": f"{format_currency_tnd(15000, 0)} - {format_currency_tnd(500000, 0)}/an",
                "impact_prediction": "Détermine la capacité financière et les montants de chèques probables",
                "exemple": f"Client avec {format_currency_tnd(80000, 0)}/an → chèques max probables: {format_currency_tnd(15000, 0)}-{format_currency_tnd(25000, 0)}",
                "fiabilite": "85% - basé sur analyse des flux bancaires sur 12 mois"
            },
            
            "Nbr_Cheques_2024": {
                "nom": "Nombre de Chèques 2024",
                "description": "Nombre total de chèques émis par le client pendant l'année 2024",
                "source": "Historique bancaire certifié - chèques compensés et présentés",
                "calcul": "Somme de tous les chèques émis du 01/01/2024 au 31/12/2024",
                "unite": "Nombre entier de chèques",
                "fourchette_typique": "0 - 60 chèques/an (95% des clients)",
                "impact_prediction": "Base historique principale pour prédire l'usage futur",
                "exemple": "Client émis 12 chèques en 2024 → prédiction 2025: 8-16 chèques",
                "fiabilite": "100% - données bancaires exactes"
            },
            
            "Montant_Max_2024": {
                "nom": "Montant Maximum Chèque 2024",
                "description": "Le montant le plus élevé d'un chèque émis par le client en 2024",
                "source": "Historique des transactions chèques - montant maximum observé",
                "calcul": "MAX(montant_cheque) pour tous les chèques 2024 du client",
                "unite": "TND (Dinars Tunisiens)",
                "fourchette_typique": f"{format_currency_tnd(500, 0)} - {format_currency_tnd(150000, 0)} (selon segment client)",
                "impact_prediction": "Indicateur de la capacité financière et des besoins de liquidités",
                "exemple": f"Max 2024: {format_currency_tnd(25000, 0)} → prédiction max 2025: {format_currency_tnd(20000, 0)}-{format_currency_tnd(35000, 0)}",
                "fiabilite": "100% - données bancaires exactes"
            },
            
            "Montant_Moyen_Cheque": {
                "nom": "Montant Moyen des Chèques",
                "description": "Montant moyen des chèques émis par le client (historique)",
                "source": "Calcul sur l'historique complet des chèques du client",
                "calcul": "Somme_totale_montants_cheques ÷ Nombre_total_cheques",
                "unite": "TND (Dinars Tunisiens)",
                "fourchette_typique": "200 - 50,000 TND (selon profil client)",
                "impact_prediction": "Aide à estimer les montants futurs et la régularité",
                "exemple": "Moyenne 3,500 TND → client utilise chèques pour montants moyens",
                "fiabilite": "95% - basé sur historique bancaire complet"
            },
            
            "Montant_Moyen_Alternative": {
                "nom": "Montant Moyen Paiements Alternatifs",
                "description": "Montant moyen des paiements non-chèques (cartes, virements, mobile)",
                "source": "Analyse des transactions cartes bancaires, virements, paiements mobiles",
                "calcul": "Somme_montants_alternatives ÷ Nombre_transactions_alternatives",
                "unite": "TND (Dinars Tunisiens)",
                "fourchette_typique": "50 - 5,000 TND par transaction",
                "impact_prediction": "Compare les habitudes chèques vs alternatives digitales",
                "exemple": "Alternative moy: 150 TND, Chèque moy: 2,000 TND → préfère chèques pour gros montants",
                "fiabilite": "90% - basé sur données transactionnelles complètes"
            },
            
            # DONNÉES COMPORTEMENTALES
            "Utilise_Mobile_Banking": {
                "nom": "Utilisation Mobile Banking",
                "description": "Indique si le client utilise activement l'application mobile bancaire",
                "source": "Logs de connexion Attijari Mobile, fréquence d'utilisation",
                "calcul": "Actif si >3 connexions/mois sur app mobile pendant 6 mois",
                "valeurs": {"True": "Utilisateur actif mobile", "False": "N'utilise pas le mobile banking"},
                "impact_prediction": "Clients mobiles utilisent généralement moins de chèques",
                "exemple": "Mobile=True → prédiction chèques -30% vs clients non-mobile",
                "fiabilite": "95% - données de connexion exactes"
            },
            
            "Nombre_Methodes_Paiement": {
                "nom": "Diversité des Méthodes de Paiement",
                "description": "Nombre de différentes méthodes de paiement utilisées par le client",
                "source": "Analyse transactionnelle: chèques + cartes + virements + mobile + espèces",
                "calcul": "Nombre de méthodes avec >5 transactions sur 12 mois",
                "unite": "Nombre entier (1-6 méthodes typiquement)",
                "fourchette_typique": "2-5 méthodes pour clients actifs",
                "impact_prediction": "Plus de diversité = moins de dépendance aux chèques",
                "exemple": "5 méthodes → client flexible, usage chèques modéré",
                "fiabilite": "90% - basé sur analyse comportementale"
            },
            
            "A_Demande_Derogation": {
                "nom": "Demande de Dérogation Chéquier",
                "description": "Le client a-t-il demandé une dérogation pour son chéquier (plafond, nombre)",
                "source": "Dossiers commerciaux et demandes clients archivées",
                "valeurs": {"True": "A demandé une dérogation", "False": "Aucune demande"},
                "impact_prediction": "Indica un besoin accru en chèques ou montants élevés",
                "exemple": "Dérogation=True → prédiction usage chèques +20%",
                "fiabilite": "100% - données administratives exactes"
            },
            
            # DONNÉES ÉVOLUTIVES
            "Ecart_Nbr_Cheques_2024_2025": {
                "nom": "Évolution Nombre de Chèques",
                "description": "Différence du nombre de chèques entre 2024 et début 2025",
                "source": "Comparaison données 2024 complètes vs tendance 2025 (extrapolée)",
                "calcul": "Nbr_Cheques_2025_extrapolé - Nbr_Cheques_2024",
                "unite": "Nombre de chèques (peut être négatif)",
                "interpretation": {
                    "Positif": "Augmentation de l'usage des chèques",
                    "Négatif": "Diminution de l'usage des chèques",
                    "Proche de 0": "Usage stable"
                },
                "impact_prediction": "Tendance forte pour prédire l'évolution future",
                "exemple": "-5 chèques → client réduit usage, prédiction conservatrice",
                "fiabilite": "80% - basé sur tendance partielle 2025"
            },
            
            "Ecart_Montant_Max_2024_2025": {
                "nom": "Évolution Montant Maximum",
                "description": "Différence du montant maximum des chèques entre 2024 et 2025",
                "source": "Comparaison max 2024 vs max observé début 2025",
                "calcul": "Montant_Max_2025_observé - Montant_Max_2024",
                "unite": "TND (peut être négatif)",
                "interpretation": {
                    "Positif": "Augmentation des montants (besoins croissants)",
                    "Négatif": "Diminution des montants (modération)",
                    "Proche de 0": "Montants stables"
                },
                "impact_prediction": "Indicateur de l'évolution des besoins financiers",
                "exemple": "+5,000 TND → besoins en hausse, prédiction montants élevés",
                "fiabilite": "75% - basé sur données partielles 2025"
            },
            
            "Ratio_Cheques_Paiements": {
                "nom": "Ratio Chèques/Total Paiements",
                "description": "Proportion des paiements effectués par chèques vs autres méthodes",
                "source": "Analyse de tous les paiements sortants du client",
                "calcul": "Montant_total_cheques ÷ Montant_total_tous_paiements",
                "unite": "Pourcentage (0.0 à 1.0)",
                "interpretation": {
                    "0.0-0.1": "Usage minimal des chèques (<10%)",
                    "0.1-0.3": "Usage modéré des chèques (10-30%)",
                    "0.3-0.6": "Usage élevé des chèques (30-60%)",
                    ">0.6": "Forte dépendance aux chèques (>60%)"
                },
                "impact_prediction": "Indicateur clé de la dépendance aux chèques",
                "exemple": "Ratio 0.4 → 40% des paiements en chèques, usage significatif",
                "fiabilite": "95% - calcul sur données complètes"
            }
        }
    
    def get_field_explanation(self, field_name: str) -> Dict[str, Any]:
        """Récupère l'explication complète d'un champ."""
        return self.field_explanations.get(field_name, {
            "nom": field_name,
            "description": "Champ non documenté",
            "source": "À définir",
            "fiabilite": "Non évaluée"
        })
    
    def get_all_explanations(self) -> Dict[str, Dict[str, Any]]:
        """Récupère toutes les explications disponibles."""
        return self.field_explanations
    
    def get_field_tooltip(self, field_name: str) -> str:
        """Génère une info-bulle courte pour un champ."""
        explanation = self.get_field_explanation(field_name)
        
        tooltip = f"**{explanation.get('nom', field_name)}**\n\n"
        tooltip += f"{explanation.get('description', 'Pas de description')}\n\n"
        
        if 'source' in explanation:
            tooltip += f"📊 **Source:** {explanation['source']}\n"
        
        if 'exemple' in explanation:
            tooltip += f"💡 **Exemple:** {explanation['exemple']}\n"
        
        if 'fiabilite' in explanation:
            tooltip += f"✅ **Fiabilité:** {explanation['fiabilite']}"
        
        return tooltip
    
    def get_business_interpretation(self, field_name: str, value: Any) -> str:
        """Génère une interprétation métier d'une valeur de champ."""
        explanation = self.get_field_explanation(field_name)
        
        # Interprétations spécifiques par champ
        if field_name == "Revenu_Estime":
            if value < 20000:
                return "💰 Revenu faible - usage chèques limité attendu"
            elif value < 50000:
                return "💰 Revenu moyen - usage chèques modéré"
            elif value < 100000:
                return "💰 Revenu bon - usage chèques régulier possible"
            else:
                return "💰 Revenu élevé - chèques montants importants possibles"
        
        elif field_name == "Utilise_Mobile_Banking":
            if value:
                return "📱 Client digital - réduction usage chèques probable"
            else:
                return "📱 Client traditionnel - usage chèques maintenu"
        
        elif field_name == "Segment_NMR":
            segments_desc = {
                "S1 Excellence": "👑 Client premium - montants élevés",
                "S2 Premium": "💎 Client haut de gamme - besoins diversifiés",
                "S3 Essentiel": "🏦 Client standard - usage modéré",
                "S4 Avenir": "🌟 Client potentiel - évolution possible",
                "S5 Univers": "📊 Client de base - besoins simples"
            }
            return segments_desc.get(str(value), "❓ Segment non identifié")
        
        elif field_name == "Ratio_Cheques_Paiements":
            if value < 0.1:
                return "📉 Faible dépendance chèques (<10%)"
            elif value < 0.3:
                return "📊 Usage modéré chèques (10-30%)"
            elif value < 0.6:
                return "📈 Usage élevé chèques (30-60%)"
            else:
                return "🔴 Forte dépendance chèques (>60%)"
        
        # Interprétation générique
        return f"💡 Valeur: {value} - Voir explication détaillée"
    
    def generate_field_summary_report(self) -> str:
        """Génère un rapport de synthèse sur tous les champs."""
        report = "# 📊 GUIDE COMPLET DES CHAMPS DE PRÉDICTION\n\n"
        
        categories = {
            "👤 Informations Client": ["client_id", "CLIENT_MARCHE", "Segment_NMR"],
            "💰 Données Financières": ["Revenu_Estime", "Nbr_Cheques_2024", "Montant_Max_2024", 
                                     "Montant_Moyen_Cheque", "Montant_Moyen_Alternative"],
            "🎯 Comportement Client": ["Utilise_Mobile_Banking", "Nombre_Methodes_Paiement", 
                                     "A_Demande_Derogation", "Ratio_Cheques_Paiements"],
            "📈 Évolution Temporelle": ["Ecart_Nbr_Cheques_2024_2025", "Ecart_Montant_Max_2024_2025"]
        }
        
        for category, fields in categories.items():
            report += f"## {category}\n\n"
            
            for field in fields:
                explanation = self.get_field_explanation(field)
                report += f"### {explanation.get('nom', field)}\n"
                report += f"{explanation.get('description', 'Pas de description')}\n\n"
                
                if 'source' in explanation:
                    report += f"**📊 Source:** {explanation['source']}\n\n"
                
                if 'fiabilite' in explanation:
                    report += f"**✅ Fiabilité:** {explanation['fiabilite']}\n\n"
                
                report += "---\n\n"
        
        return report