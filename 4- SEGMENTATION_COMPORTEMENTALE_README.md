# Système de Segmentation Comportementale Bancaire

## Vue d'ensemble

Le système de segmentation comportementale est un module avancé qui analyse le comportement financier des clients bancaires pour proposer des recommandations personnalisées et ciblées. Il utilise une approche multi-dimensionnelle basée sur l'analyse des habitudes de paiement, l'adoption du digital et le profil de sophistication financière.

## Architecture du Système

### Fichier Principal
- **`src/utils/behavioral_segmentation.py`** : Moteur principal de segmentation comportementale

### Classe Principale
- **`BehavioralSegmentationEngine`** : Classe qui implémente toute la logique de segmentation

## Méthodologie de Segmentation

### 4 Dimensions Comportementales

#### 1. **Dépendance aux Chèques (check_dependency)**
- **Description** : Niveau de dépendance aux chèques comme méthode de paiement
- **Calcul** : Ratio chèques / total transactions + pondération historique
- **Échelle** : 0.0 (aucune dépendance) à 1.0 (très forte dépendance)
- **Impact Business** : Plus le score est élevé, plus le client résiste au digital

**Facteurs pris en compte :**
- Ratio de paiement par chèques vs alternatives
- Montant moyen des chèques vs alternatives
- Évolution temporelle (2024 → 2025)
- Ajustements contextuels pour usage structurel

#### 2. **Adoption Digitale (digital_adoption)**
- **Description** : Niveau d'adoption des services bancaires digitaux
- **Calcul** : Mobile banking + diversité méthodes paiement + usage services en ligne
- **Échelle** : 0.0 (aucune adoption) à 1.0 (adoption complète)
- **Impact Business** : Plus le score est élevé, plus le client est réceptif aux nouveaux services

**Facteurs pris en compte :**
- Utilisation mobile banking (pondération 60%)
- Diversité des méthodes de paiement (25%)
- Segment client premium/technologique (10%)
- Évolution positive récente (5%)

#### 3. **Évolution des Paiements (payment_evolution)**
- **Description** : Évolution des habitudes de paiement (tendance)
- **Calcul** : Analyse de l'évolution 2024→2025 + projection tendance
- **Échelle** : 0.0 (régression) à 1.0 (forte progression digitale)
- **Impact Business** : Indique la direction d'évolution du client

**Logique d'évaluation :**
- Forte réduction chèques (>5) : Score 0.9
- Réduction modérée (2-5) : Score 0.7
- Stabilité : Score 0.5
- Augmentation : Score décroissant

#### 4. **Sophistication Financière (financial_sophistication)**
- **Description** : Niveau de sophistication financière et bancaire
- **Calcul** : Segment NMR + revenus + diversité produits + historique
- **Échelle** : 0.0 (basique) à 1.0 (très sophistiqué)
- **Impact Business** : Influence la complexité des produits proposables

**Mapping des segments :**
- S1 Excellence : 0.9
- S2 Premium : 0.8
- S3 Essentiel : 0.6
- S4 Avenir : 0.7 (potentiel jeune)
- S5 Univers : 0.4

## 6 Segments Comportementaux

### 1. TRADITIONNEL_RÉSISTANT (15-20% population)
**Profil :** Clients fortement dépendants aux chèques, résistants au digital
- Usage intensif des chèques (>60% des paiements)
- Faible adoption mobile banking (<30%)
- Évolution négative vers le digital
- Préfère les services traditionnels

**Critères de segmentation :**
- check_dependency > 0.6
- digital_adoption < 0.3
- payment_evolution < 0.4

**Stratégie business :** Accompagnement progressif, formation, services hybrides

### 2. TRADITIONNEL_MODÉRÉ (25-30% population)
**Profil :** Clients avec usage modéré des chèques, ouverts au changement
- Usage modéré des chèques (30-60% des paiements)
- Adoption digitale limitée mais présente
- Évolution lente vers le digital
- Sensible aux avantages pratiques

**Critères de segmentation :**
- check_dependency : 0.3-0.6
- digital_adoption : 0.3-0.6
- payment_evolution : 0.3-0.6

**Stratégie business :** Incitation douce, démonstration bénéfices, transition graduelle

### 3. DIGITAL_TRANSITOIRE (25-30% population)
**Profil :** Clients en transition active vers le digital
- Usage décroissant des chèques
- Adoption progressive du mobile banking
- Évolution positive claire
- Expérimente de nouveaux services

**Critères de segmentation :**
- check_dependency : 0.2-0.5
- digital_adoption : 0.5-0.7
- payment_evolution : 0.5-0.8

**Stratégie business :** Accélération transition, nouveaux services, support technique

### 4. DIGITAL_ADOPTER (15-20% population)
**Profil :** Clients adopteurs avancés des services digitaux
- Usage minimal des chèques (<20% des paiements)
- Forte adoption mobile banking (>70%)
- Évolution continue vers le digital
- Utilise services bancaires avancés

**Critères de segmentation :**
- check_dependency < 0.2
- digital_adoption > 0.7
- payment_evolution > 0.6

**Stratégie business :** Services premium, innovations, fidélisation par la technologie

### 5. DIGITAL_NATIF (8-12% population)
**Profil :** Clients natifs digitaux, avant-gardistes
- Usage quasi-nul des chèques (<10%)
- Maîtrise complète des outils digitaux
- Demandeur d'innovations
- Influence les autres clients

**Critères de segmentation :**
- check_dependency < 0.1
- digital_adoption > 0.8
- payment_evolution > 0.7

**Stratégie business :** Partenariat innovation, tests bêta, services exclusifs

### 6. ÉQUILIBRE_MIXTE (7-10% population)
**Profil :** Clients avec approche équilibrée et flexible
- Usage adaptatif selon le contexte
- Adoption sélective du digital
- Évolution stable et mesurée
- Privilégie l'efficacité

**Critères de segmentation :**
- check_dependency : 0.2-0.4
- digital_adoption : 0.4-0.7
- payment_evolution : 0.4-0.7

**Stratégie business :** Solutions sur-mesure, choix multiples, conseil personnalisé

## Algorithme de Segmentation

### Logique Hiérarchique
Le système utilise une approche hiérarchique pour déterminer le segment :

1. **DIGITAL_NATIF** (critères les plus stricts)
2. **DIGITAL_ADOPTER** (forte adoption)
3. **DIGITAL_TRANSITOIRE** (évolution positive)
4. **TRADITIONNEL_RÉSISTANT** (forte résistance)
5. **TRADITIONNEL_MODÉRÉ** (usage modéré)
6. **ÉQUILIBRE_MIXTE** (défaut pour profils mixtes)

### Score de Modernité Composite
Calcul pondéré des 4 dimensions :
- Dépendance chèques : 30% (inversé)
- Adoption digitale : 25%
- Évolution paiements : 25%
- Sophistication financière : 20%

## Utilisation du Système

### Initialisation
```python
from src.utils.behavioral_segmentation import BehavioralSegmentationEngine

# Créer le moteur de segmentation
segmentation_engine = BehavioralSegmentationEngine()
```

### Analyse d'un Client
```python
client_data = {
    'CLI': 'CLIENT_001',
    'Nbr_Cheques_2024': 45,
    'Utilise_Mobile_Banking': 1,
    'Nombre_Methodes_Paiement': 4,
    'Ecart_Nbr_Cheques_2024_2025': -8,
    'Segment_NMR': 'S2 Premium',
    'Revenu_Estime': 65000
}

# Analyser le comportement
analysis = segmentation_engine.analyze_client_behavior(client_data)

# Résultats
segment = analysis['behavior_segment']
scores = analysis['behavioral_scores']
strategy = analysis['recommendation_profile']
confidence = analysis['analysis_metadata']['analysis_confidence']
```

### Résultat d'Analyse
```python
{
    'behavioral_scores': {
        'check_dependency_score': 0.42,
        'digital_adoption_score': 0.75,
        'payment_evolution_score': 0.68,
        'financial_sophistication_score': 0.80,
        'modernity_score': 0.70
    },
    'behavior_segment': 'DIGITAL_TRANSITOIRE',
    'segment_details': {
        'description': 'Clients en transition active vers le digital',
        'business_strategy': 'Accélération transition, nouveaux services, support technique'
    },
    'recommendation_profile': {
        'segment_strategy': {
            'approach': 'Accélération transition',
            'priority_services': ['mobile_banking_advanced', 'paiements_digitaux'],
            'communication_style': 'Encourageante, technique modérée'
        },
        'estimated_conversion_rate': 0.65,
        'recommended_contact_frequency': 'Hebdomadaire - support transition'
    }
}
```

## Intégration avec le Système de Recommandation

### Lien avec RecommendationEngine
Le système de segmentation s'intègre avec le moteur de recommandation existant :

```python
# Dans recommendation_engine.py
from utils.behavioral_segmentation import BehavioralSegmentationEngine

class PersonalizedRecommendationEngine:
    def __init__(self):
        self.behavior_analyzer = ClientBehaviorAnalyzer()  # Ancien système
        self.segmentation_engine = BehavioralSegmentationEngine()  # Nouveau système
    
    def generate_recommendations(self, client_data):
        # Utiliser la segmentation comportementale avancée
        behavioral_analysis = self.segmentation_engine.analyze_client_behavior(client_data)
        segment = behavioral_analysis['behavior_segment']
        
        # Adapter les recommandations selon le segment
        recommendations = self._get_segment_specific_recommendations(segment, client_data)
        return recommendations
```

## Métriques et Validation

### Confiance d'Analyse
Basée sur la complétude des données :
- Champs requis : 6 champs critiques
- Bonus qualité : ratio chèques/paiements disponible
- Score : 0.0 à 1.0

### Probabilité de Segment
Calcul des distances euclidiennes aux centres des segments :
- Distance aux 6 centres de segments
- Conversion en probabilités normalisées
- Retour des probabilités pour tous les segments

### Métriques par Segment
- **Taux de conversion estimés** : 15% à 90% selon segment
- **Fréquence de contact recommandée** : Mensuelle à continue
- **Métriques prioritaires** : Spécifiques par segment

## Validation et Tests

### Tests Automatisés
Le système inclut des tests pour :
- Cohérence des scores (0.0 à 1.0)
- Logique de segmentation
- Calculs des probabilités
- Intégration avec données réelles

### Validation Business
- Vérification de la distribution des segments
- Cohérence avec le comportement observé
- Validation des stratégies par segment

## Maintenance et Évolution

### Points d'Attention
1. **Calibration périodique** des seuils de segmentation
2. **Mise à jour** des critères selon évolution marché
3. **Validation** des taux de conversion réels
4. **Optimisation** des pondérations selon performance

### Extensions Possibles
- Segmentation géographique
- Analyse prédictive d'évolution de segment
- Personnalisation des seuils par région
- Intégration données externes (âge, profession)

## Documentation Technique

### Dépendances
- pandas : Manipulation des données
- numpy : Calculs numériques
- typing : Annotations de type
- json : Sérialisation des résultats

### Structure des Données
Tous les champs du dictionnaire `data_dictionary.json` sont supportés avec gestion des valeurs manquantes.

### Performance
- Analyse d'un client : ~5ms
- Traitement batch 1000 clients : ~5s
- Mémoire utilisée : ~10MB pour 10000 clients

---
