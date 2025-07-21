# -*- coding: utf-8 -*-
"""
Utilitaires pour la gestion cohérente des identifiants clients
"""

from typing import Dict, Any, Optional

def extract_client_id(client_data: Dict[str, Any]) -> str:
    """
    Extrait l'identifiant client de manière cohérente depuis différentes sources de données.
    
    Args:
        client_data: Données du client contenant potentiellement différents champs d'ID
        
    Returns:
        L'identifiant client ou 'unknown' si non trouvé
    """
    # Priorité des champs d'ID : CLI (dataset final) > client_id > CLI_client (historique)
    return (
        client_data.get('CLI') or 
        client_data.get('client_id') or 
        client_data.get('CLI_client') or 
        'unknown'
    )

def normalize_client_data(client_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise les données client pour utiliser un champ d'ID cohérent.
    
    Args:
        client_data: Données brutes du client
        
    Returns:
        Données normalisées avec champ 'client_id' standardisé
    """
    normalized_data = client_data.copy()
    
    # Standardiser l'ID client
    client_id = extract_client_id(client_data)
    normalized_data['client_id'] = client_id
    
    # Conserver les champs originaux pour compatibilité
    if not normalized_data.get('CLI'):
        normalized_data['CLI'] = client_id
    
    return normalized_data

def validate_client_id(client_id: str) -> bool:
    """
    Valide si un identifiant client est correct.
    
    Args:
        client_id: Identifiant à valider
        
    Returns:
        True si l'ID est valide, False sinon
    """
    if not client_id or client_id == 'unknown':
        return False
    
    # Les IDs réels peuvent être:
    # - Numériques: "0", "1234"
    # - Alphanumériques: "01af40c5", "4da2632c0d"
    # - Mixtes: "00002a93"
    return len(client_id) > 0 and client_id.replace('_', '').replace('-', '').isalnum()

def get_id_type(client_id: str) -> str:
    """
    Détermine le type d'identifiant client.
    
    Args:
        client_id: Identifiant client
        
    Returns:
        Type d'ID: 'numeric', 'alphanumeric', 'mixed', ou 'invalid'
    """
    if not validate_client_id(client_id):
        return 'invalid'
    
    clean_id = client_id.replace('_', '').replace('-', '')
    
    if clean_id.isdigit():
        return 'numeric'
    elif clean_id.isalnum() and any(c.isalpha() for c in clean_id):
        if any(c.isdigit() for c in clean_id):
            return 'mixed'
        else:
            return 'alphabetic'
    else:
        return 'alphanumeric'

def format_client_id_for_display(client_id: str) -> str:
    """
    Formate un identifiant client pour l'affichage dans l'interface utilisateur.
    
    Args:
        client_id: Identifiant brut
        
    Returns:
        Identifiant formaté pour affichage
    """
    if not validate_client_id(client_id):
        return "ID Invalide"
    
    # Pour les IDs longs, afficher les premiers et derniers caractères
    if len(client_id) > 12:
        return f"{client_id[:6]}...{client_id[-6:]}"
    
    return client_id.upper()