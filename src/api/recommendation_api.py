# -*- coding: utf-8 -*-
"""
API pour le Système de Recommandation Personnalisée

Ce module fournit les endpoints API pour accéder aux fonctionnalités
du système de recommandation bancaire.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from ..models.recommendation_manager import RecommendationManager


class RecommendationAPI:
    """API pour les recommandations personnalisées."""
    
    def __init__(self, data_path: str = "data/processed"):
        self.manager = RecommendationManager(data_path)
        self.api_log = []
    
    def _log_api_call(self, endpoint: str, params: dict, response_status: str):
        """Log des appels API."""
        self.api_log.append({
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "params": params,
            "status": response_status
        })
    
    def get_client_recommendations(self, client_id: str) -> Dict[str, Any]:
        """
        Endpoint: GET /api/recommendations/client/{client_id}
        Obtient les recommandations pour un client spécifique EXISTANT.
        """
        try:
            result = self.manager.get_client_recommendations(client_id)
            
            # Vérifier si le résultat est une erreur
            if isinstance(result, dict) and 'error' in result:
                self._log_api_call("get_client_recommendations", {"client_id": client_id}, "ERROR")
                return {
                    "status": "error",
                    "error": result['error'],
                    "timestamp": datetime.now().isoformat()
                }
            
            self._log_api_call("get_client_recommendations", {"client_id": client_id}, "SUCCESS")
            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._log_api_call("get_client_recommendations", {"client_id": client_id}, "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_manual_client_recommendations(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Endpoint: POST /api/recommendations/manual
        Génère des recommandations pour des données client saisies manuellement (NOUVEAUX clients).
        """
        try:
            # Utiliser la méthode du manager pour données manuelles
            result = self.manager.get_manual_client_recommendations(client_data)
            
            # Vérifier si le résultat est une erreur
            if isinstance(result, dict) and 'error' in result:
                self._log_api_call("get_manual_client_recommendations", {"client_id": client_data.get('CLI', 'manual')}, "ERROR")
                return {
                    "status": "error",
                    "error": result['error'],
                    "timestamp": datetime.now().isoformat()
                }
            
            self._log_api_call("get_manual_client_recommendations", {"client_id": client_data.get('CLI', 'manual')}, "SUCCESS")
            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._log_api_call("get_manual_client_recommendations", {"client_id": client_data.get('CLI', 'manual')}, "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_batch_recommendations(self, client_ids: List[str] = None, 
                                 limit: int = 100) -> Dict[str, Any]:
        """
        Endpoint: POST /api/recommendations/batch
        Génère des recommandations pour plusieurs clients.
        """
        try:
            result = self.manager.get_batch_recommendations(client_ids, limit)
            self._log_api_call("get_batch_recommendations", 
                             {"client_ids": client_ids, "limit": limit}, "SUCCESS")
            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._log_api_call("get_batch_recommendations", 
                             {"client_ids": client_ids, "limit": limit}, "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_segment_recommendations(self, segment: str = None, 
                                  market: str = None) -> Dict[str, Any]:
        """
        Endpoint: GET /api/recommendations/segment
        Génère des recommandations par segment ou marché.
        """
        try:
            result = self.manager.get_segment_recommendations(segment, market)
            
            # Vérifier si le résultat est une erreur
            if isinstance(result, dict) and 'error' in result:
                self._log_api_call("get_segment_recommendations", 
                                 {"segment": segment, "market": market}, "ERROR")
                return {
                    "status": "error",
                    "error": result['error'],
                    "timestamp": datetime.now().isoformat()
                }
            
            self._log_api_call("get_segment_recommendations", 
                             {"segment": segment, "market": market}, "SUCCESS")
            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._log_api_call("get_segment_recommendations", 
                             {"segment": segment, "market": market}, "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def record_service_adoption(self, client_id: str, service_id: str, 
                               adoption_date: str = None) -> Dict[str, Any]:
        """
        Endpoint: POST /api/recommendations/adoption
        Enregistre l'adoption d'un service par un client.
        """
        try:
            result = self.manager.record_service_adoption(client_id, service_id, adoption_date)
            self._log_api_call("record_service_adoption", 
                             {"client_id": client_id, "service_id": service_id}, "SUCCESS")
            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._log_api_call("record_service_adoption", 
                             {"client_id": client_id, "service_id": service_id}, "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_adoption_statistics(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Endpoint: GET /api/recommendations/statistics
        Obtient les statistiques d'adoption des services.
        """
        try:
            result = self.manager.get_adoption_statistics(period_days)
            self._log_api_call("get_adoption_statistics", 
                             {"period_days": period_days}, "SUCCESS")
            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._log_api_call("get_adoption_statistics", 
                             {"period_days": period_days}, "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_effectiveness_report(self) -> Dict[str, Any]:
        """
        Endpoint: GET /api/recommendations/effectiveness
        Génère un rapport d'efficacité complet.
        """
        try:
            result = self.manager.get_effectiveness_report()
            self._log_api_call("get_effectiveness_report", {}, "SUCCESS")
            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._log_api_call("get_effectiveness_report", {}, "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_client_profile_analysis(self, client_id: str) -> Dict[str, Any]:
        """
        Endpoint: GET /api/recommendations/profile/{client_id}
        Analyse détaillée du profil d'un client.
        """
        try:
            result = self.manager.get_client_profile_analysis(client_id)
            self._log_api_call("get_client_profile_analysis", 
                             {"client_id": client_id}, "SUCCESS")
            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._log_api_call("get_client_profile_analysis", 
                             {"client_id": client_id}, "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def export_recommendations(self, client_ids: List[str] = None, 
                             format: str = "json") -> Dict[str, Any]:
        """
        Endpoint: POST /api/recommendations/export
        Exporte les recommandations dans différents formats.
        """
        try:
            result = self.manager.export_recommendations(client_ids, format)
            self._log_api_call("export_recommendations", 
                             {"client_ids": client_ids, "format": format}, "SUCCESS")
            return {
                "status": "success",
                "data": {"export_path": result},
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._log_api_call("export_recommendations", 
                             {"client_ids": client_ids, "format": format}, "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Endpoint: GET /api/recommendations/status
        Obtient le statut du système de recommandation.
        """
        try:
            result = self.manager.get_system_status()
            self._log_api_call("get_system_status", {}, "SUCCESS")
            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._log_api_call("get_system_status", {}, "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_available_services(self) -> Dict[str, Any]:
        """
        Endpoint: GET /api/recommendations/services
        Obtient la liste des services disponibles.
        """
        try:
            services = self.manager.recommendation_engine.services_catalog
            self._log_api_call("get_available_services", {}, "SUCCESS")
            return {
                "status": "success",
                "data": {
                    "services": services,
                    "total_services": len(services)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._log_api_call("get_available_services", {}, "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_behavior_segments(self) -> Dict[str, Any]:
        """
        Endpoint: GET /api/recommendations/segments
        Obtient la liste des segments comportementaux.
        """
        try:
            segments = {
                "TRADITIONNEL_RESISTANT": {
                    "description": "Clients très dépendants des chèques, résistants au changement",
                    "characteristics": ["Forte utilisation des chèques", "Faible adoption digitale"],
                    "approach": "Accompagnement progressif"
                },
                "TRADITIONNEL_MODERE": {
                    "description": "Clients modérément dépendants des chèques",
                    "characteristics": ["Usage modéré des chèques", "Adoption digitale limitée"],
                    "approach": "Transition douce"
                },
                "DIGITAL_TRANSITOIRE": {
                    "description": "Clients en transition vers le digital",
                    "characteristics": ["Réduction progressive des chèques", "Adoption digitale croissante"],
                    "approach": "Optimisation digitale"
                },
                "DIGITAL_ADOPTER": {
                    "description": "Clients adoptant activement les solutions digitales",
                    "characteristics": ["Faible usage des chèques", "Forte adoption digitale"],
                    "approach": "Services avancés"
                },
                "DIGITAL_NATIF": {
                    "description": "Clients natifs du digital",
                    "characteristics": ["Usage minimal des chèques", "Maîtrise complète du digital"],
                    "approach": "Solutions innovantes"
                },
                "EQUILIBRE": {
                    "description": "Clients équilibrés entre traditionnel et digital",
                    "characteristics": ["Usage équilibré", "Adoption sélective"],
                    "approach": "Optimisation mixte"
                }
            }
            
            self._log_api_call("get_behavior_segments", {}, "SUCCESS")
            return {
                "status": "success",
                "data": {
                    "segments": segments,
                    "total_segments": len(segments)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self._log_api_call("get_behavior_segments", {}, "ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_api_logs(self, limit: int = 100) -> Dict[str, Any]:
        """
        Endpoint: GET /api/recommendations/logs
        Obtient les logs des appels API.
        """
        try:
            logs = self.api_log[-limit:] if len(self.api_log) > limit else self.api_log
            return {
                "status": "success",
                "data": {
                    "logs": logs,
                    "total_logs": len(self.api_log)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Exemple d'utilisation avec Flask (optionnel)
def create_flask_routes(app, api_instance):
    """
    Crée les routes Flask pour l'API de recommandation.
    Exemple d'intégration avec Flask.
    """
    
    @app.route('/api/recommendations/client/<client_id>', methods=['GET'])
    def client_recommendations(client_id):
        return api_instance.get_client_recommendations(client_id)
    
    @app.route('/api/recommendations/batch', methods=['POST'])
    def batch_recommendations():
        from flask import request
        data = request.get_json() or {}
        client_ids = data.get('client_ids')
        limit = data.get('limit', 100)
        return api_instance.get_batch_recommendations(client_ids, limit)
    
    @app.route('/api/recommendations/segment', methods=['GET'])
    def segment_recommendations():
        from flask import request
        segment = request.args.get('segment')
        market = request.args.get('market')
        return api_instance.get_segment_recommendations(segment, market)
    
    @app.route('/api/recommendations/adoption', methods=['POST'])
    def record_adoption():
        from flask import request
        data = request.get_json() or {}
        client_id = data.get('client_id')
        service_id = data.get('service_id')
        adoption_date = data.get('adoption_date')
        return api_instance.record_service_adoption(client_id, service_id, adoption_date)
    
    @app.route('/api/recommendations/statistics', methods=['GET'])
    def adoption_statistics():
        from flask import request
        period_days = int(request.args.get('period_days', 30))
        return api_instance.get_adoption_statistics(period_days)
    
    @app.route('/api/recommendations/effectiveness', methods=['GET'])
    def effectiveness_report():
        return api_instance.get_effectiveness_report()
    
    @app.route('/api/recommendations/profile/<client_id>', methods=['GET'])
    def client_profile(client_id):
        return api_instance.get_client_profile_analysis(client_id)
    
    @app.route('/api/recommendations/export', methods=['POST'])
    def export_recommendations():
        from flask import request
        data = request.get_json() or {}
        client_ids = data.get('client_ids')
        format = data.get('format', 'json')
        return api_instance.export_recommendations(client_ids, format)
    
    @app.route('/api/recommendations/status', methods=['GET'])
    def system_status():
        return api_instance.get_system_status()
    
    @app.route('/api/recommendations/services', methods=['GET'])
    def available_services():
        return api_instance.get_available_services()
    
    @app.route('/api/recommendations/segments', methods=['GET'])
    def behavior_segments():
        return api_instance.get_behavior_segments()
    
    @app.route('/api/recommendations/logs', methods=['GET'])
    def api_logs():
        from flask import request
        limit = int(request.args.get('limit', 100))
        return api_instance.get_api_logs(limit)
    
    return app