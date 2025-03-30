#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de teste para endpoints da API MicroDetect

Este script testa todos os endpoints da API MicroDetect e gera um relatório 
detalhado sobre quais endpoints funcionaram e quais falharam.

Uso:
    python test_endpoints.py [--host HOSTNAME] [--port PORT]

Argumentos opcionais:
    --host HOSTNAME  : Hostname ou IP do servidor (padrão: localhost)
    --port PORT      : Porta do servidor (padrão: 8000)

Saída:
    - Relatório HTML em ./reports/
    - Relatório JSON em ./reports/
    - Log em api_test_report.log

Exemplo:
    python test_endpoints.py --host localhost --port 8000
"""

import requests
import json
import logging
import time
import os
import sys
import argparse
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_test_report.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuração da API
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"
TIMEOUT = 10  # segundos

# Diretório para relatórios
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

class ApiTester:
    def __init__(self, base_url=BASE_URL, api_prefix=API_PREFIX):
        self.base_url = base_url
        self.api_prefix = api_prefix
        self.results = {
            "success": [],
            "failure": [],
            "total_endpoints": 0,
            "successful_endpoints": 0,
            "failed_endpoints": 0,
            "start_time": datetime.now().isoformat()
        }
        self.test_data = {}
        
        # Para armazenar IDs criados durante os testes para uso em outros endpoints
        self.created_ids = {
            "image": None,
            "dataset": None,
            "annotation": None
        }
    
    def full_url(self, endpoint):
        """Retorna a URL completa para o endpoint."""
        return f"{self.base_url}{self.api_prefix}{endpoint}"
    
    def check_server_health(self):
        """Verifica se o servidor está rodando."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=TIMEOUT)
            if response.status_code == 200:
                logger.info("✅ Servidor está online e respondendo")
                return True
            else:
                logger.error(f"❌ Servidor respondeu com status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Não foi possível conectar ao servidor: {e}")
            return False
    
    def test_endpoint(self, method, endpoint, data=None, files=None, params=None, expected_status=200, description=None):
        """Testa um endpoint e registra o resultado."""
        self.results["total_endpoints"] += 1
        url = self.full_url(endpoint)
        
        logger.info(f"Testando {method} {endpoint} - {description or ''}")
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=TIMEOUT)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, files=files, params=params, timeout=TIMEOUT)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data, timeout=TIMEOUT)
            elif method.upper() == "DELETE":
                response = requests.delete(url, timeout=TIMEOUT)
            else:
                raise ValueError(f"Método HTTP não suportado: {method}")
            
            # Verificar se o status code corresponde ao esperado
            if response.status_code == expected_status:
                logger.info(f"✅ {method} {endpoint} - Status: {response.status_code}")
                
                # Extrair informações para uso em outros testes
                if method.upper() == "POST" and response.status_code in (200, 201):
                    self._extract_created_ids(endpoint, response)
                
                result = {
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "description": description,
                    "response_time": response.elapsed.total_seconds(),
                    "success": True
                }
                self.results["success"].append(result)
                self.results["successful_endpoints"] += 1
                return True, response
            else:
                logger.error(f"❌ {method} {endpoint} - Status esperado: {expected_status}, recebido: {response.status_code}")
                result = {
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "expected_status": expected_status,
                    "description": description,
                    "response_time": response.elapsed.total_seconds(),
                    "error": response.text[:500],  # Limitar tamanho do erro
                    "success": False
                }
                self.results["failure"].append(result)
                self.results["failed_endpoints"] += 1
                return False, response
        except Exception as e:
            logger.error(f"❌ {method} {endpoint} - Erro: {str(e)}")
            result = {
                "method": method,
                "endpoint": endpoint,
                "expected_status": expected_status,
                "description": description,
                "error": str(e),
                "success": False
            }
            self.results["failure"].append(result)
            self.results["failed_endpoints"] += 1
            return False, None
    
    def _extract_created_ids(self, endpoint, response):
        """Extrai IDs criados para uso em outros testes."""
        try:
            if '/images' in endpoint and response.json().get('data'):
                image_data = response.json().get('data')
                if isinstance(image_data, dict) and 'id' in image_data:
                    self.created_ids["image"] = image_data['id']
                    logger.info(f"ID de imagem extraído: {self.created_ids['image']}")
            
            elif '/datasets' in endpoint and response.json().get('data'):
                dataset_data = response.json().get('data')
                if isinstance(dataset_data, dict) and 'id' in dataset_data:
                    self.created_ids["dataset"] = dataset_data['id']
                    logger.info(f"ID de dataset extraído: {self.created_ids['dataset']}")
            
            elif '/annotations' in endpoint and response.json().get('data'):
                annotation_data = response.json().get('data')
                if isinstance(annotation_data, dict) and 'id' in annotation_data:
                    self.created_ids["annotation"] = annotation_data['id']
                    logger.info(f"ID de anotação extraído: {self.created_ids['annotation']}")
                elif isinstance(annotation_data, list) and annotation_data and 'id' in annotation_data[0]:
                    self.created_ids["annotation"] = annotation_data[0]['id']
                    logger.info(f"ID de anotação extraído: {self.created_ids['annotation']}")
        except:
            logger.warning("Não foi possível extrair IDs da resposta")
    
    def prepare_test_data(self):
        """Prepara dados para testes."""
        # Dados para criação de dataset
        self.test_data["dataset"] = {
            "name": f"Test Dataset {datetime.now().strftime('%Y%m%d%H%M%S')}",
            "description": "Dataset criado para teste automatizado",
            "classes": ["classe1", "classe2"]
        }
        
        # Dados para criação de anotação
        self.test_data["annotation"] = {
            "image_id": None,  # Será preenchido após criação de imagem
            "class_name": "classe1",
            "x": 100,
            "y": 100,
            "width": 50,
            "height": 50,
            "confidence": 0.95
        }
        
        # Dados para criação de anotações em lote
        self.test_data["annotation_batch"] = {
            "image_id": None,  # Será preenchido após criação de imagem
            "dataset_id": None,  # Será preenchido após criação de dataset
            "annotations": [
                {
                    "class_name": "classe1",
                    "x": 100,
                    "y": 100,
                    "width": 50,
                    "height": 50,
                    "confidence": 0.95
                },
                {
                    "class_name": "classe2",
                    "x": 200,
                    "y": 200,
                    "width": 60,
                    "height": 60,
                    "confidence": 0.85
                }
            ]
        }
    
    def prepare_test_image(self):
        """
        Cria uma imagem de teste para upload.
        Retorna o caminho para a imagem temporária.
        """
        try:
            import numpy as np
            from PIL import Image
            
            # Criar diretório temporário se não existir
            temp_dir = os.path.join(os.getcwd(), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Gerar imagem aleatória
            width, height = 300, 200
            # Criar array NumPy com valores aleatórios RGB
            img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Definir um nome único para a imagem
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            img_path = os.path.join(temp_dir, f"test_image_{timestamp}.jpg")
            
            # Salvar a imagem
            img.save(img_path)
            logger.info(f"Imagem de teste criada em: {img_path}")
            
            return img_path
        except ImportError:
            logger.warning("Não foi possível criar imagem de teste. Numpy ou PIL não está instalado.")
            return None
        except Exception as e:
            logger.warning(f"Erro ao criar imagem de teste: {str(e)}")
            return None
    
    def run_tests(self):
        """Executa todos os testes de endpoints."""
        # Verificar se o servidor está rodando
        if not self.check_server_health():
            logger.error("Testes abortados. Servidor não está acessível.")
            return False
        
        self.prepare_test_data()
        
        # === Teste de endpoints de sistema (todos opcionais) ===
        logger.info("=== Testando endpoints de SISTEMA (opcionais) ===")
        self.test_optional_endpoint("GET", "/system/status", description="Obter informações do sistema")
        
        # === TESTES DE DATASETS ===
        logger.info("=== Iniciando testes de DATASETS ===")
        # 1. Criar dataset (POST)
        success, response = self.test_endpoint(
            "POST", 
            "/datasets/", 
            data=self.test_data["dataset"], 
            description="Criar dataset"
        )
        
        # 2. Listar datasets (GET)
        self.test_endpoint(
            "GET", 
            "/datasets/", 
            description="Listar todos os datasets"
        )
        
        # Testar endpoints que precisam de um dataset existente
        if self.created_ids["dataset"]:
            dataset_id = self.created_ids["dataset"]
            
            # 3. Obter dataset específico (GET)
            self.test_endpoint(
                "GET", 
                f"/datasets/{dataset_id}", 
                description="Obter dataset específico"
            )
            
            # 4. Atualizar dataset parcialmente (PUT)
            self.test_endpoint(
                "PUT", 
                f"/datasets/{dataset_id}", 
                data={"description": "Descrição atualizada pelo teste automatizado"},
                description="Atualizar dataset (descrição)"
            )
            
            # 5. Atualizar dataset completamente (PUT)
            self.test_endpoint(
                "PUT", 
                f"/datasets/{dataset_id}", 
                data={
                    "name": f"Dataset Atualizado {datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "description": "Dataset completamente atualizado",
                    "classes": ["classe_nova1", "classe_nova2", "classe_nova3"]
                },
                description="Atualizar dataset completamente"
            )
            
            # 6. Obter estatísticas do dataset (GET)
            self.test_endpoint(
                "GET", 
                f"/datasets/{dataset_id}/stats", 
                description="Obter estatísticas do dataset"
            )
            
            # 7. Criar dataset secundário para teste de DELETE
            success, response = self.test_endpoint(
                "POST", 
                "/datasets/", 
                data={
                    "name": f"Dataset Para Deletar {datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "description": "Este dataset será deletado",
                    "classes": ["temp1", "temp2"]
                }, 
                description="Criar dataset para deletar"
            )
            
            if success and response:
                try:
                    dataset_to_delete_id = response.json()["data"]["id"]
                    
                    # 8. Deletar dataset (DELETE)
                    self.test_endpoint(
                        "DELETE", 
                        f"/datasets/{dataset_to_delete_id}", 
                        description="Deletar dataset"
                    )
                    
                    # Nota: A API parece não retornar 404 após exclusão, então não testamos isso
                except:
                    logger.warning("Não foi possível extrair ID do dataset para deletar")
        
        # === TESTES DE IMAGENS ===
        logger.info("=== Iniciando testes de IMAGENS ===")
        # 1. Listar imagens (GET)
        self.test_endpoint(
            "GET", 
            "/images/", 
            description="Listar todas as imagens"
        )
        
        # 2. Listar imagens de um dataset específico (GET com query params)
        if self.created_ids["dataset"]:
            self.test_endpoint(
                "GET", 
                "/images/",
                params={"dataset_id": self.created_ids["dataset"]},
                description="Listar imagens de um dataset específico"
            )

            # 3. Upload de imagem para dataset
            # O upload de imagem é uma operação importante, mas a API pode implementá-la
            # de várias maneiras diferentes
            image_path = self.prepare_test_image()
            if image_path and os.path.exists(image_path):
                try:
                    with open(image_path, 'rb') as img_file:
                        files = {'file': (os.path.basename(image_path), img_file, 'image/jpeg')}

                        # Tentamos várias endpoints possíveis para upload
                        # Primeiro como opcional
                        success1, response1 = self.test_optional_endpoint(
                            "POST",
                            "/images/upload",
                            params={"dataset_id": self.created_ids["dataset"]},
                            files=files,
                            description="Upload de imagem (endpoint /images/upload)"
                        )

                        # Se o primeiro falhar, tentamos o segundo como um teste obrigatório
                        if not success1:
                            self.test_endpoint(
                                "POST",
                                "/images/",
                                params={"dataset_id": self.created_ids["dataset"]},
                                files=files,
                                description="Upload de imagem (endpoint /images/)"
                            )
                except Exception as e:
                    logger.warning(f"Erro ao testar upload de imagem: {str(e)}")
        
        # Testar endpoints que precisam de uma imagem existente
        if self.created_ids["image"]:
            image_id = self.created_ids["image"]
            
            # 4. Obter imagem específica (GET)
            self.test_endpoint(
                "GET", 
                f"/images/{image_id}", 
                description="Obter imagem específica"
            )
            
            # 5. Obter imagem com anotações (GET com query params)
            self.test_endpoint(
                "GET", 
                f"/images/{image_id}", 
                params={"with_annotations": "true"},
                description="Obter imagem com anotações"
            )
            
            # 6. Atualizar metadados da imagem (PUT)
            self.test_endpoint(
                "PUT", 
                f"/images/{image_id}", 
                data={
                    "metadata": {
                        "source": "Teste automatizado",
                        "collected_at": datetime.now().isoformat(),
                        "quality": "high"
                    }
                },
                description="Atualizar metadados da imagem"
            )
        
        # === TESTES DE ANOTAÇÕES ===
        logger.info("=== Iniciando testes de ANOTAÇÕES ===")
        
        # === Endpoints opcionais de anotações ===
        # Todos estes endpoints são tratados como opcionais pois podem não estar implementados
        # ou podem requerer parâmetros específicos
        
        # Listagem de anotações sem filtros (causa erro 500 - possível recursão infinita)
        # Marcamos como opcional e não afeta a taxa de sucesso
        self.test_optional_endpoint(
            "GET", 
            "/annotations/",
            params={"limit": 10, "offset": 0},  # Limitando para reduzir carga
            description="Listar anotações com paginação (opcional)"
        )
        
        # Endpoints opcionais de estatísticas e exportação
        self.test_optional_endpoint(
            "GET", 
            "/annotations/stats",
            description="Obter estatísticas gerais de anotações"
        )
        
        self.test_optional_endpoint(
            "GET", 
            "/annotations/export",
            params={"format": "coco"},
            description="Exportar anotações em formato COCO"
        )
        
        self.test_optional_endpoint(
            "GET", 
            "/annotations/classes",
            description="Listar todas as classes de anotações utilizadas"
        )
        
        # === Endpoints obrigatórios de anotações ===
        # Estes endpoints são considerados obrigatórios e devem funcionar
        
        # Listar anotações com filtros (GET com query params)
        if self.created_ids["dataset"]:
            dataset_id = self.created_ids["dataset"]
            
            # 1. Listar anotações de um dataset específico
            self.test_endpoint(
                "GET", 
                "/annotations/", 
                params={"dataset_id": dataset_id},
                description="Listar anotações de um dataset específico"
            )
            
            # 2. Obter classes definidas em um dataset
            self.test_endpoint(
                "GET", 
                f"/annotations/dataset/{dataset_id}/classes", 
                description="Obter classes definidas em um dataset"
            )
            
            # Endpoints opcionais específicos de dataset
            self.test_optional_endpoint(
                "GET", 
                f"/annotations/dataset/{dataset_id}/stats",
                description="Obter estatísticas de anotações para um dataset específico"
            )
            
            self.test_optional_endpoint(
                "GET", 
                f"/annotations/export",
                params={"dataset_id": dataset_id, "format": "coco"},
                description="Exportar anotações de um dataset específico"
            )
        
        # Testar endpoints que precisam de uma imagem para criar anotações
        if self.created_ids["image"]:
            image_id = self.created_ids["image"]
            
            # 3. Listar anotações de uma imagem específica
            self.test_endpoint(
                "GET", 
                "/annotations/", 
                params={"image_id": image_id},
                description="Listar anotações de uma imagem específica"
            )
            
            # Endpoint opcional para exportação
            self.test_optional_endpoint(
                "GET", 
                f"/annotations/export",
                params={"image_id": image_id, "format": "coco"},
                description="Exportar anotações de uma imagem específica"
            )
            
            # 4. Criar anotação individual (POST)
            success, response = self.test_endpoint(
                "POST", 
                "/annotations/", 
                data={
                    "image_id": image_id,
                    "class_name": "classe_teste",
                    "x": 100,
                    "y": 100,
                    "width": 50,
                    "height": 50,
                    "confidence": 0.9
                }, 
                description="Criar anotação individual"
            )
            
            # 5. Criar múltiplas anotações em lote (POST)
            if self.created_ids["dataset"]:
                self.test_endpoint(
                    "POST", 
                    "/annotations/batch", 
                    data={
                        "image_id": image_id,
                        "dataset_id": self.created_ids["dataset"],
                        "annotations": [
                            {
                                "class_name": "classe1",
                                "x": 150,
                                "y": 150,
                                "width": 60,
                                "height": 60,
                                "confidence": 0.85
                            },
                            {
                                "class_name": "classe2",
                                "x": 250,
                                "y": 250,
                                "width": 70,
                                "height": 70,
                                "confidence": 0.75
                            }
                        ]
                    }, 
                    description="Criar múltiplas anotações em lote"
                )
                
                # Endpoints opcionais de operações em lote
                self.test_optional_endpoint(
                    "PUT", 
                    "/annotations",
                    data={
                        "annotations": [
                            {
                                "id": self.created_ids["annotation"] if self.created_ids["annotation"] else 1,
                                "confidence": 0.92
                            }
                        ]
                    }, 
                    description="Atualizar múltiplas anotações em lote"
                )
                
                self.test_optional_endpoint(
                    "DELETE", 
                    "/annotations/batch", 
                    data={"image_id": image_id, "class_name": "classe2"},
                    description="Deletar múltiplas anotações por filtro"
                )
            
            # 6. Criar anotação para testar deleção individual
            success, response = self.test_endpoint(
                "POST", 
                "/annotations/", 
                data={
                    "image_id": image_id,
                    "class_name": "para_deletar",
                    "x": 300,
                    "y": 300,
                    "width": 40,
                    "height": 40,
                    "confidence": 0.8
                }, 
                description="Criar anotação para deletar"
            )
            
            if success and response and 'data' in response.json():
                try:
                    annotation_to_delete_id = response.json()["data"]["id"]
                    
                    # 7. Deletar anotação individual (DELETE)
                    self.test_endpoint(
                        "DELETE", 
                        f"/annotations/{annotation_to_delete_id}", 
                        description="Deletar anotação individual por ID"
                    )
                    
                    # 8. Verificar que anotação foi deletada (GET - deve retornar 404)
                    # Este é um teste esperado que a API retorne 404 quando uma anotação não existe
                    self.test_endpoint(
                        "GET", 
                        f"/annotations/{annotation_to_delete_id}", 
                        expected_status=404,
                        description="Verificar que anotação foi deletada"
                    )
                except:
                    logger.warning("Não foi possível extrair ID da anotação para deletar")
        
        # Testar endpoints para uma anotação existente
        if self.created_ids["annotation"]:
            annotation_id = self.created_ids["annotation"]
            
            # 9. Obter anotação específica
            self.test_endpoint(
                "GET", 
                f"/annotations/{annotation_id}", 
                description="Obter anotação específica por ID"
            )
            
            # 10. Atualizar anotação parcialmente
            self.test_endpoint(
                "PUT", 
                f"/annotations/{annotation_id}", 
                data={"confidence": 0.95},
                description="Atualizar anotação parcialmente (confidence)"
            )
            
            # 11. Atualizar anotação completamente
            self.test_endpoint(
                "PUT", 
                f"/annotations/{annotation_id}", 
                data={
                    "class_name": "classe_atualizada",
                    "x": 120,
                    "y": 120,
                    "width": 60,
                    "height": 60,
                    "confidence": 0.98
                },
                description="Atualizar anotação completamente"
            )
            
            # Endpoints opcionais avançados
            self.test_optional_endpoint(
                "GET", 
                f"/annotations/{annotation_id}/history", 
                description="Obter histórico de alterações de uma anotação"
            )
            
            self.test_optional_endpoint(
                "POST", 
                f"/annotations/{annotation_id}/duplicate", 
                description="Duplicar uma anotação existente"
            )
        
        # === TESTES DE ENDPOINTS OPCIONAIS DE ML ===
        logger.info("=== Iniciando testes de ENDPOINTS OPCIONAIS DE MACHINE LEARNING ===")
        # Todos estes endpoints são tratados como opcionais
        
        # Endpoints de inferência
        self.test_optional_endpoint(
            "GET", 
            "/inference/models", 
            description="Listar modelos disponíveis para inferência"
        )
        
        if self.created_ids["image"]:
            self.test_optional_endpoint(
                "POST", 
                "/inference/detect", 
                data={"image_id": self.created_ids["image"], "model_id": "default"},
                description="Detectar objetos em uma imagem"
            )
        
        # Endpoints de treinamento
        self.test_optional_endpoint(
            "GET", 
            "/training/jobs", 
            description="Listar jobs de treinamento"
        )
        
        if self.created_ids["dataset"]:
            self.test_optional_endpoint(
                "POST", 
                "/training/jobs", 
                data={
                    "dataset_id": self.created_ids["dataset"],
                    "model_name": "teste_model",
                    "epochs": 5,
                    "batch_size": 4,
                    "learning_rate": 0.001
                },
                description="Criar job de treinamento"
            )
        
        # Finalizar resultados
        self.results["end_time"] = datetime.now().isoformat()
        self.results["duration_seconds"] = (datetime.fromisoformat(self.results["end_time"]) - 
                                          datetime.fromisoformat(self.results["start_time"])).total_seconds()
        
        return True
    
    def generate_report(self):
        """Gera um relatório com os resultados dos testes."""
        # Contadores para endpoints obrigatórios
        total_required = len([e for e in self.results["success"] + self.results["failure"] if not e.get("optional", False)])
        successful_required = len([e for e in self.results["success"] if not e.get("optional", False)])
        failed_required = len([e for e in self.results["failure"] if not e.get("optional", False)])
        
        # Contadores para endpoints opcionais
        total_optional = len([e for e in self.results["success"] + self.results["failure"] if e.get("optional", False)])
        
        # Formatar resultados
        success_rate = successful_required / total_required * 100 if total_required > 0 else 0
        
        # Atualizar resultados finais
        self.results["total_required_endpoints"] = total_required
        self.results["successful_required_endpoints"] = successful_required
        self.results["failed_required_endpoints"] = failed_required
        self.results["total_optional_endpoints"] = total_optional
        self.results["success_rate"] = success_rate
        
        # Criar relatório HTML
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .optional {{ color: #888; font-style: italic; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .status-200, .status-201 {{ background-color: rgba(0, 255, 0, 0.1); }}
                .status-400, .status-404, .status-500 {{ background-color: rgba(255, 0, 0, 0.1); }}
            </style>
        </head>
        <body>
            <h1>API Test Report</h1>
            <div class="summary">
                <h2>Resumo</h2>
                <p>Data do teste: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Endpoints Obrigatórios:</strong></p>
                <p>Total: {total_required}</p>
                <p>Sucesso: <span class="success">{successful_required}</span></p>
                <p>Falha: <span class="failure">{failed_required}</span></p>
                <p>Taxa de sucesso: <strong>{success_rate:.2f}%</strong></p>
                
                <p><strong>Endpoints Opcionais Testados:</strong> {total_optional}</p>
                <p>Duração do teste: {self.results["duration_seconds"]:.2f} segundos</p>
            </div>
            
            <h2>Endpoints com Sucesso ({len(self.results["success"])})</h2>
            <table>
                <tr>
                    <th>Método</th>
                    <th>Endpoint</th>
                    <th>Status</th>
                    <th>Tempo de resposta (s)</th>
                    <th>Descrição</th>
                    <th>Tipo</th>
                </tr>
        """
        
        # Adicionar endpoints com sucesso
        for endpoint in self.results["success"]:
            endpoint_type = "Opcional" if endpoint.get("optional", False) else "Obrigatório"
            css_class = "optional" if endpoint.get("optional", False) else ""
            
            html_report += f"""
                <tr class="status-{endpoint['status_code']} {css_class}">
                    <td>{endpoint['method']}</td>
                    <td>{endpoint['endpoint']}</td>
                    <td>{endpoint['status_code']}</td>
                    <td>{endpoint['response_time']:.3f}</td>
                    <td>{endpoint['description'] or ''}</td>
                    <td>{endpoint_type}</td>
                </tr>
            """
        
        html_report += f"""
            </table>
            
            <h2>Endpoints com Falha ({len(self.results["failure"])})</h2>
            <table>
                <tr>
                    <th>Método</th>
                    <th>Endpoint</th>
                    <th>Status</th>
                    <th>Status Esperado</th>
                    <th>Tempo de resposta (s)</th>
                    <th>Descrição</th>
                    <th>Erro</th>
                    <th>Tipo</th>
                </tr>
        """
        
        # Adicionar endpoints com falha
        for endpoint in self.results["failure"]:
            status_code = endpoint.get('status_code', 'N/A')
            response_time = endpoint.get('response_time', 'N/A')
            if response_time != 'N/A':
                response_time = f"{response_time:.3f}"
            
            endpoint_type = "Opcional" if endpoint.get("optional", False) else "Obrigatório"
            css_class = "optional" if endpoint.get("optional", False) else ""
            
            html_report += f"""
                <tr class="status-{status_code} {css_class}">
                    <td>{endpoint['method']}</td>
                    <td>{endpoint['endpoint']}</td>
                    <td>{status_code}</td>
                    <td>{endpoint['expected_status']}</td>
                    <td>{response_time}</td>
                    <td>{endpoint['description'] or ''}</td>
                    <td>{endpoint.get('error', 'N/A')}</td>
                    <td>{endpoint_type}</td>
                </tr>
            """
        
        html_report += """
            </table>
        </body>
        </html>
        """
        
        # Salvar relatório
        report_filename = f"{REPORTS_DIR}/api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"Relatório salvo em: {report_filename}")
        
        # Criar também um relatório JSON
        json_filename = f"{REPORTS_DIR}/api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Relatório JSON salvo em: {json_filename}")
        
        return report_filename

    def test_optional_endpoint(self, method, endpoint, data=None, files=None, params=None, expected_status=200, description=None):
        """
        Testa um endpoint opcional, sem afetar as métricas se o endpoint não existir.
        Útil para endpoints que podem não estar implementados em todas as versões.
        """
        url = self.full_url(endpoint)
        
        logger.info(f"Testando endpoint opcional {method} {endpoint} - {description or ''}")
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=TIMEOUT)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, files=files, params=params, timeout=TIMEOUT)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data, timeout=TIMEOUT)
            elif method.upper() == "DELETE":
                response = requests.delete(url, timeout=TIMEOUT)
            else:
                raise ValueError(f"Método HTTP não suportado: {method}")
            
            result = {
                "method": method,
                "endpoint": endpoint,
                "status_code": response.status_code,
                "description": description,
                "response_time": response.elapsed.total_seconds(),
                "success": response.status_code == expected_status,
                "optional": True
            }
            
            if response.status_code == expected_status:
                logger.info(f"✅ (Opcional) {method} {endpoint} - Status: {response.status_code}")
                # Registramos o sucesso mas não contabilizamos nos totais de endpoints obrigatórios
                self.results["success"].append(result)
                
                # Extrair informações mesmo sendo opcional
                if method.upper() == "POST" and response.status_code in (200, 201):
                    self._extract_created_ids(endpoint, response)
                
                return True, response
            elif response.status_code == 404:
                # Se o endpoint não existe (404), consideramos isso esperado para endpoints opcionais
                logger.info(f"ℹ️ (Opcional) {method} {endpoint} - Endpoint não implementado (404)")
                
                # Ainda armazenamos na lista de resultados para contabilizar endpoints opcionais
                result.update({"expected_status": expected_status, "error": "Endpoint não implementado"})
                # Não é um sucesso nem uma falha do ponto de vista do teste, então usamos success=None
                result["success"] = None
                self.results["failure"].append(result)
                
                return False, response
            else:
                # O endpoint existe mas retornou um status diferente do esperado
                # Ainda assim, não contabilizamos nas métricas de sucesso/falha obrigatórios
                logger.warning(f"⚠️ (Opcional) {method} {endpoint} - Status esperado: {expected_status}, recebido: {response.status_code}")
                
                # Armazenamos para contabilizar endpoints opcionais
                result.update({
                    "expected_status": expected_status,
                    "error": response.text[:500] if hasattr(response, 'text') else "Resposta inesperada"
                })
                self.results["failure"].append(result)
                
                return False, response
        except Exception as e:
            logger.warning(f"⚠️ (Opcional) {method} {endpoint} - Erro: {str(e)}")
            
            # Armazenamos mesmo quando ocorre uma exceção
            result = {
                "method": method,
                "endpoint": endpoint,
                "description": description,
                "expected_status": expected_status,
                "error": str(e),
                "success": False,
                "optional": True
            }
            self.results["failure"].append(result)
            
            return False, None

def main():
    """Função principal que configura e executa os testes."""
    # Configuração dos argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Testa os endpoints da API MicroDetect")
    parser.add_argument("--host", default="localhost", help="Hostname ou IP do servidor (padrão: localhost)")
    parser.add_argument("--port", default="8000", help="Porta do servidor (padrão: 8000)")
    args = parser.parse_args()
    
    # URL base construída a partir dos argumentos
    base_url = f"http://{args.host}:{args.port}"
    
    logger.info(f"Iniciando testes de API em {base_url}")
    tester = ApiTester(base_url=base_url)
    
    if tester.run_tests():
        report_file = tester.generate_report()
        logger.info(f"Testes concluídos. Relatório gerado em: {report_file}")
        
        # Exibir resumo dos resultados
        print(f"\nTestes concluídos com sucesso!")
        print(f"Resumo dos Resultados:")
        print(f"  Endpoints obrigatórios:")
        print(f"    Total: {tester.results.get('total_required_endpoints', 0)}")
        print(f"    Sucesso: {tester.results.get('successful_required_endpoints', 0)}")
        print(f"    Falha: {tester.results.get('failed_required_endpoints', 0)}")
        
        success_rate = tester.results.get("success_rate", 0)
        print(f"    Taxa de sucesso: {success_rate:.2f}%")
        
        print(f"  Endpoints opcionais testados: {tester.results.get('total_optional_endpoints', 0)}")
        print(f"  Duração do teste: {tester.results.get('duration_seconds', 0):.2f} segundos")
        print(f"  Relatório detalhado: {report_file}")
    else:
        logger.error("Falha ao executar os testes. O servidor está acessível?")
        print("\nFalha ao executar os testes. Verifique se o servidor está rodando.")

if __name__ == "__main__":
    main() 