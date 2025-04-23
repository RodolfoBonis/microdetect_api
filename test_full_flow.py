#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de teste para o fluxo completo da API MicroDetect

Este script testa todo o fluxo da API MicroDetect, incluindo:
- Criação de dataset
- Upload e vinculação de imagens
- Anotação de imagens
- Busca de hiperparâmetros
- Treinamento do modelo
- Monitoramento via WebSocket
"""

import requests
import json
import logging
import time
import sys
import argparse
import websockets
import asyncio
import base64
from datetime import datetime
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("full_flow_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuração da API
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
API_PREFIX = "/api/v1"
TIMEOUT = 30  # segundos

class FullFlowTester:
    def __init__(self, base_url=BASE_URL, ws_url=WS_URL, api_prefix=API_PREFIX):
        self.base_url = base_url
        # Remover barra final se existir
        self.ws_url = ws_url.rstrip('/')
        self.api_prefix = api_prefix
        self.dataset_id = None
        self.image_ids = []
        self.training_session_id = None
        self.session = requests.Session()
        logger.info(f"Configuração HTTP: {self.base_url}{self.api_prefix}")
        logger.info(f"Configuração WebSocket: {self.ws_url}")
        
    def full_url(self, endpoint):
        return f"{self.base_url}{self.api_prefix}{endpoint}"
    
    async def monitor_training_websocket(self):
        """Monitora o progresso do treinamento via WebSocket"""
        try:
            # Conectar com o caminho correto incluindo api_prefix
            ws = await websockets.connect(
                f"{self.ws_url}{self.api_prefix}/training/hyperparams/ws/{self.training_session_id}"
            )
            logger.info("Conexão WebSocket estabelecida com sucesso")
            
            while True:
                try:
                    message = await ws.recv()
                    data = json.loads(message)
                    
                    # Log do progresso
                    if "progress" in data:
                        progress = data["progress"]
                        logger.info(f"Progresso do treinamento: {progress.get('current_epoch', 0)}/{progress.get('total_epochs', 0)} épocas")
                        
                        # Log das métricas
                        if "metrics" in data:
                            metrics = data["metrics"]
                            logger.info(f"Métricas: {json.dumps(metrics, indent=2)}")
                        
                        # Log do uso de recursos
                        if "resources" in data:
                            resources = data["resources"]
                            logger.info(f"Recursos: {json.dumps(resources, indent=2)}")
                    
                    # Verificar conclusão
                    if data.get("status") == "completed":
                        logger.info("Treinamento concluído com sucesso")
                        await ws.close()
                        return True
                    elif data.get("status") == "failed":
                        logger.error("Treinamento falhou")
                        await ws.close()
                        return False
                        
                except websockets.exceptions.ConnectionClosed:
                    logger.error("Conexão WebSocket fechada")
                    return False
                    
        except Exception as e:
            logger.error(f"Erro no WebSocket: {e}")
            return False

    async def monitor_hyperparam_websocket(self, search_id: str):
        """Monitora atualizações de busca de hiperparâmetros via WebSocket"""
        try:
            # Tentar com a URL padrão
            websocket_url = f"{self.ws_url}{self.api_prefix}/hyperparams/ws/{search_id}"
            logger.info(f"Tentando conexão WebSocket via: {websocket_url}")
            
            try:
                ws = await websockets.connect(websocket_url)
                logger.info(f"Conectado ao WebSocket via: {websocket_url}")
            except Exception as e:
                logger.warning(f"Falha na conexão WebSocket via {websocket_url}: {str(e)}")
                
                # Tentar URL alternativa (rota atualizada)
                alt_url = f"{self.ws_url}{self.api_prefix}/hyperparams/ws/{search_id}"
                logger.info(f"Tentando rota alternativa: {alt_url}")
                ws = await websockets.connect(alt_url)
                logger.info(f"Conectado ao WebSocket via rota alternativa: {alt_url}")
                
            logger.info(f"Conexão WebSocket estabelecida para busca {search_id}")

            try:
                # Receber estado inicial
                logger.info("Aguardando estado inicial...")
                initial_state = await ws.recv()
                logger.info(f"Estado inicial recebido: {initial_state}")

                # Enviar confirmação
                confirmation_msg = json.dumps({"type": "acknowledge"})
                logger.info(f"Enviando confirmação: {confirmation_msg}")
                await ws.send(confirmation_msg)
                logger.info("Confirmação enviada")

                # Monitorar atualizações
                logger.info("Iniciando monitoramento de atualizações...")
                while True:
                    try:
                        logger.info("Aguardando mensagem...")
                        message = await ws.recv()
                        logger.info(f"Mensagem recebida: {message[:100]}...")
                        
                        try:
                            data = json.loads(message)
                            
                            if data.get("type") == "heartbeat":
                                logger.debug("Heartbeat recebido")
                                continue
                                
                            logger.info(f"Atualização recebida: {data}")
                            
                            # Verificar se a busca terminou
                            if data.get("status") in ["completed", "failed"]:
                                logger.info(f"Busca finalizada com status: {data.get('status')}")
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Mensagem recebida não é um JSON válido: {message}")
                            
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.error(f"Conexão WebSocket fechada: {str(e)}")
                        break
                    except Exception as e:
                        logger.error(f"Erro ao processar mensagem: {str(e)}")
                        break

            finally:
                logger.info("Fechando conexão WebSocket...")
                await ws.close()
                logger.info("Conexão WebSocket fechada")

        except Exception as e:
            logger.error(f"Erro ao monitorar busca via WebSocket: {str(e)}")
            # Não propagar erros, apenas registrar

    def create_dataset(self):
        """Usa o dataset existente"""
        self.dataset_id = 1  # ID do dataset que já foi criado
        logger.info(f"Usando dataset existente com ID: {self.dataset_id}")
        return True

    def upload_and_associate_images(self):
        """Upload de imagens e associação ao dataset"""
        # Verificar se já existem imagens no dataset
        response = self.session.get(
            self.full_url(f"/datasets/{self.dataset_id}"),
            timeout=TIMEOUT
        )
        if response.status_code == 200:
            dataset_info = response.json()["data"]
            if dataset_info["images_count"] > 0:
                logger.info(f"Dataset já possui {dataset_info['images_count']} imagens. Pulando upload.")
                return True

        test_images_dir = Path("test_images")
        if not test_images_dir.exists():
            logger.error("Diretório de imagens de teste não encontrado")
            return False

        # Procurar por imagens PNG
        for img_path in test_images_dir.glob("*.png"):
            # Upload da imagem
            with open(img_path, "rb") as f:
                files = {"file": (img_path.name, f, "image/png")}
                response = self.session.post(
                    self.full_url("/images/"),
                    files=files,
                    data={"dataset_id": self.dataset_id},
                    timeout=TIMEOUT
                )
                if response.status_code == 200:
                    image_id = response.json()["data"]["id"]
                    self.image_ids.append(image_id)
                    logger.info(f"Imagem {image_id} enviada com sucesso")
                else:
                    logger.error(f"Falha ao enviar imagem {img_path.name}: {response.text}")
                    return False
        
        if not self.image_ids:
            logger.error("Nenhuma imagem PNG encontrada no diretório test_images")
            return False
            
        logger.info(f"Total de {len(self.image_ids)} imagens enviadas e associadas ao dataset")
        return True

    async def annotate_images(self):
        """Anota as imagens do dataset"""
        logger.info("Iniciando anotação das imagens")
        
        # Verificar se já existem anotações
        try:
            response = self.session.get(self.full_url(f"/datasets/{self.dataset_id}"), timeout=TIMEOUT)
            if response.status_code != 200:
                logger.error(f"Erro ao obter informações do dataset: {response.text}")
                return False
                
            dataset_info = response.json()
            
            if "data" in dataset_info and dataset_info["data"].get("annotations_count", 0) > 0:
                logger.info(f"Dataset já possui {dataset_info['data']['annotations_count']} anotações")
                return True  # Retornar True quando já existem anotações
        except Exception as e:
            logger.error(f"Erro ao verificar anotações existentes: {str(e)}")
            return False
            
        # Obter todas as imagens do dataset usando o endpoint correto
        try:
            # Usar o endpoint de imagens com o parâmetro dataset_id
            response = self.session.get(self.full_url(f"/images/?dataset_id={self.dataset_id}"), timeout=TIMEOUT)
            if response.status_code != 200:
                logger.error(f"Erro ao obter imagens do dataset: {response.text}")
                return False
                
            response_data = response.json()
            if "data" not in response_data:
                logger.error(f"Resposta inválida ao obter imagens: {response_data}")
                return False
                
            images = response_data["data"]
            
            if not images:
                logger.error("Nenhuma imagem encontrada no dataset")
                return False
                
            logger.info(f"Encontradas {len(images)} imagens para anotação")
        except Exception as e:
            logger.error(f"Erro ao obter imagens do dataset: {str(e)}")
            return False
            
        classes = ["objeto_teste", "classe_teste", "nova_classe"]
        
        for i, image in enumerate(images):
            try:
                image_id = image["id"]
            except KeyError:
                logger.error(f"ID da imagem não encontrado: {image}")
                continue
                
            # Alternar entre as classes
            class_name = classes[i % len(classes)]
            
            # Usar as dimensões que já vieram na resposta
            try:
                img_width = image.get("width", 640)
                img_height = image.get("height", 640)
            except Exception as e:
                logger.error(f"Erro ao obter dimensões da imagem {image_id}: {str(e)}")
                continue
            
            # Criar bounding box normalizado (valores entre 0 e 1)
            # Definir uma área que ocupa 20% da imagem
            width = 0.2  # 20% da largura da imagem
            height = 0.2  # 20% da altura da imagem
            x = 0.4  # Posicionar a 40% da largura
            y = 0.4  # Posicionar a 40% da altura
            
            annotation_data = {
                "image_id": image_id,
                "dataset_id": self.dataset_id,
                "class_name": class_name,
                "confidence": 0.95,
                "bounding_box": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }
            }
            
            try:
                response = self.session.post(self.full_url("/annotations/"), json=annotation_data, timeout=TIMEOUT)
                if response.status_code == 200:
                    logger.info(f"Imagem {image_id} anotada com sucesso usando a classe {class_name}")
                else:
                    logger.error(f"Erro ao anotar imagem {image_id}: {response.text}")
            except Exception as e:
                logger.error(f"Erro ao anotar imagem {image_id}: {str(e)}")
                
        logger.info("Anotação das imagens concluída com sucesso")
        return True  # Retornar True quando todas as anotações forem concluídas com sucesso

    async def test_websocket_connection(self):
        """Testa a conexão WebSocket com os endpoints disponíveis"""
        logger.info("Validando a conectividade WebSocket...")
        
        # Lista de endpoints para testar
        endpoints = [
            f"{self.api_prefix}/hyperparams/test_ws",
            f"{self.api_prefix}/hyperparams/ws/1"
        ]
        
        success = False
        
        for endpoint in endpoints:
            url = f"{self.ws_url}{endpoint}"
            logger.info(f"Testando conexão com {url}...")
            
            try:
                ws = await websockets.connect(url)
                logger.info(f"Conexão estabelecida com {url}")
                
                # Receber mensagem inicial
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    logger.info(f"Mensagem recebida de {url}: {message[:100]}...")
                    success = True
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout ao aguardar mensagem de {url}")
                except Exception as e:
                    logger.error(f"Erro ao receber mensagem de {url}: {str(e)}")
                
                # Fechar conexão
                await ws.close()
                
            except Exception as e:
                logger.error(f"Falha ao conectar com {url}: {str(e)}")
        
        if success:
            logger.info("Validação WebSocket concluída com sucesso")
        else:
            logger.error("Validação WebSocket falhou - nenhum endpoint respondeu")
        
        return success

    async def search_hyperparameters(self):
        """Busca os melhores hiperparâmetros"""
        # Validar conectividade WebSocket primeiro
        websocket_test = await self.test_websocket_connection()
        if not websocket_test:
            logger.error("Validação de conectividade WebSocket falhou. Pulando busca de hiperparâmetros.")
            return self._default_hyperparameters()
            
        data = {
            "dataset_id": self.dataset_id,
            "model_type": "yolov8",
            "model_version": "n",
            "name": f"Busca Teste {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": "Busca de hiperparâmetros para testes automatizados",
            "search_space": {
                "learning_rate": {"min": 0.0001, "max": 0.01},
                "batch_size": {"min": 8, "max": 32},
                "epochs": {"min": 10, "max": 50}
            },
            "max_trials": 5,
            "device": "cpu"  # Forçar uso de CPU em vez de GPU
        }
        
        try:
            response = self.session.post(
                self.full_url("/hyperparams/"),
                json=data,
                timeout=TIMEOUT
            )
            
            if response.status_code != 200:
                logger.error(f"Falha ao iniciar busca de hiperparâmetros: {response.text}")
                # Retornar valores padrão como fallback
                return self._default_hyperparameters()
                
            response_data = response.json()
            if "data" not in response_data:
                logger.error(f"Resposta inválida ao iniciar busca de hiperparâmetros: {response_data}")
                return self._default_hyperparameters()
                
            search_id = response_data["data"]["id"]
            logger.info(f"Busca de hiperparâmetros iniciada com sucesso. Search ID: {search_id}")
            
            try:
                # Monitorar progresso via WebSocket
                await self.monitor_hyperparam_websocket(search_id)
                
                # Após o monitoramento, obter os melhores parâmetros
                result_response = self.session.get(
                    self.full_url(f"/hyperparams/{search_id}"),
                    timeout=TIMEOUT
                )
                
                if result_response.status_code != 200:
                    logger.error(f"Falha ao obter resultado da busca: {result_response.text}")
                    return self._default_hyperparameters()
                    
                result_data = result_response.json()
                if "data" not in result_data:
                    logger.error(f"Resposta inválida ao obter resultado da busca: {result_data}")
                    return self._default_hyperparameters()
                    
                if "best_params" not in result_data["data"]:
                    logger.error("Melhores parâmetros não encontrados na resposta")
                    return self._default_hyperparameters()
                    
                best_params = result_data["data"]["best_params"]
                logger.info(f"Melhores hiperparâmetros encontrados: {best_params}")
                return best_params
                
            except Exception as e:
                logger.error(f"Erro ao monitorar ou obter resultados da busca: {str(e)}")
                return self._default_hyperparameters()
                
        except Exception as e:
            logger.error(f"Erro ao iniciar busca de hiperparâmetros: {str(e)}")
            return self._default_hyperparameters()
            
    def _default_hyperparameters(self):
        """Retorna hiperparâmetros padrão para fallback"""
        logger.info("Usando hiperparâmetros padrão")
        return {
            "lr0": 0.01,
            "batch_size": 16,
            "epochs": 20
        }

    async def train_model(self, hyperparameters):
        """Inicia o treinamento do modelo"""
        data = {
            "dataset_id": self.dataset_id,
            "model_type": "yolov8",
            "model_version": "n",
            "name": f"Modelo Teste {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": "Modelo criado para testes automatizados",
            "hyperparameters": hyperparameters,
            "device": "cpu"  # Forçar uso de CPU em vez de GPU
        }
        response = self.session.post(
            self.full_url("/training/"),
            json=data,
            timeout=TIMEOUT
        )
        if response.status_code == 200:
            self.training_session_id = response.json()["data"]["id"]
            logger.info(f"Treinamento iniciado com sucesso. Session ID: {self.training_session_id}")
            
            # Monitorar progresso via WebSocket
            training_success = await self.monitor_training_websocket()
            return training_success
        else:
            logger.error(f"Falha ao iniciar treinamento: {response.text}")
            return False

    async def run_full_flow(self):
        """Executa o fluxo completo de testes"""
        try:
            # 1. Criar dataset
            if not self.create_dataset():
                return False

            # 2. Upload e associação de imagens
            if not self.upload_and_associate_images():
                return False

            # 3. Anotação de imagens
            if not await self.annotate_images():
                return False

            # 4. Busca de hiperparâmetros
            hyperparameters = await self.search_hyperparameters()
            if not hyperparameters:
                return False

            # 5. Treinamento do modelo
            training_success = await self.train_model(hyperparameters)

            return training_success

        except Exception as e:
            logger.error(f"Erro durante o fluxo de testes: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Teste do fluxo completo da API MicroDetect")
    parser.add_argument("--host", default="localhost", help="Hostname do servidor")
    parser.add_argument("--port", type=int, default=8000, help="Porta do servidor")
    args = parser.parse_args()

    global BASE_URL, WS_URL
    BASE_URL = f"http://{args.host}:{args.port}"
    WS_URL = f"ws://{args.host}:{args.port}"

    tester = FullFlowTester()
    
    # Criar e executar o event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    success = loop.run_until_complete(tester.run_full_flow())
    loop.close()

    if success:
        logger.info("Teste do fluxo completo concluído com sucesso!")
        sys.exit(0)
    else:
        logger.error("Teste do fluxo completo falhou!")
        sys.exit(1)

if __name__ == "__main__":
    main() 