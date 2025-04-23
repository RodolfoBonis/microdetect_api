#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sqlite3
import shutil
from pathlib import Path

def main():
    # Obter o diretório home do usuário
    home_dir = str(Path.home())
    data_dir = os.path.join(home_dir, ".microdetect", "data")
    db_path = os.path.join(home_dir, ".microdetect", "microdetect.db")
    
    print("Este script irá:")
    print(f"1. Excluir todo o conteúdo de {data_dir}")
    print(f"2. Limpar tabelas específicas no banco de dados {db_path}")
    print("\nAVISO: Esta operação não pode ser desfeita!")
    
    # Solicitar confirmação do usuário
    confirmacao = input("\nVocê tem certeza que deseja continuar? (s/N): ").lower()
    
    if confirmacao != 's':
        print("Operação cancelada.")
        return
    
    # Passo 1: Limpar o diretório de dados
    print(f"\nLimpando o diretório {data_dir}...")
    try:
        if os.path.exists(data_dir):
            # Remover todos os arquivos e subdiretórios
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Diretório removido: {item_path}")
                else:
                    os.remove(item_path)
                    print(f"Arquivo removido: {item_path}")
            print(f"Conteúdo de {data_dir} foi removido com sucesso.")
        else:
            print(f"O diretório {data_dir} não existe.")
    except Exception as e:
        print(f"Erro ao limpar o diretório de dados: {e}")
    
    # Passo 2: Executar as consultas SQL
    print(f"\nConectando ao banco de dados {db_path}...")
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Lista de comandos SQL para executar
            sql_commands = [
                "DELETE FROM hyperparam_searches;",
                "DELETE FROM training_sessions;",
                "DELETE FROM dataset_images;",
                "DELETE FROM images;",
                "DELETE FROM annotations;"
            ]
            
            # Executar cada comando SQL
            for cmd in sql_commands:
                try:
                    cursor.execute(cmd)
                    print(f"Executado: {cmd}")
                except sqlite3.Error as e:
                    print(f"Erro ao executar '{cmd}': {e}")
            
            # Confirmar as alterações e fechar a conexão
            conn.commit()
            conn.close()
            print("\nOperações no banco de dados concluídas com sucesso.")
        else:
            print(f"O banco de dados {db_path} não existe.")
    except Exception as e:
        print(f"Erro ao conectar ou manipular o banco de dados: {e}")
    
    print("\nProcesso de limpeza concluído.")

if __name__ == "__main__":
    main() 