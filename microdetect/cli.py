"""
Interface de linha de comando para o MicroDetect
"""

import argparse
import sys
import logging
from microdetect import __version__

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('microdetect')


def main():
    """Ponto de entrada principal para o CLI"""
    parser = argparse.ArgumentParser(description='MicroDetect Backend')
    subparsers = parser.add_subparsers(dest='command')

    # Comando start-server
    start_parser = subparsers.add_parser('start-server', help='Iniciar o servidor da API')
    start_parser.add_argument('--host', default='127.0.0.1', help='Host para o servidor')
    start_parser.add_argument('--port', type=int, default=8000, help='Porta para o servidor')
    start_parser.add_argument('--data-dir', type=str, help='Diretório para armazenar dados')

    # Comando version
    version_parser = subparsers.add_parser('version', help='Exibir a versão')

    # Comando create-migration
    migration_parser = subparsers.add_parser('create-migration', help='Criar uma nova migração')
    migration_parser.add_argument('message', help='Mensagem da migração')

    # Comando apply-migrations
    apply_migrations_parser = subparsers.add_parser('apply-migrations', help='Aplicar migrações pendentes')

    # Comando check-updates
    check_updates_parser = subparsers.add_parser('check-updates', help='Verificar atualizações disponíveis')

    args = parser.parse_args()

    if args.command == 'start-server':
        from microdetect.server import start_server
        start_server(host=args.host, port=args.port, data_dir=args.data_dir)

    elif args.command == 'version':
        print(f"MicroDetect versão {__version__}")

    elif args.command == 'create-migration':
        from microdetect.database.migrations import create_migration
        create_migration(args.message)

    elif args.command == 'apply-migrations':
        from microdetect.database.migrations import apply_migrations
        apply_migrations()

    elif args.command == 'check-updates':
        print(f"Versão atual: {__version__}")
        # Este comando seria usado pelo aplicativo Flutter para verificar se há atualizações
        # Você pode implementar uma lógica mais sofisticada se necessário

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()