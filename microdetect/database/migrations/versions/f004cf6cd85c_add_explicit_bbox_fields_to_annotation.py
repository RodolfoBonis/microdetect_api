"""Add explicit bbox fields to Annotation

Revision ID: f004cf6cd85c
Revises: 4204cd67bb2f
Create Date: 2025-03-27 22:00:09.661702

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f004cf6cd85c'
down_revision: Union[str, None] = '4204cd67bb2f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite não suporta ALTER TABLE para modificar colunas existentes
    # Vamos apenas adicionar as novas colunas
    
    # Adicionar campos explícitos para bounding box
    op.add_column('annotations', sa.Column('x', sa.Float(), nullable=True))
    op.add_column('annotations', sa.Column('y', sa.Float(), nullable=True))
    op.add_column('annotations', sa.Column('width', sa.Float(), nullable=True))
    op.add_column('annotations', sa.Column('height', sa.Float(), nullable=True))
    op.add_column('annotations', sa.Column('area', sa.Float(), nullable=True))
    
    # Ignoramos as outras mudanças geradas automaticamente que podem causar problemas no SQLite


def downgrade() -> None:
    # Remover as colunas adicionadas
    op.drop_column('annotations', 'area')
    op.drop_column('annotations', 'height')
    op.drop_column('annotations', 'width')
    op.drop_column('annotations', 'y')
    op.drop_column('annotations', 'x') 