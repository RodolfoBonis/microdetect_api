"""Remover colunas redundantes da tabela dataset_images

Revision ID: 3f3c375f19d9
Revises: 68b6373e8c4f
Create Date: 2025-03-26 01:42:21.583994

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = '3f3c375f19d9'
down_revision = '68b6373e8c4f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Remover colunas redundantes da tabela dataset_images
    op.drop_column('dataset_images', 'file_path')
    op.drop_column('dataset_images', 'url')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Adicionar as colunas removidas de volta
    op.add_column('dataset_images', sa.Column('url', sa.VARCHAR(length=255), nullable=True))
    op.add_column('dataset_images', sa.Column('file_path', sa.VARCHAR(length=255), nullable=False))
    # ### end Alembic commands ### 