"""criado annotations

Revision ID: eacde9ec6022
Revises: f004cf6cd85c
Create Date: 2025-03-28 16:29:05.429095

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = 'eacde9ec6022'
down_revision = 'f004cf6cd85c'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('ix_annotations_id', table_name='annotations')
    op.drop_table('annotations')
    op.drop_index('ix_inference_results_id', table_name='inference_results')
    op.drop_table('inference_results')
    op.drop_index('ix_images_id', table_name='images')
    op.drop_table('images')
    op.drop_index('ix_dataset_images_dataset_id', table_name='dataset_images')
    op.drop_index('ix_dataset_images_id', table_name='dataset_images')
    op.drop_index('ix_dataset_images_image_id', table_name='dataset_images')
    op.drop_table('dataset_images')
    op.drop_index('ix_training_sessions_id', table_name='training_sessions')
    op.drop_table('training_sessions')
    op.drop_index('ix_models_id', table_name='models')
    op.drop_table('models')
    op.drop_index('ix_datasets_id', table_name='datasets')
    op.drop_table('datasets')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('datasets',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('name', sa.VARCHAR(length=255), nullable=False),
    sa.Column('description', sa.TEXT(), nullable=True),
    sa.Column('created_at', sa.DATETIME(), nullable=True),
    sa.Column('updated_at', sa.DATETIME(), nullable=True),
    sa.Column('classes', sqlite.JSON(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_datasets_id', 'datasets', ['id'], unique=False)
    op.create_table('models',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('name', sa.VARCHAR(length=255), nullable=False),
    sa.Column('description', sa.TEXT(), nullable=True),
    sa.Column('path', sa.VARCHAR(length=255), nullable=False),
    sa.Column('version', sa.VARCHAR(length=50), nullable=True),
    sa.Column('created_at', sa.DATETIME(), nullable=True),
    sa.Column('updated_at', sa.DATETIME(), nullable=True),
    sa.Column('config', sqlite.JSON(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_models_id', 'models', ['id'], unique=False)
    op.create_table('training_sessions',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('name', sa.VARCHAR(length=255), nullable=False),
    sa.Column('start_time', sa.DATETIME(), nullable=True),
    sa.Column('end_time', sa.DATETIME(), nullable=True),
    sa.Column('status', sa.VARCHAR(length=50), nullable=True),
    sa.Column('hyperparameters', sqlite.JSON(), nullable=True),
    sa.Column('metrics', sqlite.JSON(), nullable=True),
    sa.Column('dataset_id', sa.INTEGER(), nullable=False),
    sa.Column('model_id', sa.INTEGER(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_training_sessions_id', 'training_sessions', ['id'], unique=False)
    op.create_table('dataset_images',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('dataset_id', sa.INTEGER(), nullable=False),
    sa.Column('image_id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['image_id'], ['images.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_dataset_images_image_id', 'dataset_images', ['image_id'], unique=False)
    op.create_index('ix_dataset_images_id', 'dataset_images', ['id'], unique=False)
    op.create_index('ix_dataset_images_dataset_id', 'dataset_images', ['dataset_id'], unique=False)
    op.create_table('images',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('file_name', sa.VARCHAR(length=255), nullable=False),
    sa.Column('file_path', sa.VARCHAR(length=255), nullable=False),
    sa.Column('file_size', sa.INTEGER(), nullable=True),
    sa.Column('width', sa.INTEGER(), nullable=True),
    sa.Column('height', sa.INTEGER(), nullable=True),
    sa.Column('created_at', sa.DATETIME(), nullable=True),
    sa.Column('updated_at', sa.DATETIME(), nullable=True),
    sa.Column('dataset_id', sa.INTEGER(), nullable=True),
    sa.Column('metadata', sqlite.JSON(), nullable=True),
    sa.Column('url', sa.VARCHAR(length=255), nullable=True),
    sa.Column('image_metadata', sqlite.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_images_id', 'images', ['id'], unique=False)
    op.create_table('inference_results',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('predictions', sqlite.JSON(), nullable=True),
    sa.Column('metrics', sqlite.JSON(), nullable=True),
    sa.Column('created_at', sa.DATETIME(), nullable=True),
    sa.Column('image_id', sa.INTEGER(), nullable=True),
    sa.Column('model_id', sa.INTEGER(), nullable=True),
    sa.ForeignKeyConstraint(['image_id'], ['images.id'], ),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_inference_results_id', 'inference_results', ['id'], unique=False)
    op.create_table('annotations',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('x_min', sa.FLOAT(), nullable=True),
    sa.Column('y_min', sa.FLOAT(), nullable=True),
    sa.Column('x_max', sa.FLOAT(), nullable=True),
    sa.Column('y_max', sa.FLOAT(), nullable=True),
    sa.Column('class_name', sa.VARCHAR(length=100), nullable=True),
    sa.Column('confidence', sa.FLOAT(), nullable=True),
    sa.Column('created_at', sa.DATETIME(), nullable=True),
    sa.Column('updated_at', sa.DATETIME(), nullable=True),
    sa.Column('image_id', sa.INTEGER(), nullable=False),
    sa.Column('dataset_id', sa.INTEGER(), nullable=True),
    sa.Column('x', sa.FLOAT(), nullable=True),
    sa.Column('y', sa.FLOAT(), nullable=True),
    sa.Column('width', sa.FLOAT(), nullable=True),
    sa.Column('height', sa.FLOAT(), nullable=True),
    sa.Column('area', sa.FLOAT(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['image_id'], ['images.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_annotations_id', 'annotations', ['id'], unique=False)
    # ### end Alembic commands ### 