from typing import List, Optional
from sqlalchemy.orm import Session

from microdetect.models.dataset import Dataset
from microdetect.schemas.dataset import DatasetCreate, DatasetUpdate


class DatasetService:
    def __init__(self, db: Session):
        self.db = db

    def get(self, dataset_id: int) -> Optional[Dataset]:
        """
        Obter um dataset pelo ID.
        """
        return self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    def get_multi(self, skip: int = 0, limit: int = 100) -> List[Dataset]:
        """
        Obter múltiplos datasets.
        """
        return self.db.query(Dataset).offset(skip).limit(limit).all()
    
    def create(self, dataset_in: DatasetCreate) -> Dataset:
        """
        Criar um novo dataset.
        """
        db_dataset = Dataset(**dataset_in.dict())
        self.db.add(db_dataset)
        self.db.commit()
        self.db.refresh(db_dataset)
        return db_dataset
    
    def update(self, db_dataset: Dataset, dataset_in: DatasetUpdate) -> Dataset:
        """
        Atualizar um dataset existente.
        """
        update_data = dataset_in.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_dataset, field, value)
        
        self.db.commit()
        self.db.refresh(db_dataset)
        return db_dataset
    
    def remove(self, dataset_id: int) -> None:
        """
        Remover um dataset.
        """
        db_dataset = self.get(dataset_id)
        if db_dataset:
            self.db.delete(db_dataset)
            self.db.commit()