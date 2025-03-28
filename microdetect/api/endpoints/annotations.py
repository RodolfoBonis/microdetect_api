from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional, Dict, Any
from microdetect.database.database import get_db
from microdetect.models.annotation import Annotation
from microdetect.models.image import Image
from microdetect.models.dataset import Dataset
from microdetect.schemas.annotation import (
    AnnotationCreate, AnnotationUpdate, 
    AnnotationResponse, AnnotationBatch
)

router = APIRouter()

@router.post("/", response_model=AnnotationResponse)
def create_annotation(
    annotation: AnnotationCreate, 
    db: Session = Depends(get_db)
):
    """Cria uma nova anotação"""
    # Verificar se a imagem existe
    image = db.query(Image).filter(Image.id == annotation.image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    
    # Se o dataset_id foi fornecido, verificar se existe
    if annotation.dataset_id:
        dataset = db.query(Dataset).filter(Dataset.id == annotation.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset não encontrado")
        
        # Verificar se a classe está definida no dataset
        if annotation.class_name and dataset.classes:
            if annotation.class_name not in dataset.classes:
                # Se a classe não existe no dataset, adicioná-la
                classes = dataset.classes or []
                classes.append(annotation.class_name)
                dataset.classes = classes
                db.commit()
    
    # Criar a anotação
    db_annotation = Annotation(**annotation.dict())
    db.add(db_annotation)
    db.commit()
    db.refresh(db_annotation)
    
    return db_annotation

@router.get("/", response_model=List[AnnotationResponse])
def list_annotations(
    image_id: Optional[int] = None,
    dataset_id: Optional[int] = None,
    class_name: Optional[str] = None,
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """Lista anotações com filtros opcionais"""
    query = db.query(Annotation)
    
    if image_id:
        query = query.filter(Annotation.image_id == image_id)
    
    if dataset_id:
        query = query.filter(Annotation.dataset_id == dataset_id)
    
    if class_name:
        query = query.filter(Annotation.class_name == class_name)
    
    annotations = query.offset(skip).limit(limit).all()
    return annotations

@router.get("/{annotation_id}", response_model=AnnotationResponse)
def get_annotation(annotation_id: int, db: Session = Depends(get_db)):
    """Obtém uma anotação específica"""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if annotation is None:
        raise HTTPException(status_code=404, detail="Anotação não encontrada")
    return annotation

@router.put("/{annotation_id}", response_model=AnnotationResponse)
def update_annotation(
    annotation_id: int, 
    annotation: AnnotationUpdate, 
    db: Session = Depends(get_db)
):
    """Atualiza uma anotação existente"""
    db_annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if db_annotation is None:
        raise HTTPException(status_code=404, detail="Anotação não encontrada")
    
    # Se estiver atualizando a classe, verificar se ela existe no dataset
    if hasattr(annotation, 'class_name') and annotation.class_name is not None:
        dataset_id = db_annotation.dataset_id
        if dataset_id:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset and dataset.classes:
                if annotation.class_name not in dataset.classes:
                    # Se a classe não existe no dataset, adicioná-la
                    classes = dataset.classes or []
                    classes.append(annotation.class_name)
                    dataset.classes = classes
                    db.commit()
    
    # Atualizar os campos da anotação
    annotation_data = annotation.dict(exclude_unset=True)
    for key, value in annotation_data.items():
        setattr(db_annotation, key, value)
    
    db.commit()
    db.refresh(db_annotation)
    
    return db_annotation

@router.delete("/{annotation_id}")
def delete_annotation(annotation_id: int, db: Session = Depends(get_db)):
    """Remove uma anotação"""
    db_annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if db_annotation is None:
        raise HTTPException(status_code=404, detail="Anotação não encontrada")
    
    db.delete(db_annotation)
    db.commit()
    
    return {"message": "Anotação removida com sucesso"}

@router.post("/batch", response_model=List[AnnotationResponse])
def create_annotations_batch(
    annotations: AnnotationBatch, 
    db: Session = Depends(get_db)
):
    """Cria múltiplas anotações em lote"""
    results = []
    
    # Verificar se a imagem existe
    image = db.query(Image).filter(Image.id == annotations.image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    
    # Se o dataset_id foi fornecido, verificar se existe
    dataset = None
    if annotations.dataset_id:
        dataset = db.query(Dataset).filter(Dataset.id == annotations.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    # Processar cada anotação do lote
    for annotation_data in annotations.annotations:
        # Definir image_id e dataset_id do lote para esta anotação
        annotation_dict = annotation_data.dict()
        annotation_dict["image_id"] = annotations.image_id
        if annotations.dataset_id:
            annotation_dict["dataset_id"] = annotations.dataset_id
        
        # Verificar se a classe está definida no dataset
        if dataset and annotation_dict.get("class_name"):
            class_name = annotation_dict["class_name"]
            if dataset.classes and class_name not in dataset.classes:
                # Se a classe não existe no dataset, adicioná-la
                classes = dataset.classes or []
                classes.append(class_name)
                dataset.classes = classes
                db.commit()
        
        # Criar a anotação
        db_annotation = Annotation(**annotation_dict)
        db.add(db_annotation)
        results.append(db_annotation)
    
    db.commit()
    
    # Atualizar os objetos com os IDs gerados
    for annotation in results:
        db.refresh(annotation)
    
    return results

@router.get("/dataset/{dataset_id}/classes", response_model=List[Dict[str, Any]])
def get_dataset_classes(dataset_id: int, db: Session = Depends(get_db)):
    """Obtém todas as classes definidas em um dataset e suas contagens"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    # Obter a lista de classes definidas no dataset
    defined_classes = dataset.classes or []
    
    # Contar anotações por classe
    class_query = text("""
        SELECT a.class_name, COUNT(a.id) as count
        FROM annotations a
        JOIN images i ON i.id = a.image_id
        WHERE (i.dataset_id = :dataset_id OR 
            EXISTS (SELECT 1 FROM dataset_images di WHERE di.dataset_id = :dataset_id AND di.image_id = i.id))
            AND a.class_name IS NOT NULL
        GROUP BY a.class_name
        ORDER BY count DESC
    """)
    class_counts = db.execute(class_query, {"dataset_id": dataset_id}).fetchall()
    
    # Calcular contagem de anotações
    annotation_query = text("""
        SELECT COUNT(a.id) 
        FROM annotations a
        JOIN images i ON i.id = a.image_id
        WHERE i.dataset_id = :dataset_id OR 
            EXISTS (SELECT 1 FROM dataset_images di WHERE di.dataset_id = :dataset_id AND di.image_id = i.id)
    """)
    total_annotations = db.execute(annotation_query, {"dataset_id": dataset_id}).scalar() or 0
    
    # Criar dicionário de classes e suas contagens
    class_distributions = {}
    for class_name, count in class_counts:
        percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
        class_distributions[class_name] = {
            "count": count,
            "percentage": round(percentage, 2)
        }
    
    # Montar a resposta incluindo todas as classes definidas
    result = []
    for class_name in defined_classes:
        if class_name in class_distributions:
            result.append({
                "class_name": class_name,
                "count": class_distributions[class_name]["count"],
                "percentage": class_distributions[class_name]["percentage"],
                "is_used": True
            })
        else:
            result.append({
                "class_name": class_name,
                "count": 0,
                "percentage": 0,
                "is_used": False
            })
    
    # Adicionar classes que têm anotações mas não estão definidas no dataset
    for class_name, details in class_distributions.items():
        if class_name not in defined_classes:
            result.append({
                "class_name": class_name,
                "count": details["count"],
                "percentage": details["percentage"],
                "is_used": True,
                "is_undefined": True
            })
    
    return result 