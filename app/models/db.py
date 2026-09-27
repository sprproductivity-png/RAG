from sqlalchemy import Column, String, Integer, Float, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/rag")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class DBTask(Base):
    __tablename__ = "tasks"
    
    task_id = Column(String, primary_key=True, index=True)
    status = Column(String, index=True)
    stage = Column(String)
    progress = Column(Integer, default=0)
    stage_message = Column(Text)
    filename = Column(String)
    file_size = Column(Integer)
    file_type = Column(String)
    collection_name = Column(String)
    title = Column(String, nullable=True)
    cache_path = Column(String)
    chunking_strategy = Column(String)
    use_contextual_retrieval = Column(Boolean, default=True)
    context_provider = Column(String, nullable=True)
    created_at = Column(String)
    started_at = Column(String, nullable=True)
    completed_at = Column(String, nullable=True)
    duration_ms = Column(Float, nullable=True)
    chunks_created = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    sample_chunks = Column(JSON, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)
