from sqlalchemy import Column, String, Float, UUID, ForeignKey, Integer, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from db import Base
from uuid import uuid4

class Localization(Base):
    __tablename__ = "localizations"
    
    zone_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    zone_name = Column(String)
    
    classes_areas = relationship("ClassesArea", back_populates="localization")


class Classification(Base):
    __tablename__ = "classifications"

    classification_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    class_name = Column(String)

    classes_areas = relationship("ClassesArea", back_populates="classification")

class ClassesArea(Base):
    __tablename__ = "classes_areas"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    localization_id = Column(UUID(as_uuid=True), ForeignKey("localizations.zone_id"))
    area = Column(Float)
    class_id = Column(UUID(as_uuid=True), ForeignKey("classifications.classification_id"))

    classification = relationship("Classification", back_populates="classes_areas")
    localization = relationship("Localization", back_populates="classes_areas")
    projects = relationship("Project", back_populates="classes_area")

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String, unique=True)
    password = Column(String)
    
    projects = relationship("Project", back_populates="user")

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    code = Column(UUID(as_uuid=True), default=uuid4)
    
    projects = relationship("Project", back_populates="patient")


class Status(Base):
    __tablename__ = "statuses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    status_name = Column(String)

    projects = relationship("Project", back_populates="status")


class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    upload_date = Column(TIMESTAMP)
    classes_area_id = Column(UUID(as_uuid=True), ForeignKey("classes_areas.id"))
    name = Column(String)
    status_id = Column(UUID(as_uuid=True), ForeignKey("statuses.id"))

    user = relationship("User", back_populates="projects")
    patient = relationship("Patient", back_populates="projects")
    classes_area = relationship("ClassesArea", back_populates="projects")
    status = relationship("Status", back_populates="projects")