import sys
from datetime import datetime
from sqlalchemy import Column, String, Float, ForeignKey, TIMESTAMP, event
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship
from db.db import Base
import uuid

class Localization(Base):
    __tablename__ = "localizations"

    zone_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    zone_name = Column(String)

    classes_areas = relationship("ClassesArea", back_populates="localization")


class Classification(Base):
    __tablename__ = "classifications"

    classification_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    class_name = Column(String)

    classes_areas = relationship("ClassesArea", back_populates="classification")


class ClassesArea(Base):
    __tablename__ = "classes_areas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    localization_id = Column(UUID(as_uuid=True), ForeignKey("localizations.zone_id"))
    area = Column(Float)
    class_id = Column(UUID(as_uuid=True), ForeignKey("classifications.classification_id"))
    research_id = Column(UUID(as_uuid=True), ForeignKey("researches.id"))

    research = relationship("Research", back_populates="classes_areas")
    classification = relationship("Classification", back_populates="classes_areas")
    localization = relationship("Localization", back_populates="classes_areas")


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True)
    password = Column(String)

    researches = relationship("Research", back_populates="user")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(String, unique=True)

    researches = relationship("Research", back_populates="patient", cascade="all, delete-orphan")


class Status(Base):
    __tablename__ = "statuses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status_name = Column(String)

    researches = relationship("Research", back_populates="status")


class Research(Base):
    __tablename__ = "researches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    upload_date = Column(TIMESTAMP)
    name = Column(String)
    status_id = Column(UUID(as_uuid=True), ForeignKey("statuses.id"))
    error_message = Column(String, nullable=True)

    user = relationship("User", back_populates="researches")
    patient = relationship("Patient", back_populates="researches")
    classes_areas = relationship("ClassesArea", back_populates="research", cascade="all, delete-orphan")
    status = relationship("Status", back_populates="researches")
