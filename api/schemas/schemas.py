from datetime import datetime
from pydantic import BaseModel, EmailStr, field_serializer
from typing import Optional, List, Dict
from uuid import UUID


class LocalizationSchema(BaseModel):
    zone_name: str


class ClassificationSchema(BaseModel):
    class_name: str


class ClassesAreaSchema(BaseModel):
    data: List[List[str]]

class ClassesAreaCreateSchema(BaseModel):
    localization_name: str
    classification_name: str
    area: Dict[str, float]


class UserSchema(BaseModel):
    email: EmailStr
    password: str


class PatientSchema(BaseModel):
    pass


class StatusSchema(BaseModel):
    status_name: str


class ResearchSchema(BaseModel):
    user_id: UUID
    patient_id: UUID
    upload_date: Optional[datetime] = None
    name: str
    status_id: UUID

    @field_serializer('upload_date')
    def serialize_dt(self, dt: Optional[datetime], _info):
        if dt is None:
            return None
        return dt.replace(microsecond=0).isoformat(sep=' ')

class ResearchCreateSchema(BaseModel):
    user_email: EmailStr
    patient_id: UUID
    status_id:  Optional[UUID] = None
    name: Optional[str] = None
    upload_date: Optional[datetime] = None

    class Config:
        from_attributes = True
class ResearchStatusUpdateByIdSchema(BaseModel):
    status_id: UUID


class Localization(LocalizationSchema):
    zone_id: UUID

    class Config:
        from_attributes = True


class Classification(ClassificationSchema):
    classification_id: UUID

    class Config:
        from_attributes = True


class ClassesArea(BaseModel):
    id: UUID
    area: float
    localization: Localization
    classification: Classification
    class Config:
        from_attributes = True


class User(UserSchema):
    id: UUID

    class Config:
        from_attributes = True


class Patient(PatientSchema):
    id: UUID
    code: str

    class Config:
        from_attributes = True


class Status(StatusSchema):
    id: UUID

    class Config:
        from_attributes = True


class Research(BaseModel):
    id: UUID
    name:str
    upload_date: Optional[datetime] = None
    user: User
    patient: Patient
    classes_areas: List[ClassesArea] = []
    status: Status

    class Config:
        from_attributes = True