from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_serializer
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


# Входная схема (регистрация/логин): принимает пароль, но НИКОГДА не используется как response_model.
class UserSchema(BaseModel):
    email: EmailStr
    password: str


# Ответ на успешный логин: JWT-токен доступа.
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


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
    model_config = ConfigDict(from_attributes=True)

    user_email: EmailStr
    patient_id: UUID
    status_id:  Optional[UUID] = None
    name: Optional[str] = None
    upload_date: Optional[datetime] = None

class ResearchStatusUpdateByIdSchema(BaseModel):
    status_id: UUID


class ResearchFailSchema(BaseModel):
    error_message: Optional[str] = None


class Localization(LocalizationSchema):
    model_config = ConfigDict(from_attributes=True)

    zone_id: UUID


class Classification(ClassificationSchema):
    model_config = ConfigDict(from_attributes=True)

    classification_id: UUID


class ClassesArea(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    area: float
    localization: Localization
    classification: Classification


# Выходная схема: НЕ наследует UserSchema, поэтому пароль (хеш) никогда не попадает в ответы API.
class User(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr


class Patient(PatientSchema):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    code: str


class Status(StatusSchema):
    model_config = ConfigDict(from_attributes=True)

    id: UUID


class Research(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name:str
    upload_date: Optional[datetime] = None
    error_message: Optional[str] = None
    user: User
    patient: Patient
    classes_areas: List[ClassesArea] = Field(default_factory=list)
    status: Status
