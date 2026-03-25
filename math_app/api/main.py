from fastapi import FastAPI
from api.routers import processing

app = FastAPI(docs_url="/docs")
app.include_router(processing.router) 