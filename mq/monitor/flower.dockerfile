FROM python:3.12-slim
RUN pip install --no-cache-dir flower==2.0.*
# Use exec form so container PID is Flower process; ENV var passed via docker-compose
ENTRYPOINT ["flower"]
CMD ["--broker=${CELERY_BROKER_URL}", "--port=5555"]
