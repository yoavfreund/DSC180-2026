FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY learner/ ./learner/
RUN make -C learner test && make -C learner

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BOOSTING_DATA_DIR=/data \
    BOOSTING_LIBRARY=/app/learner/libweak.so
WORKDIR /app
RUN pip install --no-cache-dir numpy==2.3.5
RUN pip install --no-cache-dir 'PyJWT[crypto]==2.10.1'
COPY main.py predict_ensemble.py config.py ./
COPY agents/ ./agents/
COPY common/ ./common/
COPY coordinator/ ./coordinator/
COPY --from=builder /build/learner/libweak.so ./learner/libweak.so
ENTRYPOINT ["python", "main.py"]
