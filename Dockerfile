# One Linux environment for every teammate (macOS, Windows, Linux hosts).
# The Agent SDK wheel ships its own `claude` program, so no Node.js is needed.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev binutils make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN make -C eval eval

# Agent-written C runs as an unprivileged user that sees only /app and the mounted data.
RUN useradd --create-home agent && mkdir -p /app/runs && chown -R agent /app
USER agent
ENV FACE_DATA=/data PYTHONUNBUFFERED=1
CMD ["python", "-m", "pytest", "-q", "tests"]
