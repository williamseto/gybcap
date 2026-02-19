FROM python:3.11-slim

WORKDIR /app

# Install system deps for TA-Lib and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
RUN wget -q https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr && make -j$(nproc) && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

COPY requirements.txt .
# Install only the packages needed for the dashboard (skip torch/cuda etc.)
RUN pip install --no-cache-dir \
    numpy pandas scikit-learn xgboost scipy \
    fastapi "uvicorn[standard]" apscheduler \
    yfinance requests python-dotenv \
    ta-lib hmmlearn

COPY strategies/ ./strategies/
COPY dashboard/ ./dashboard/

# Persistent volume mount point
VOLUME ["/app/dashboard/cache", "/app/raw_data"]

ENV PYTHONPATH=/app
ENV DASHBOARD_DATA_MODE=yfinance_only
ENV DASHBOARD_PORT=8080
ENV DASHBOARD_HOST=0.0.0.0

EXPOSE 8080

CMD ["python", "-m", "dashboard.server"]
