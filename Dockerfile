FROM nexus.rz.bankenit.de:50004/python:3.8.20-bookworm

ARG BUILD_CONTEXT=openshift

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV BIN_DIR=/app/docker/bin
ENV CERTS_DIR=/app/docker/conf/crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ENV PIP_RESUME_RETRIES=10
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY docker ./docker
COPY requirements.txt ./

RUN echo "### Import certificates" && \
    chmod +x "$BIN_DIR"/install_certs.sh && \
    "$BIN_DIR"/install_certs.sh "${CERTS_DIR}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

RUN --mount=type=secret,id=pip_conf,target=/etc/pip.conf \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/docker/bin/*.sh

CMD ["/bin/bash"]
