# setup
FROM python:3.11.5

# Install system dependencies FIRST (before any pip installs)
RUN apt-get update && apt-get -y install \
    build-essential \
    libssl-dev \
    ca-certificates \
    libasound2 \
    libasound2-dev \
    portaudio19-dev \
    libsndfile1-dev \
    libffi-dev \
    wget \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app
COPY *.py /app
COPY pyproject.toml /app
COPY src/ /app/src/
COPY examples/ /app/examples/
WORKDIR /app

RUN ls --recursive /app/

# Now pip install will work because system deps are already there
RUN pip3 install --upgrade -r requirements.txt
RUN python -m build .
RUN pip3 install .
RUN pip3 install gunicorn

# Azure TTS config (keep this part as-is)
RUN wget -O - https://www.openssl.org/source/openssl-1.1.1w.tar.gz | tar zxf -
WORKDIR openssl-1.1.1w
RUN ./config --prefix=/usr/local
RUN make -j $(nproc)
RUN make install_sw install_ssldirs
RUN ldconfig -v
ENV SSL_CERT_DIR=/etc/ssl/certs

ENV PYTHONUNBUFFERED=1
WORKDIR /app
EXPOSE 8000

# run
CMD ["gunicorn", "--workers=2", "--log-level", "debug", "--chdir", "examples/server", "--capture-output", "daily-bot-manager:app", "--bind=0.0.0.0:8000"]
