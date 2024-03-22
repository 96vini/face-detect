FROM ubuntu:latest

# Atualizar repositórios e instalar dependências
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3 \
    python3-pip \
    ffmpeg \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Instalar pacotes Python necessários
RUN pip3 install numpy opencv-python

# Copiar o código do aplicativo para o container
COPY . /app
WORKDIR /app

# Definir a entrada padrão
CMD ["python3", "app.py"]
