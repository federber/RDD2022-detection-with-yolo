# Используем базовый образ с поддержкой Python и CUDA
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Устанавливаем необходимые зависимости
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости Python
WORKDIR /app
#COPY requirements.txt .
#COPY project_config.json /app/project_config.json
#COPY /China_MotorBike /app/China_MotorBike
COPY yolo_config.yaml .

COPY rdd_yolo.py .
COPY utils.py .

RUN pip3 install --no-cache-dir torch ultralytics opencv-python numpy pandas scikit-learn PyYAML argparse


# Указываем рабочую директорию
WORKDIR /app

# Запуск скрипта
CMD ["python3", "rdd_yolo.py"]