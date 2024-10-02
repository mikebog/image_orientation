# Используем базовый образ Python 3.9 (или любой другой подходящий)
FROM python:3.9-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Устанавливаем переменные среды для отключения CUDA
ENV TORCH_CUDA_ARCH_LIST=""
ENV CUDA_HOME="/usr/local/cuda"

# Устанавливаем pip и необходимые инструменты
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем PyTorch без CUDA
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем остальные зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта в рабочую директорию
COPY . .

# Указываем порт, который будет использоваться приложением (например, 8000 для FastAPI)
EXPOSE 8000

# Команда для запуска FastAPI приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]