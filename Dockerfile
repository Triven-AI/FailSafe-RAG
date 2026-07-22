# 1. Base Image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. System dependencies for OCR/Processing (Tesseract, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application
COPY . .

# 6. Set Python Path so 'src' module imports work correctly
ENV PYTHONPATH=/app

# (The execution CMD is handled by docker-compose for each microservice)