FROM python:3.9-slim

WORKDIR /code

# Install system dependencies (required for OpenCV and imaging libraries)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all application code
COPY . /code

# Run the FastAPI server on port 7860 (Hugging Face default)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
