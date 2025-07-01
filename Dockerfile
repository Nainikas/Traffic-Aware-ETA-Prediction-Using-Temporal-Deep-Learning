# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy everything
COPY requirements.txt .
COPY src ./src
COPY models ./models
COPY data ./data

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
