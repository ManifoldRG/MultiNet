FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Set CUDA environment variables
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy entire Magma repository
COPY . /app

# Install Magma with server dependencies
RUN pip install -e ".[server]"

# Set environment variables
ENV MODEL_ID="microsoft/Magma-8B"
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run the server
CMD ["python", "/app/server/main.py"]
