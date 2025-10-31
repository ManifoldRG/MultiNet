# Stage 1: Build environment
# Use a minimal 'base' image that only provides the essential CUDA runtime.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3.10-venv \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv $VENV_PATH

ENV PATH="$VENV_PATH/bin:$PATH"

COPY src/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

#-------------------------------------------------------------------
# Install specific requirements for adapter
# Replace with your own requirements
COPY src/v1/modules/Magma/magma_action_requirements.txt .
RUN pip install --no-cache-dir -r magma_action_requirements.txt
#-------------------------------------------------------------------
    
# Stage 2: Runtime image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Install Python3 in the runtime stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Set user and group ids for host user
# Override with --build-arg if needed:
# docker build --build-arg UID=1001 --build-arg GID=1001 .
ARG UID=1000
ARG GID=1000

# Add a group and user
RUN addgroup --gid $GID app && \
    adduser --uid $UID --ingroup app --disabled-password --gecos '' app

# Set the working directory
WORKDIR /home/app/multinet

# Create mount points and set ownership
RUN mkdir /models /data && \
    chown -R $UID:$GID /home/app/multinet /models /data

# Copy the virtual environment and source code from the builder stage
COPY --from=builder /opt/venv /opt/venv

COPY --chown=$UID:$GID src/ src/
COPY --chown=$UID:$GID scripts/ scripts/
COPY --chown=$UID:$GID definitions/ definitions/
USER app

# Set runtime environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "scripts/eval_harness/evaluate.py"]
CMD ["--help"]
