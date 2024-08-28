# Use the official TensorFlow Docker image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    curl \
    screen
#    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio

# Copy requirements.txt to the working directory
COPY ./requirements.txt /workspace/requirements.txt

# Install dependencies from requirements.txt
RUN pip3 install -r /workspace/requirements.txt

# Install FinRL
RUN pip3 install finrl

# Install Jupyter Notebook
RUN pip3 install jupyter

# Install Avalanche
RUN pip3 install avalanche-lib
RUN pip3 install avalanche-lib[rl]

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Expose a port for Jyputer Notebook
EXPOSE 8888

# Define the default command
CMD ["bash"]
