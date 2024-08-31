# Use the official TensorFlow Docker image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

# Install necessary packages
RUN apt-get update && \
    apt-get install -y \
    curl \
    screen
#    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio

# Copy requirements.txt to the working directory
COPY ./requirements.txt /workspace/requirements.txt

# Install dependencies from requirements.txt
RUN pip install -r /workspace/requirements.txt

# Install FinRL
RUN pip3 install finrl

# Install Jupyter Notebook
RUN pip install jupyter

# Install Avalanche
RUN pip install avalanche-lib
RUN pip install avalanche-lib[rl]

# Clean up
RUN apt-get clean

# Set the working directory
WORKDIR /workspace

# Expose a port for Jyputer Notebook
EXPOSE 8888

# Define the default command
CMD ["bash"]
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
