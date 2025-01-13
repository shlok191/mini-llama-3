# Use an NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Installing some needed system dependencies
RUN apt-get update && apt-get install -y python3-pip python3-venv curl

# Setting up a virtual environment for Python
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Installing some dependencies needed before pip install can be run
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

# Add Rust to the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install setuptools-rust torch uvicorn

# Copying over the SSL certificates
COPY fullchain1.pem /app/cert.pem
COPY privkey1.pem /app/key.pem

RUN chmod 644 /app/cert.pem
RUN chmod 600 /app/key.pem

COPY model /app/model
WORKDIR /app/model
RUN pip install -e .

# Copy the checkpoints directory
COPY checkpoints /app/checkpoints
COPY website/server /app/server

WORKDIR /app/server

# Making a port available
EXPOSE 2000

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Run uvicorn when the container launches
CMD ["uvicorn", "stream_generation:app", "--host", "0.0.0.0", "--port", "2000", \ 
"--ssl-keyfile", "/app/key.pem", "--ssl-certfile", "/app/cert.pem"]