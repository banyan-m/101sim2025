FROM mcr.microsoft.com/devcontainers/python:0-3.10

# MuJoCo runtime libs (CPU-only headless OpenGL)
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglew-dev patchelf curl && \
    rm -rf /var/lib/apt/lists/*

# Set up MuJoCo license first
COPY setup-mujoco.sh /tmp/setup-mujoco.sh
RUN chmod +x /tmp/setup-mujoco.sh && /tmp/setup-mujoco.sh

# Now install Python packages after license is in place
COPY requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# Install OSMesa for headless rendering and FFmpeg for video encoding
RUN apt-get update && \
    apt-get install -y libosmesa6 libosmesa6-dev libgl1-mesa-glx ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set MuJoCo to use OSMesa backend by default
ENV MUJOCO_GL=osmesa 