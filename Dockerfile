# Use the NVIDIA PyTorch image as the base
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set the working directory
WORKDIR /workdir

# Copy your files into the image
COPY . /workdir

# Update package lists
RUN apt-get update

# Install dependencies
RUN apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1-mesa-dev \
    apt-utils \
    vim \
    python3-pycuda

# Clean up the package lists to reduce image size
RUN rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python packages
RUN pip install opencv-python scikit-learn seaborn pandas

## NEED TO UPDATE PYCUDA

# Clone YOLOv5 and install its dependencies
RUN git clone https://github.com/ultralytics/yolov5.git /Tensorflow_Vision/Tensorflow_Vision/Tensorflow_Yolov5/yolov5 && \
    pip install -r /Tensorflow_Vision/Tensorflow_Vision/Tensorflow_Yolov5/yolov5/requirements.txt

RUN echo "Build completed successfully"
