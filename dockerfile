# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set noninteractive installation mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update package repository and install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy the Streamlit app and any required modules into the container
COPY dashboard_admin.py .
COPY Login.py .

# Expose Streamlit's default port
EXPOSE 8501

# Set Streamlit to run in headless mode
ENV STREAMLIT_SERVER_HEADLESS=true

# Start the Streamlit app when the container launches
CMD ["streamlit", "run", "dashboard_admin.py"]
