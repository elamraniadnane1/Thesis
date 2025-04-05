# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install Python and required build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy the Login.py file (and any other needed modules) into the container
COPY Login.py .

# Expose Streamlit's default port
EXPOSE 8501

# Ensure Streamlit runs in headless mode
ENV STREAMLIT_SERVER_HEADLESS=true

# Run the Login.py Streamlit application
CMD ["streamlit", "run", "Login.py"]
