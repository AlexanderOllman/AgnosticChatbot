FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Ensure any Python dependencies in requirements.txt are installed
# RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install -r requirements.txt

# Install necessary packages: Git, Git LFS, and curl for installing kubectl
RUN apt-get update && \
    apt-get install -y curl nano git git-lfs ca-certificates && \
    git lfs install && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm kubectl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Make port 80 and 8080 available to the world outside this container
EXPOSE 80
EXPOSE 8080

# Run app.py when the container launches
CMD ["python3", "app.py"]
