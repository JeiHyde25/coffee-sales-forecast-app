# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.13.3
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser


# Copy requirements into the image
COPY requirements.txt .

# Install dependencies using copied file
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the source code into the container.
COPY . .

# Create writable directories and give permissions to appuser (UID 10001)
RUN mkdir -p /tmp/.streamlit /tmp/matplotlib /tmp/kagglehub && \
    chmod -R 777 /tmp/.streamlit /tmp/matplotlib /tmp/kagglehub && \
    chmod -R 777 /app

# Set environment variables to redirect cache/config writes
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV KAGGLEHUB_CACHE_DIR=/tmp/kagglehub
ENV HOME=/tmp

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8080

# Run the application.
CMD streamlit run src/notebook.py --server.port=8080 --server.address=0.0.0.0
