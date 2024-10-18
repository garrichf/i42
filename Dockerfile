FROM python:3.9-slim
WORKDIR /app
COPY . .
ARG PLATFORM
RUN if [ "$PLATFORM" = "mac" ]; then \
        pip install -r requirements-mac.txt; \
    elif [ "$PLATFORM" = "windows" ]; then \
        pip install -r requirements-window.txt; \
    fi
CMD ["python", "app.py"]

