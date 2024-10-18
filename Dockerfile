FROM python:3.9-slim
WORKDIR /app
COPY . .
CMD ["sh", "-c", "\
    if [[ \"$(uname -s)\" == \"Darwin\" ]]; then \
        echo 'Installing Mac dependencies'; \
        pip install -r requirements-mac.txt; \
    elif [[ \"$(uname -s)\" == *MINGW* ]]; then \
        echo 'Installing Windows dependencies'; \
        pip install -r requirements-windows.txt; \
    else \
        echo 'Unsupported OS'; \
        exit 1; \
    fi && \
    python app.py"]

