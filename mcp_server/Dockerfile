FROM python:3.11-slim
WORKDIR /app

COPY mcp_requirements.txt .
RUN pip install --no-cache-dir -r mcp_requirements.txt

COPY run_mcp_server.py .
COPY data/synthetic_openflights.db ./data/

EXPOSE 8080
CMD ["python", "run_mcp_server.py"]