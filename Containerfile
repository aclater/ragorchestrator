# ragorchestrator — LangGraph supervisor agent for ragpipe
# UBI10 minimal, non-root, healthcheck defined

FROM registry.access.redhat.com/ubi10/python-312:latest

LABEL maintainer="aclater" \
      description="LangGraph supervisor agent for agentic RAG orchestration" \
      io.k8s.description="ragorchestrator — agentic RAG via LangGraph" \
      io.k8s.display-name="ragorchestrator"

WORKDIR /opt/app-root/src

COPY pyproject.toml .
RUN pip install --no-cache-dir -e '.[dev]' && \
    pip install --no-cache-dir .

COPY ragorchestrator/ ragorchestrator/

USER 1001

EXPOSE 8095

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8095/health')" || exit 1

CMD ["python", "-m", "ragorchestrator"]
