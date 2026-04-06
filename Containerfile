# ragorchestrator — LangGraph supervisor agent for ragpipe
# UBI9 Python 3.12, non-root, healthcheck defined
# UBI10 does not yet publish python-312; using UBI9 until available

FROM registry.access.redhat.com/ubi9/python-312@sha256:7ba356eca7f476bcf9a8c51714e43353376d37e0bbd4e43ceec7b1bcc6ff9675

LABEL maintainer="aclater" \
      description="LangGraph supervisor agent for agentic RAG orchestration" \
      io.k8s.description="ragorchestrator — agentic RAG via LangGraph" \
      io.k8s.display-name="ragorchestrator"

WORKDIR /opt/app-root/src

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY ragorchestrator/ ragorchestrator/

USER 1001

EXPOSE 8095

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8095/health')" || exit 1

CMD ["python", "-m", "ragorchestrator"]
