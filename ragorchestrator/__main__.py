"""Entry point for python -m ragorchestrator."""

import uvicorn

from ragorchestrator.app import PORT

if __name__ == "__main__":
    uvicorn.run(
        "ragorchestrator.app:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
