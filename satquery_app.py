"""
SatQuery AI: Interactive Multimodal Remote-Sensing AI Assistant
Problem Statement: SIH26167
Main Application Entrypoint
"""

import sys
import os

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def main():
    # 1. Resolve host and port from environment or CLI arguments
    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "8080"))
    except ValueError:
        port = 8080

    # Parse CLI flags or positional port argument
    args = sys.argv[1:]
    if "--port" in args:
        idx = args.index("--port")
        if idx + 1 < len(args):
            try:
                port = int(args[idx + 1])
            except ValueError:
                pass
    elif len(args) > 0 and not args[0].startswith("-"):
        try:
            port = int(args[0])
        except ValueError:
            pass

    if "--host" in args:
        idx = args.index("--host")
        if idx + 1 < len(args):
            host = args[idx + 1]

    log_level = os.getenv("LOG_LEVEL", "info").lower()

    print("==========================================================")
    print("           SATQUERY AI - REMOTE SENSING ASSISTANT          ")
    print("                     SIH Problem: SIH26167                ")
    print("==========================================================")
    print(f"Server binding : http://{host}:{port}")
    print(f"Health Endpoint: http://localhost:{port}/health")
    print(f"Swagger Docs   : http://localhost:{port}/docs")
    print(f"Frontend App   : http://localhost:{port}/")
    print("==========================================================")

    try:
        import uvicorn
        from satquery.server.fastapi_app import app
        uvicorn.run(app, host=host, port=port, log_level=log_level)
    except Exception as e:
        print(f"Uvicorn fallback due to: {e}. Starting standard multi-threaded HTTP server...")
        from satquery.server.app import run_server
        run_server(host=host, port=port)


if __name__ == "__main__":
    main()
