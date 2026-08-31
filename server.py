"""
OURIONSPECTRA — FastAPI Server Entry Point.
Run with: python server.py [--host 0.0.0.0] [--port 8000] [--reload]
"""

import argparse
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the OurionSpectra FastAPI Backend Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")

    args = parser.parse_args()

    print(f"Starting OurionSpectra API on http://{args.host}:{args.port}")
    print(f"Interactive Swagger documentation at: http://{args.host}:{args.port}/docs")
    print(f"ReDoc alternative documentation at: http://{args.host}:{args.port}/redoc")

    uvicorn.run("ourionspectra.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
