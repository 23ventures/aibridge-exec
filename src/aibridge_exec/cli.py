"""``aibridge-exec`` command line."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aibridge-exec")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the HTTP server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1)
    serve.add_argument("--log-level", default="info")
    sub.add_parser("providers", help="list built-in providers")
    args = parser.parse_args(argv)

    if args.command == "providers":
        from .providers import list_providers

        for provider in list_providers():
            print(f"{provider.id:20} {provider.info.label}")
        return 0

    try:
        import uvicorn
    except ImportError:
        print('The server needs extra packages: pip install "aibridge-exec[server]"', file=sys.stderr)
        return 1
    uvicorn.run("aibridge_exec.server.asgi:app", host=args.host, port=args.port, workers=args.workers,
                log_level=args.log_level, proxy_headers=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
