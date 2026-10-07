"""Optional HTTP server (``pip install "aibridge-exec[server]"``; run ``aibridge-exec serve``)."""
from .app import create_app
from .settings import ServerSettings

__all__ = ["ServerSettings", "create_app"]
