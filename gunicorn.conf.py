"""Gunicorn config for Render / production."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5050')}"
workers = 1
threads = 4
timeout = 120
preload_app = False
worker_class = "gthread"
