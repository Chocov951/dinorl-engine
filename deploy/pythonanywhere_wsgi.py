"""WSGI entry point imported by PythonAnywhere."""

from dinorl_engine.service.api import create_app

application = create_app()
