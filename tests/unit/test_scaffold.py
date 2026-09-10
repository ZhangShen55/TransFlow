from fastapi import FastAPI

from transflow.main import create_app


def test_application_factory_starts() -> None:
    app = create_app()

    assert isinstance(app, FastAPI)
    assert app.title == "TransFlow"
