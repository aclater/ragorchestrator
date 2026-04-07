"""Shared pytest configuration and fixtures."""


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False, help="Run live integration tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        skip_live = __import__("pytest").mark.skip(reason="need --live option to run")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
