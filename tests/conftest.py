import pytest

from credit_risk.config import get_paths, load_config


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def paths():
    return get_paths()


def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" in item.nodeid.split("::")[0]:
            item.add_marker(pytest.mark.integration)
