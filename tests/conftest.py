import pytest
from models.wallet_engine import WalletEngine


@pytest.fixture(scope="session")
def engine():
    e = WalletEngine()
    e.load_data()
    return e
