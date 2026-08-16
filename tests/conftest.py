"""
Pytest configuration for the Stelar Interior AI Voice Agent tests.

Sets up asyncio mode and common fixtures.
"""

import pytest


# Use asyncio mode for all async tests
@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
