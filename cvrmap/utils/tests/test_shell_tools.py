"""
pytest test_ file for shell_tools.py module
"""


def test_get_version():
    """
        Function to test get_version()

    """
    from ..shell_tools import get_version
    s = get_version()
    assert isinstance(s, str)