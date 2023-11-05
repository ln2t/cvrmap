"""
pytest test_ file for io_tools.py module
"""


def test_arguments_manager():
    """
        Function to test arguments_manager()

    """
    from ..io_tools import arguments_manager
    args = arguments_manager('dummyversionnumber')
    assert isinstance(args, dict)