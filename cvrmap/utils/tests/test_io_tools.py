"""
pytest test_ file for io_tools.py module
"""

def test_arguments_manager():
    """
        Function to test arguments_manager()

    """
    from ..io_tools import arguments_manager
    import sys
    from unittest.mock import patch

    with patch.object(sys, 'argv', ['/mnt/erasme/git/ln2t/cvrmap/cvrmap/cvrmap.py', 'some_bids_dir', 'some_derivatives_dir', 'participant']):
        args = arguments_manager('dummyversionnumber')

        assert args.bids_dir == 'some_bids_dir'
        assert args.output_dir == 'some_derivatives_dir'
        assert args.analysis_level == 'participant'

def test_get_subjects_to_analyze():
    """
        Function to test get_subjects_to_analyze()

    """
    from ..io_tools import get_subjects_to_analyze
    from bids import BIDSLayout
    from argparse import Namespace
    import pytest
    layout = BIDSLayout('.', validate=False)
    args = Namespace(**{'participant_label': '007'})
    with pytest.raises(SystemExit) as e:
        subjects = get_subjects_to_analyze(args, layout)
    assert e.type == SystemExit

def test_get_fmriprep_dir():
    """
        Function to test get_fmriprep_dir()

    """
    from ..io_tools import get_fmriprep_dir
    from argparse import Namespace
    import pytest

    args = Namespace(**{'fmriprep_dir': '.'})
    assert get_fmriprep_dir(args) == '.'

    args = Namespace(**{'fmriprep_dir': 'crazy_place_42'})
    with pytest.raises(SystemExit) as e:
        get_fmriprep_dir(args)
    assert  e.type == SystemExit