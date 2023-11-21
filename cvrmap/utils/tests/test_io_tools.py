"""
pytest test_ file for io_tools.py module
"""
import os


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
        get_subjects_to_analyze(args, layout)
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
    assert e.type == SystemExit

def test_get_task():
    """
        Function to test get_task()

    """
    from ..io_tools import get_task
    from bids import BIDSLayout
    from argparse import Namespace
    import pytest
    layout = BIDSLayout('.', validate=False)
    args = Namespace(**{'task': 'some_task'})
    with pytest.raises(SystemExit) as e:
        get_task(args, layout)
    assert e.type == SystemExit


def test_get_custom_label():
    """
        Function to test get_custom_label()

    """
    from ..io_tools import get_custom_label
    from argparse import Namespace

    args = Namespace(**{'label': 'crazyname'})
    res = get_custom_label(args)
    assert res == '_label-crazyname'

    args = Namespace(**{'label': None})
    res = get_custom_label(args)
    assert res == ''


def test_get_space():
    """
        Function to test get_space()

    """
    from ..io_tools import get_space
    from bids import BIDSLayout
    from argparse import Namespace
    import pytest
    import os
    from shutil import rmtree
    from pathlib import Path

    # initiate dummy derivative folder

    test_root = '/tmp/tmp_pytest'

    if os.path.isdir(test_root):
        rmtree(test_root)

    Path(test_root).mkdir(exist_ok=True, parents=True)

    dataset_description = os.path.join(test_root,
                                       'dataset_description.json')
    with open(dataset_description, 'w') as ds_desc:
        ds_desc.write(('{"Name": "dummy_value", "BIDSVersion": "dummy_value", '
                       '"DatasetType": "derivative", "GeneratedBy": '
                       '[{"Name": "dummy_value"}, {"Version": "dummy_value"}]}'))
        ds_desc.close()

    args = Namespace(**{'space': 'random_spacename'})
    layout = BIDSLayout(test_root, validate=False)
    layout.add_derivatives('some_crazy_name')
    #assert layout.get_spaces(scope='derivatives') == []

    with pytest.raises(SystemExit) as e:
        get_space(args, layout)
    assert e.type == SystemExit


def test_set_flags():
    """
        Function to test set_flags()

    """
    from ..io_tools import set_flags
    from argparse import Namespace

    args = Namespace(**{'sloppy': 'value1', 'overwrite': 'value2',
                        'use_aroma': 'value3', 'vesselsignal': 'value4',
                        'globalsignal': 'value5'
                        })

    flags = set_flags(args)

    assert flags['sloppy'] == args.sloppy
    assert flags['overwrite'] == args.overwrite
    assert flags['ica_aroma'] == args.use_aroma
    assert flags['vesselsignal'] == args.vesselsignal
    assert flags['globalsignal'] == args.globalsignal


def test_setup_subject_output_paths():
    """
        Function to test setup_subject_output_paths()

    """
    from ..io_tools import setup_subject_output_paths
    from argparse import Namespace
    import os
    from shutil import rmtree

    args = Namespace(**{'sloppy': 'value1', 'overwrite': 'value2',
                        'use_aroma': 'value3', 'vesselsignal': 'value4',
                        'globalsignal': 'value5'
                        })
    output_dir = '/tmp/tmp_pytest'
    subject_label = 'dummylabel'
    space = 'dummyspace'
    res = None
    custom_label = 'dummycustomlabel'

    if os.path.isdir(output_dir):
        rmtree(output_dir)

    outputs = setup_subject_output_paths(output_dir, subject_label, space, res, args, custom_label)

    assert isinstance(outputs, dict)
    assert os.path.isdir(output_dir)

    keys_to_check = ['report', 'cvr', 'delay', 'denoised', 'etco2', 'vesselsignal', 'globalsignal', 'breathing_figure',
     'boldmean_figure', 'vesselsignal_figure', 'globalsignal_figure', 'cvr_figure', 'delay_figure', 'vesselmask_figure',
     'globalmask_figure', 'summary_reportlet', 'denoising_reportlet']

    assert keys_to_check == list(outputs.keys())

    if os.path.isdir(output_dir):
        rmtree(output_dir)

def test_get_physio_data():
    """
        Function to test get_physio_data()

    """
    from ..io_tools import get_physio_data
    from bids import BIDSLayout
    import numpy as np
    from pathlib import Path
    from shutil import rmtree

    test_root = '/tmp/tmp_pytest'

    if os.path.isdir(test_root):
        rmtree(test_root)


    bids_filter = dict()

    bids_filter['task'] = 'dummytask'
    bids_filter['subject'] = 'dummysub'
    bids_filter['space'] = 'dummyspace'

    data_dir = os.path.join(test_root, 'sub-' + bids_filter['subject'], 'func')

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    fn = 'sub-' + bids_filter['subject'] + '_task-' + bids_filter['task'] + '_physio.tsv.gz'
    fn_json = 'sub-' + bids_filter['subject'] + '_task-' + bids_filter['task'] + '_physio.json'
    fn_full = os.path.join(data_dir, fn)
    fn_json_full = os.path.join(data_dir, fn_json)

    data = np.random.random([10, 2])

    np.savetxt(fn_full, data)

    json_content = """
    {
        "SamplingFrequency": 100,
        "StartTime": 0,
        "Columns": [
            "co2"
        ],
        "co2": {
            "Units": "dummyunits"
        }
    }
    """

    with open(fn_json_full, 'w') as f:
        f.write(json_content)

    layout = BIDSLayout(test_root, validate=False)

    physio = get_physio_data(bids_filter, layout)

    assert physio.units == 'dummyunits'
    assert np.all(physio.data == data.T[1])


def test_get_aroma_noise_ic_list():
    """
        Function to test get_aroma_noise_ic_list()

    """
    from ..io_tools import get_aroma_noise_ic_list
    from bids import BIDSLayout
    from pathlib import Path
    from shutil import rmtree

    test_root = '/tmp/tmp_pytest'

    if os.path.isdir(test_root):
        rmtree(test_root)

    Path(test_root).mkdir(parents=True, exist_ok=True)

    layout = BIDSLayout(test_root, validate=False)

    bids_filter = dict()

    bids_filter['task'] = 'dummytask'
    bids_filter['subject'] = 'dummysub'
    bids_filter['space'] = 'dummyspace'

    dummy_ica_list = '1,2,3,42'

    data_dir = os.path.join(test_root, 'derivatives', 'fmriprep', 'sub-' + bids_filter['subject'], 'func')

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    fn = 'sub-' + bids_filter['subject'] + '_task-' + bids_filter['task'] + '_physio.tsv.gz'
    fn_full = os.path.join(data_dir, fn)

    with open(fn_full, 'w') as f:
        f.write(dummy_ica_list)

    # get_aroma_noise_ic_list(bids_filter, layout)

    # work in progress



