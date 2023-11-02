"""
pytest test_ file for math_tools.py module
"""

def test_match_timecourses():
    """
        Test function for math_timecourses. Generate random data.
    """
    import numpy as np
    from ..math_tools import match_timecourses
    n1 = np.random.randint(100, 150)
    n2 = np.random.randint(100, 150)
    y1 = np.random.random(n1)
    y2 = np.random.random(n2)
    delay = np.random.randint(-30,30)
    y1_m, y2_m = match_timecourses(y1, y2, delay)
    assert len(y1_m) == len(y2_m)


def test_tccorr():
    """
        Test function for tccorr.
    """

    import numpy as np
    from ..math_tools import tccorr
    from ..processing import DataObj

    eps = 1E-3  # a small number to check strong (anti-)correlations

    # test 1: same sampling frequency, strong anti-correlation

    data1 = np.arange(100)
    data2 = -np.arange(100)
    do1 = DataObj(data=data1, sampling_frequency=1)
    do2 = DataObj(data=data2, sampling_frequency=1)

    assert np.abs(tccorr(do1, do2) + 1.) < eps

    # test 2: different sampling frequencies, strong correlation

    data1 = np.arange(100)
    data2 = np.arange(200)
    do1 = DataObj(data=data1, sampling_frequency=1)
    do2 = DataObj(data=data2, sampling_frequency=0.8)

    assert np.abs(tccorr(do1, do2) - 1.) < eps

    # test 3: different sampling frequencies (other way around), strong correlation

    data1 = np.arange(180)
    data2 = np.arange(100)
    do1 = DataObj(data=data1, sampling_frequency=1.3)
    do2 = DataObj(data=data2, sampling_frequency=1)

    assert np.abs(tccorr(do1, do2) - 1.) < eps


def test_build_shifted_signal():
    """
        Function to test build_shifted_signal
    """

    from ..math_tools import build_shifted_signal
    import numpy as np
    from ..processing import DataObj

    # test 1

    data1 = np.arange(180)
    data2 = np.arange(100)
    probe = DataObj(data=data1, sampling_frequency=1.3, data_type='timecourse')
    target = DataObj(data=data2, sampling_frequency=1.1, data_type='timecourse')

    delta_t = 5  # some random value in seconds

    shifted_signal = build_shifted_signal(probe, target, delta_t)

    assert shifted_signal.sampling_frequency == target.sampling_frequency
    assert shifted_signal.data.shape[0] == data2.shape[0]

    # test 2

    data1 = np.arange(100)
    data2 = np.arange(100)
    probe = DataObj(data=data1, sampling_frequency=1, data_type='timecourse')
    target = DataObj(data=data2, sampling_frequency=1.5, data_type='timecourse')

    delta_t = -4  # some random value in seconds

    shifted_signal = build_shifted_signal(probe, target, delta_t)

    assert shifted_signal.sampling_frequency == target.sampling_frequency
    assert shifted_signal.data.shape[0] == data2.shape[0]


def test_compute_global_signal():
    """
        Test function for compute_global_signal
    """

    from ..math_tools import compute_global_signal
    import numpy as np
    from ..processing import DataObj

    data = np.random.random([130, 50, 110])
    fake_fmri_data = DataObj(data=data, sampling_frequency=3, data_type='bold')

    gs = compute_global_signal(fake_fmri_data)

    assert gs.data_type == 'timecourse'
    assert gs.sampling_frequency == fake_fmri_data.sampling_frequency
    assert np.mean(gs.data) == 1

    fake_fmri_data_double = DataObj(data=2*data, sampling_frequency=3, data_type='bold')
    gs_double = compute_global_signal(fake_fmri_data_double)

    assert np.mean(gs_double.data) == 1


def test_get_corrected_noiselist():
    """
        Test function for get_corrected_noiselist
    """

    from ..math_tools import get_corrected_noiselist
    from ..processing import DataObj
    import numpy as np
    import pandas as pd

    # test 1

    n_vol = 200  # seconds
    probe_sf = 50
    probe = DataObj(data=np.random.random(6666), data_type='timecourse', sampling_frequency=probe_sf)
    aroma_noise_ic_list = [1, 2, 3]
    melodic_mixing_df = pd.DataFrame()
    melodic_mixing_df['1'] = np.random.random(n_vol)
    melodic_mixing_df['2'] = np.random.random(n_vol)
    melodic_mixing_df['3'] = np.random.random(n_vol)
    melodic_mixing_df['4'] = np.random.random(n_vol)
    melodic_mixing_df['5'] = np.random.random(n_vol)
    melodic_mixing_df['6'] = np.random.random(n_vol)
    sf = 3  # BOLD sampling frequency
    noise_ic_pearson_r_threshold = 0
    aroma_flag = False

    corr_list = get_corrected_noiselist(probe, aroma_noise_ic_list, melodic_mixing_df, sf, noise_ic_pearson_r_threshold, aroma_flag)

    assert len(corr_list) == 0

    # test 2

    noise_ic_pearson_r_threshold = 0.99999
    aroma_flag = False

    corr_list = get_corrected_noiselist(probe, aroma_noise_ic_list, melodic_mixing_df, sf, noise_ic_pearson_r_threshold,
                                        aroma_flag)

    assert len(corr_list) == len(aroma_noise_ic_list)

    # test 3

    noise_ic_pearson_r_threshold = 0.6
    aroma_flag = True

    corr_list = get_corrected_noiselist(probe, aroma_noise_ic_list, melodic_mixing_df, sf, noise_ic_pearson_r_threshold,
                                        aroma_flag)

    assert np.all(np.array(corr_list) + 1 == aroma_noise_ic_list)