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