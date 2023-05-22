"""
Preferences for the shell prints: colors, messaging and progress bars
"""

# imports

from datetime import datetime

# classes

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    ERROR = BOLD + RED
    INFO = BOLD + GREEN
    WARNING = BOLD + YELLOW

# functions

def msg_info(msg):
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    # print('(py) ' + f"{bcolors.INFO}INFO{bcolors.ENDC} " + time_stamp + msg, flush=True)
    print(f"{bcolors.INFO}INFO{bcolors.ENDC} " + time_stamp + msg, flush=True)


def msg_error(msg):
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    # print('(py) ' + f"{bcolors.ERROR}ERROR{bcolors.ENDC} " + time_stamp + msg, flush=True)
    print(f"{bcolors.ERROR}ERROR{bcolors.ENDC} " + time_stamp + msg, flush=True)


def msg_warning(msg):
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    # print('(py) ' + f"{bcolors.WARNING}WARNING{bcolors.ENDC} " + time_stamp + msg, flush=True)
    print(f"{bcolors.WARNING}WARNING{bcolors.ENDC} " + time_stamp + msg, flush=True)


def printProgressBar (iteration, total, prefix = '', suffix = 'Complete', decimals = 4, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    complete_prefix = f"{bcolors.OKCYAN}Progress {bcolors.ENDC}" + prefix
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{complete_prefix} |{bar}| {percent}% {suffix}', end = printEnd, flush="True")
    # Print New Line on Complete
    if iteration == total:
        print()

def get_version ():
    """
    Print version from git info
    Returns:
    __version__
    """
    from os.path import join, dirname, realpath
    __version__ = open(join(dirname(realpath(__file__)), '..', '..', '..', '.git',
                            'HEAD')).read()


    return __version__