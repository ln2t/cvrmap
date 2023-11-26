"""
Preferences for the shell prints: colors, messaging and progress bars
"""

# classes

class bcolors:
    """
        Convenient colors for terminal prints

    """
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
    """
        Custom print to shell (information)

    Args:
        msg: str

    Returns:
        None
    """
    from datetime import datetime
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    # print('(py) ' + f"{bcolors.INFO}INFO{bcolors.ENDC} " + time_stamp + msg, flush=True)
    print(f"{bcolors.INFO}INFO{bcolors.ENDC} " + time_stamp + msg, flush=True)


def msg_error(msg):
    """
        Custom print to shell (error)

    Args:
        msg: str

    Returns:
        None
        """
    from datetime import datetime
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    # print('(py) ' + f"{bcolors.ERROR}ERROR{bcolors.ENDC} " + time_stamp + msg, flush=True)
    print(f"{bcolors.ERROR}ERROR{bcolors.ENDC} " + time_stamp + msg, flush=True)


def msg_warning(msg):
    """
        Custom print to shell (warning)

    Args:
        msg: str

    Returns:
        None
        """
    from datetime import datetime
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    # print('(py) ' + f"{bcolors.WARNING}WARNING{bcolors.ENDC} " + time_stamp + msg, flush=True)
    print(f"{bcolors.WARNING}WARNING{bcolors.ENDC} " + time_stamp + msg, flush=True)


def run(command, env={}):
    """
        Execute command as in a terminal
        Also prints any output of the command into the python shell

    Args:
        command: string the command (including arguments and options) to be executed
        env: to add stuff in the environment before running the command

    Returns:
        nothing
    """

    import os
    import subprocess  # to call stuff outside of python

    # Update env
    merged_env = os.environ
    merged_env.update(env)

    # Run command
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True,
                               env=merged_env)

    # Read whatever is printed by the command and print it into the
    # python console
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() != None:
            break
    if process.returncode != 0:
        raise Exception("Non zero return code: %d" % process.returncode)


def printProgressBar (iteration, total, prefix='', suffix='Complete', decimals=4, length=100, fill='█', print_end="\r"):
    """
        Call in a loop to create terminal progress bar

    Args:
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
    print(f'\r{complete_prefix} |{bar}| {percent}% {suffix}', end=print_end, flush="True")
    # Print New Line on Complete
    if iteration == total:
        print()


def get_version():
    """
        Print version from package

    Returns:
        str
    """

    import importlib.metadata

    version = importlib.metadata.version('cvrmap')

    return version