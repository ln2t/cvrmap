# cvrmap package

__version__ = "4.3.1"

# Set matplotlib backend to non-GUI to avoid tkinter issues in parallel processing
import os
import matplotlib
matplotlib.use('Agg')

# Set environment variables to prevent GUI backend issues
os.environ['MPLBACKEND'] = 'Agg'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'