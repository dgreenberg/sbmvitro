import os
import numpy as np
import ipywidgets as widgets
from PyQt5 import QtGui
from IPython.display import display, clear_output, Javascript
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure as figtype

#  %gui qt4 should be called from the ipynb before using these

def ylabelh(text):
    plt.ylabel(text, rotation = 0, ha = 'right')

class dirchooser:
    path = ''  # directory

    def on_button_clicked(self, b):
        din = QtGui.QFileDialog.getExistingDirectory(
            caption = 'Choose data location',
            directory = self.path
            )

        if din == '' or (not os.path.isdir(din)):  # second check shouldn't be necessary?
            return
        self.path = din
        clear_output() # clear previous text output from this cell
        print ("data directory: " + self.path)

    def __init__(
        self,
        din='',
        buttontext = 'Choose directory',
        dialogtext = 'Choose directory',
        statustext = 'Currently chosen directory:'
        ):

        if os.path.isdir(din):
            self.path = din
        button = widgets.Button(
            description = buttontext,
            background_color = '#00FF00',
            border_color = '#000000',
            border_width = '2pt'
            )
        display(button)
        button.on_click(self.on_button_clicked)
        print (statustext + self.path)

def clipboard2array():
    '''
    Return the clipboard as a numpy array using pandas.
    Useful for excel data.
    '''

    data = pd.read_clipboard(header = None)
    y = np.array(data.ix[:])
    return y

jscommand_disableAS = """
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
"""

def disable_autoscroll():
    display(Javascript(jscommand_disableAS))
    return


def savefigs_pdf(h, figfilename = 'savefigs_pdf.pdf'):
    if type(h) == figtype:
        savefigs_pdf([h], figfilename)
        return
    if type(h) != list and type(h) != tuple:
        raise TypeError("h must be a matplotlib.figure.Figure or a tuple/list of them")
    pp = PdfPages(figfilename)
    try:
        for i in range(0, len(h)):
            pp.savefig(h[i], bbox_inches='tight')
    finally:
        pp.close()
