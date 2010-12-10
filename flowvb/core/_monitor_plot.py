import wx
import math
from math import pi
import numpy as np
from scipy.io import loadmat
from matplotlib.patches import Ellipse
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import \
     FigureCanvasWxAgg as FigureCanvas

TIMER_ID = wx.NewId()


class _MonitorPlot(wx.Frame):
    """Class to provide live updated plotting to monitor the convergence of FlowVB
    """

    def __init__(self, data, scale=1):
        """Setup plotting environment
        """

        self.scale = scale
        self.data = data

        # Setup plotting window
        wx.Frame.__init__(self, None, wx.ID_ANY,
                          title="FlowVB Progress Monitor", size=(800, 600))
        self.fig = Figure((8, 6), 100)
        self.canvas = FigureCanvas(self, wx.ID_ANY, self.fig)
        self.ax = self.fig.add_subplot(111)

        # Set axis limits
        x_lims = [data[:, 0].min(), data[:, 0].max()]
        y_lims = [data[:, 1].min(), data[:, 1].max()]

        self.ax.set_xlim(x_lims)
        self.ax.set_ylim(y_lims)
        self.ax.set_autoscale_on(False)

        # Draw to screen
        self.canvas.draw()
        # Save background
        self.bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def update_plot(self, pos, cov):
        """Update the plot
        """

        # Restore empty plot (but doesn't seem to do anything)
        self.canvas.restore_region(self.bg)

        # Remove previous elements from plot
        self.ax.lines = []
        self.ax.patches = []

        # Draw the data
        self.l_data = self.ax.plot(self.data[:, 0], self.data[:, 1],
                                   color='blue', linestyle='', marker='o')

        K = pos.shape[0]
        for k in range(K):
            # Draw centers
            l_center, = self.ax.plot(pos[k, 0], pos[k, 1],
                                     color='red', marker='+')

            # Compute and draw error ellipses
            U, s, Vh = np.linalg.svd(cov[k, :, :])
            orient = math.atan2(U[1, 0], U[0, 0]) * 180 / pi
            ellipsePlot = Ellipse(xy=pos[k, :], width=2.0 * math.sqrt(s[0]),
                                  height=2.0 * math.sqrt(s[1]),
                                  angle=orient, facecolor='none',
                                  edgecolor='red')

            self.ax.add_patch(ellipsePlot)

        # Draw to screen
        self.canvas.draw()
        self.canvas.blit(self.ax.bbox)

if __name__ == "__main__":
    data = loadmat("tests/data/old_faithful/faithful.mat")['data']
    app = wx.App(False)
    frame = _MonitorPlot(data)
    frame.Show(True)
    app.MainLoop()
