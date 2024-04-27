import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
from matplotlib.animation import FuncAnimation
import glob
import os
from distutils.spawn import find_executable


class Animator:
    def __init__(self, col, original, dir="."):
        """
        Constructor
        ===========
        col: column to be read in data files
        dir: directory where data files are located
        original: using files from original or reformulated evolution
        """
        #
        # set up TeX etc.
        #
        #        font = {'family' : 'DejaVu Sans',
        #                'weight' : 'normal',
        #                'size' : 14 }
        #        rc('font', **font)
        if find_executable('latex'):
            rc('text', usetex=True)
        #        
        # create list of files and sort according to time
        #
        if (original):
            files = dir + "/" + "*" + "_orig_" + "*" + "_fields_" + "*"
        else:
            files = dir + "/" + "*" + "_reform_" + "*" + "_fields_" + "*"
        print(" Looking for files" , files)
        self.list_of_files = glob.glob(files)
        self.list_of_files.sort(key=self.get_times)
        self.number_of_files = int(len(self.list_of_files))
        if (self.number_of_files == 0):
            print(" Found no data files in directory",dir)
            sys.exit(1)
        print(" Found", self.number_of_files, "data files")
        #
        # set up figure
        #
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.column = col
        factor = 1.05   # factor for limits
        self.x_max, self.y_max, self.f_max = self.get_limits(factor, col)
        if (self.column == 2):
            print(" Plotting E^x")
            if find_executable('latex'):
                self.zlabel = r"$\bar E^x$"
            else:
                self.zlabel = "Ex"
        elif (self.column == 3):
            print(" Plotting E^y")
            if find_executable('latex'):
                self.zlabel = r"$\bar E^y$"
            else:
                self.zlabel = "Ey"
        elif (self.column == 4):
            print(" Plotting A^x")
            if find_executable('latex'):
                self.zlabel = r"$\bar A^x$"
            else:
                self.zlabel = "Ax"
        elif (self.column == 5):
            print(" Plotting A^y")
            if find_executable('latex'):
                self.zlabel = r"$\bar A^y$"
            else:
                self.zlabel = "Ay"
        elif (self.column == 7):
            print(" Plotting constraint violations")
            if find_executable('latex'):
                self.zlabel = r"$\bar {\mathcal C}$"
            else:
                self.zlabel = "C"
        else:
            print(" Unknown column number", self.column)

    def get_times(self, file):
        f = open(file)
        if f:
            line = f.readline()
            words = line.split()
            time = float(words[4])
            f.close()
            return time
        else:
            print(" Could not open file", file, "in get_times()")
            sys.exit(2)


    def get_limits(self, factor, col=2, file_number=0):
        """finds limits for plotting from file number file_number"""
        file = self.list_of_files[file_number]
        print(" Finding limits from file", file)
        f = open(file)
        if f:
            x, y, fct = np.loadtxt(file, unpack=True, 
                                   usecols=(0,1,col,))
        else:
            print(" Could not open file", file, "in get_limits()")
            sys.exit(2)
        f.close()
        x_max = factor * np.max(x)
        y_max = factor * np.max(y)
        f_min = factor * np.min(fct)
        f_max = factor * np.max(fct)
        f_max = max(abs(f_min), abs(f_max))
        if (f_max == 0.0):
            print(" Found f_max = 0 -- will grab maxima from E_x...")
            x_max, y_max, f_max = self.get_limits(factor, 2, file_number)
        return x_max, y_max, f_max


    def snapshot(self, file_number):
        self.ax.clear()
        file = self.list_of_files[file_number]
        f = open(file)
        if f:
            x, y, fct = np.loadtxt(file, unpack=True, 
                                   usecols=(0,1,self.column,))
            line = f.readline()
            words = line.split()
            time = float(words[4])
        else:
            print(" Could not open file", file,"in snapshop()")
            sys.exit(2)
        f.close()
        if find_executable('latex'):
            title = r"$\bar t = {0:.2f}$".format(time)
        else:
            title = "t = {0:.2f}".format(time)
        n_grid = int(np.sqrt(x.size))
        X = np.reshape(x, (n_grid, n_grid))
        Y = np.reshape(y, (n_grid, n_grid))
        FCT = np.reshape(fct, (n_grid, n_grid))
        self.ax.set_title(title)
        self.ax.set_xlim(0, self.x_max)
        self.ax.set_ylim(0 ,self.y_max)
        self.ax.set_zlim(-self.f_max,self.f_max)
        if find_executable('latex'):
            self.ax.set_xlabel(r"$\bar x$", size=14)
            self.ax.set_ylabel(r"$\bar y$", size=14)
        else:
            self.ax.set_xlabel("x", size=14)
            self.ax.set_ylabel("y", size=14)
        self.ax.set_zlabel(self.zlabel, size=14)
        self.ax.xaxis.set_rotate_label(False)
        self.ax.yaxis.set_rotate_label(False)
        self.ax.zaxis.set_rotate_label(False)
        mesh = self.ax.plot_surface(X, Y, FCT, cmap=cm.autumn, 
                                    alpha=0.7)
        # mesh.set_clim(-self.f_max,self.f_max)

    def animate(self, inter):
        anim = FuncAnimation(self.fig, self.snapshot, 
                             frames=np.arange(0,self.number_of_files),
                             init_func=None, blit=False,
                             repeat=False, interval=inter)
        plt.show()

    def make_still(self, plot_time, figure_file = 0):
        #
        # find file corresponding to time
        #
        time = - 1.0
        i = 0
        while i < self.number_of_files and time < plot_time:
            time = self.get_times(self.list_of_files[i])
            i += 1
        if i == self.number_of_files:
            print(" Could not find data for time", plot_time)
            sys.exit(3)
        else:
            i -= 1
            print(" Plotting data in file", self.list_of_files[i])
            self.snapshot(i)
            if figure_file == 0:
                plt.show()
            else:
                plt.savefig(figure_file, psi=300)

def main():
    """Main routine..."""
    print(" ----------------------------------------------------------------")
    print(" --- maxwell_animation.py --- use flag -h for list of options ---")
    print(" ----------------------------------------------------------------")
    #
    # set default values for variables
    #
    # column for data field in data files
    column = 2
    # directory in which data files are located
    dir = "."
    # time interval between screens in animation (in units of milliseconds)
    inter = 200
    # create single plot rather than animation if still_time >=0
    still_time = -1
    # filename if stills are supposed to be saved
    still_file = 0
    # make plots for original or reformulated equations
    original =  True
    # now look for flags to overwrite default values
    #
    for i in range(len(sys.argv)):
        if sys.argv[i] == "-h":
            usage()
            return
        if sys.argv[i] == "-field":
            if sys.argv[i+1] in ["E_x", "e_x", "Ex", "ex"]:
                column = 2
            elif sys.argv[i+1] in ["E_y", "e_y", "Ey", "ey"]:
                column = 3
            elif sys.argv[i+1] in ["A_x", "a_x", "Ax", "ax"]:
                column = 4
            elif sys.argv[i+1] in ["A_y", "a_y", "Ay", "ay"]:
                column = 5
            elif sys.argv[i+1] in ["C", "c", "constraint", "const", "Constraint"]:
                column = 7
        if sys.argv[i] == "-reform":
            original = False
        if sys.argv[i] == "-inter":
            inter = float(sys.argv[i+1])
        if sys.argv[i] == "-dir":
            dir = sys.argv[i+1]
        if sys.argv[i] == "-still":
            still_time = float(sys.argv[i+1])
        if sys.argv[i] == "-file":
            still_file = sys.argv[i+1]
    #
    # sanity check...
    #
    if still_time < 0.0 and still_file != 0:
        print("Cannot produce file for animation...")
        usage()
        return
    #
    # create animator class
    #
    ani = Animator(column,original,dir)
    #
    # create animation...
    #
    if still_time < 0.0:
        ani.animate(inter)
    #
    # or make still
    #
    else:
        ani.make_still(still_time, still_file)


def usage():
    print("Creates animation from data files produced with maxwell.py.")
    print("")
    print("The following options can be used to over-write default parameters")
    print("\t-field: field to be plotted [Default: E_x] ")
    print("\t-reform: plots for reformulated integration rather than original ")
    print("\t-inter: time interval between screens in animation [200 ms] ")
    print("\t-dir: directory in which files are located [.]")
    print("\t-still: produce 'still' at specified time, rather than animation")
    print("\t-file: save 'still' in file with specified name")
    print("For example, to produce a still of E_y at time t = 3.0, call")
    print("\tpython3 maxwell_animate.py -still 3.0 -field E_y")

if __name__ == '__main__':
    main()
