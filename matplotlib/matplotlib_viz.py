import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Polygon

import numpy as np
from scipy.interpolate import griddata

# Global variables
fig, ax = plt.subplots(figsize=(30, 30))

# Lire depuis un fichier
def charger_objet(obj, limit=0):
    coords = []
    values = []
    count = 0

    file = open(obj, "r")
    angle = 0.
    x, y = 0., 0.
    u, v = 0., 0.

    R = np.zeros((2, 2))

    for line in file.readlines():
        data = line.split()

        if count == 0:
            angle = float(data[2])

            theta = np.radians(angle)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))

            count += 1
            continue

        x, y, u, v = float(data[0]), float(
            data[1]), float(data[2]), float(data[3])

        # Rotation matrix
        coords.append(np.matmul(R, np.array((x, y))))
        values.append(np.matmul(R, np.array((u, v))))

        if limit != 0 and count == limit:
            break
        count += 1

    return np.array(coords), np.array(values)


"""Class: Fentre contenant les coords, valeurs, cellules"""


class WindowsViz:

    def __init__(self):
        self.__coords, self.__values = charger_objet("aile")
        self.__xmin, self.__xmax = min(
            self.__coords[:, 0]), max(self.__coords[:, 0])
        self.__ymin, self.__ymax = min(
            self.__coords[:, 1]), max(self.__coords[:, 1])

        self.__xs, self.__ys, self.__us, self.__vs, self.__speeds = self.recalibrate()

        self.__dispVec, self.__dispGrid, self.__dispCellule = False, False, False
        self.__dropPoints = []
        self.__cells = self.calcCells()

        self.__strm = None
        self.__beginSection = None
        self.__endSection = None

        print("WindowsViz created successfully!")

    def recalibrate(self):
        x, y = self.__coords[:, 0], self.__coords[:, 1]
        u, v = self.__values[:, 0], self.__values[:, 1]

        nx, ny = len(x), len(y)
        xr = np.linspace(x.min(), x.max(), nx)
        yr = np.linspace(y.min(), y.max(), ny)

        xs, ys = np.meshgrid(xr, yr)

        px, py, pu, pv = x.flatten(), y.flatten(), u.flatten(), v.flatten()

        us = griddata((px, py), pu, (xs, ys))
        vs = griddata((px, py), pv, (xs, ys))

        speeds = np.sqrt(us*us + vs*vs)

        return xs, ys, us, vs, speeds

    """ Function: calculter les cellules """

    def calcCells(self):
        cells = []

        tabX, tabY = np.array(self.__coords[:, 0]), np.array(
            self.__coords[:, 1])
        tabU, tabV = np.array(self.__values[:, 0]), np.array(
            self.__values[:, 1])

        for n in range(34):

            # On travaille avec deux lignes horizontales à la fois
            xs1, ys1, us1, vs1 = tabX[n:len(tabX):35], tabY[n:len(
                tabY):35], tabU[n:len(tabU):35], tabV[n:len(tabV):35]
            xs2, ys2, us2, vs2 = tabX[n+1:len(tabX):35], tabY[n+1:len(
                tabY):35], tabU[n+1:len(tabU):35], tabV[n+1:len(tabV):35]

            for i in range(0, 87):
                x11, x12, x21, x22 = xs1[i], xs1[i+1], xs2[i], xs2[i+1]
                y11, y12, y21, y22 = ys1[i], ys1[i+1], ys2[i], ys2[i+1]
                u11, u12, u21, u22 = us1[i], us1[i+1], us2[i], us2[i+1]
                v11, v12, v21, v22 = vs1[i], vs1[i+1], vs2[i], vs2[i+1]

                cells.append((
                    (x11, y11, u11, v11),
                    (x12, y12, u12, v12),
                    (x22, y22, u22, v22),
                    (x21, y21, u21, v21)
                ))
        return np.array(cells)

    # Afficher le champs de vecteurs
    def afficher_vitesse(self):
        x, y, u, v = self.__coords[:, 0], self.__coords[:,
                                                        1], self.__values[:, 0], self.__values[:, 1]
        ax.quiver(x, y, u, v, color="red", headwidth=1, scale=35)

    # Afficher le profil
    def afficher_profil(self, color):
        verts = self.__coords[0:len(self.__coords):35]

        poly = Polygon(verts)
        poly.set_color(color)
        ax.add_patch(poly)

    # Afficher le maillage
    def afficher_maillage(self):
        tabX, tabY = np.array(self.__coords[:, 0]), np.array(
            self.__coords[:, 1])

        # Vertical lines
        for n in range(0, 35 * 87, 35):
            xs, ys = tabX[n:n + 35], tabY[n:n + 35]
            plt.plot(xs, ys, c=(0, 0, 0, .2))

        # Horizontal lines
        for n in range(0, 35):
            xs, ys = tabX[n:len(tabX):35], tabY[n:len(tabY):35]
            plt.plot(xs, ys, c=(0, 0, 0, .2))

    # Afficher les cellules candidates:
    def afficher_cellules(self, color):

        # test sign
        def test_sign(vectors):
            for i in range(len(vectors)-1):
                vec1, vec2 = vectors[i], vectors[i+1]

                # Si le produit scalaire est négatif, alors les vecteurs sont opposés
                if np.dot(vec1, vec2) < 0:
                    return False

            return True

        # Begin
        for cell in self.__cells:

            x1, x2, x3, x4 = cell[:, 0]
            y1, y2, y3, y4 = cell[:, 1]
            u1, u2, u3, u4 = cell[:, 2]
            v1, v2, v3, v4 = cell[:, 3]

            # Les 4 vecteurs
            vecs = [[u1, v1], [u2, v2], [u3, v3], [u4, v4]]

            if not test_sign(vecs):
                verts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                poly = Polygon(verts)
                poly.set_color(color)
                ax.add_patch(poly)

    # Afficher l'objet
    def afficher_objet(self):

        plt.cla()

        self.afficher_profil("grey")
        if self.__dispVec:
            self.afficher_vitesse()
        if self.__dispGrid:
            self.afficher_maillage()
        if self.__dispCellule:
            self.afficher_cellules("blue")

        #print([min(self.__coords[:, 0]), max(self.__coords[:, 0]), min(self.__coords[:, 1]), max(self.__coords[:, 1])])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        # draw lines
        self.calcul_lignes()

        plt.title(
            'Visualisation: M=Grid, C=CellsCandidates, V=Vitesses, R=Reset, DoubleClick=Drop')
        plt.show()

    # calcul_lignes
    def calcul_lignes(self):
        if(len(self.__dropPoints) > 0):
            strm = ax.streamplot(self.__xs, self.__ys, self.__us, self.__vs,
                                 start_points=self.__dropPoints, color=self.__speeds, linewidth=2, cmap="viridis")
            if self.__strm is None:
                self.__strm = strm
                self.__colorbar = fig.colorbar(strm.lines)

    # Ajouter un point de lâchage
    def add_drop_point(self, dropPoint):
        x, y = dropPoint
        if (x < self.__xmin or x > self.__xmax or y < self.__ymin or y > self.__ymax):
            print("Drop point {} is out of bound!").format(dropPoint)
        else:
            self.__dropPoints.append(dropPoint)

    def add_end_section(self, point):
        self.__endSection = point

        begin, end = self.__beginSection, self.__endSection

        if begin != end:
            print("Line created at, ", (begin, end))
            x1, y1 = begin
            x2, y2 = end

            plt.plot(x1, y1, x2, y2, c="crimson", marker="o")
            self.draw_cut_section(begin, end)
            plt.show()

    def draw_cut_section(self, begin, end):

        x1, y1 = begin
        x2, y2 = end

        xs = np.linspace(x1, x2, 3080)
        ys = np.linspace(y1, y2, 3080)

        idx = np.argwhere(np.diff(np.sign(ys - self.__coords[:, 1]))).flatten()

        xr, yr = xs[idx], ys[idx]

        plt.plot(xs[idx], ys[idx], 'ro')

    def add_begin_section(self, point):
        self.__beginSection = point

    # Getters and Setters
    def toggleDispVec(self):
        self.__dispVec = not self.__dispVec

    def toggleDispGrid(self):
        self.__dispGrid = not self.__dispGrid

    def toggleDispCellules(self):
        self.__dispCellule = not self.__dispCellule

    def clearDropPoints(self):
        self.__strm = None
        self.__dropPoints = []
        self.__colorbar.remove()


# Event handlings
def evt_on_button_pressed(event, windows: WindowsViz):
    print('Key pressed', event.key)
    if event.key == 'm':
        windows.toggleDispGrid()

    if event.key == "v":
        windows.toggleDispVec()

    if event.key == "c":
        windows.toggleDispCellules()

    if event.key == "r":
        windows.clearDropPoints()

    # Display
    windows.afficher_objet()


def evt_on_mouse_pressed(event, windows: WindowsViz):

    dropPoint = (event.xdata, event.ydata)
    if event.dblclick:
        print("Particule lâchée à ", dropPoint)
        windows.add_drop_point(dropPoint)
        windows.afficher_objet()
    """
    else:
        windows.add_begin_section(dropPoint)
    """


def evt_on_mouse_released(event, windows: WindowsViz):
    endPoint = (event.xdata, event.ydata)
    windows.add_end_section(endPoint)


if __name__ == "__main__":
    # execute only if run as a script

    # Events handling
    fig.canvas.mpl_connect(
        'key_press_event', lambda event: evt_on_button_pressed(event, windows))
    fig.canvas.mpl_connect('button_press_event',
                           lambda event: evt_on_mouse_pressed(event, windows))
    #fig.canvas.mpl_connect('button_release_event', lambda event: evt_on_mouse_released(event, windows))

    windows = WindowsViz()
    windows.afficher_objet()
