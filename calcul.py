# fonction calcul
import numpy as np
import math as m

def dp(p, q, xa, xb, xc, xd):
    return ((1.0-(q)) * ((xd) - (xa)) + (q) * ((xc) - (xb)))


def dq(p, q, xa, xb, xc, xd):
    return ((1.0-(p)) * ((xb) - (xa)) + (p) * ((xc) - (xd)))


def dedans(xa, xb, xc, xd, ya, yb, yc, yd, xr, yr):

    def clean_cos(cos_angle):
        return min(1, max(cos_angle, -1))

    vx = np.array([xa-xr, xb-xr, xc-xr, xd-xr], dtype=np.float64)
    vy = np.array([ya-yr, yb-yr, yc-yr, yd-yr], dtype=np.float64)

    v, s, vp = 0, 0, 0

    for i in range(4):
        s = m.sqrt(vx[i]*vx[i]+vy[i]*vy[i])
        vx[i] = vx[i] / s
        vy[i] = vy[i] / s

    vx = np.append(vx, vx[0])
    vy = np.append(vy, vy[0])
    v = 0

    for i in range(4):
        t = (vx[i]*vx[i+1]+vy[i]*vy[i+1])
        vp = m.acos(clean_cos(t))
        v = v + vp

    if((6.20 < abs(v) and abs(v) < 6.32)):
        return True

    return False


def localiser(xp, yp):
    celi, celj = -1, -1
    for i in range(nb_vertical-1):
        for j in range(nb_horizontal-1):
            if(dedans(x[i*nb_horizontal+j], x[(i+1)*nb_horizontal+j], x[(i+1)*nb_horizontal+(j+1)], x[i*nb_horizontal+(j+1)],
                      y[i*nb_horizontal+j], y[(i+1)*nb_horizontal+j], y[(i+1)*nb_horizontal+(j+1)], y[i*nb_horizontal+(j+1)], xp, yp)):
                celi, celj = i, j
                return celi, celj

    return celi, celj


def pq2xy(i, j, p, q):
    ump, umq = 1.0 - p, 1.0 - q
    xr = ump * (umq * x[i*nb_horizontal+j] + q * x[i*nb_horizontal+(j+1)]) + \
        p * (umq * x[(i+1)*nb_horizontal+j] + q * x[(i+1)*nb_horizontal+(j+1)])
    yr = ump * (umq * y[i*nb_horizontal+j] + q * y[i*nb_horizontal+(j+1)]) + \
        p * (umq * y[(i+1)*nb_horizontal+j] + q * y[(i+1)*nb_horizontal+(j+1)])

    return xr, yr


def xypq(x, y, x00, x01, x11, x10, y00, y01, y11, y10, pmin, pmax, qmin, qmax):

    xa = (x01 + x00) / 2.0
    xb = (x11 + x01) / 2.0
    xc = (x10 + x11) / 2.0
    xd = (x00 + x10) / 2.0

    ya = (y01 + y00) / 2.0
    yb = (y11 + y01) / 2.0
    yc = (y10 + y11) / 2.0
    yd = (y00 + y10) / 2.0

    xe = (xa + xb + xc + xd) / 4.0
    ye = (ya + yb + yc + yd) / 4.0

    p = (pmin + pmax) / 2.0
    q = (qmin + qmax) / 2.0

    if((pmax - pmin) < 0.0001):
        p = (pmin + pmax) / 2.0
        q = (qmin + qmax) / 2.0
        return p, q

    if(dedans(x00, xa, xe, xd, y00, ya, ye, yd, x, y)):
        return xypq(x, y, x00, xa, xe, xd, y00, ya, ye, yd, pmin,
                    (pmin + pmax)/2.0, qmin, (qmin + qmax)/2.0)
    else:
        if(dedans(xa, x01, xb, xe, ya, y01, yb, ye, x, y)):
            return xypq(x, y, xa, x01, xb, xe, ya, y01, yb, ye, pmin, (pmin+pmax)/2.0, (qmin+qmax)/2.0, qmax)
        else:
            if(dedans(xe, xb, x11, xc, ye, yb, x11, yc, x, y)):
                return xypq(x, y, xe, xb, x11, xc, ye, yb, x11, yc, (pmin+pmax)/2.0, pmax, (qmin+qmax)/2.0, qmax)
            else:
                if(dedans(xd, xe, xc, x10, yd, ye, yc, y10, x, y)):
                    return xypq(x, y, xd, xe, xc, x10, yd, ye, yc, y10, (pmin+pmax)/2.0, pmax, qmin, (qmin+qmax)/2.0)
    return p, q


def xy2pq(i, j, xr, yr):
    return xypq(xr, yr, x[i*nb_horizontal+j], x[(i+1)*nb_horizontal+j], x[(i+1)*nb_horizontal+(j+1)], x[i*nb_horizontal+(j+1)],
                y[i*nb_horizontal+j], y[(i+1)*nb_horizontal+j], y[(i+1)*nb_horizontal+(j+1)], y[i*nb_horizontal+(j+1)], 0.0, 1.0, 0.0, 1.0)
