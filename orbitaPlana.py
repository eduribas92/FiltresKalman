import math
import matplotlib.pyplot as plt

def anomaliaExcentrica(e,n,t_p,t):
    M = n*(t - t_p)
    E_0 = M
    tol = 1e-1
    improve = True
    while improve:
        E_i = E_0 - (E_0-e*math.sin(E_0)-M) / (1-e*math.cos(E_0))
        if math.fabs(E_i - E_0) < tol:
            improve = False
        E_0 = E_i   
    return E_i

def SimulacioOrbita():
    num_punts = 15
    e = 0.6
    a = 1.
    b = a * math.sqrt(1-e*e)
    T = 60.
    step = T / num_punts
    n = 2*math.pi / T
    t_p = 0

    x = []
    y = []

    for t in range(0,num_punts):
        E = anomaliaExcentrica(e,n,t_p,t*step)
        x.append( a*(math.cos(E) - e))
        y.append( b*math.sin(E) )

    plt.rcParams['figure.figsize'] = (9, 7)
    plt.close()
    plt.scatter(x, y, c='g', alpha = 1)
    plt.scatter(0, 0, c='b', alpha = 1)
    plt.scatter(-2*e*a, 0, c='b', alpha = 0.3)
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()
    
    return