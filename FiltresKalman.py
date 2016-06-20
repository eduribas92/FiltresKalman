import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pylab

def KF_iteracio(x_act, P_act, A, B, u, Q, H, R, z, I):
    """ computa una iteracio (prediccio i correccio)
    del Filtre de Kalman """

    # PREDICCIÓ
    x_est = A * x_act + B * u
    P_est = (A * P_act) * np.transpose(A) + Q

    # CORRECCIÓ
    K = P_est * np.transpose(H) * np.linalg.inv(R + H*P_est*np.transpose(H))
    x_act = x_est + K * (z - H*x_est)
    P_act = (I-K*H)*P_est

    return x_act, P_act

def FiltreKalman1D(interval_temps, num_iteracions, estat_inicial, 
                   velocitat_real, A, B, u, Q, H, R, posicio_mesurada, I):
    """ realitza les iteracions del Filtre de Kalman per 
    sistemes d'una dimensió """
    
    posicio_estimada = []
    posicio_estimada.append(estat_inicial)
    x_act = np.matrix([[estat_inicial],
                       [velocitat_real[0][0]]])
    P_act = I

    # iteracions
    for t in range(1,num_iteracions):
        z = np.matrix([[posicio_mesurada[t][0]],
                       [velocitat_real[t][0]]])
        x_act, P_act = KF_iteracio(x_act, P_act, A, B, u, Q, H, R, z, I)
        posicio_estimada.append(x_act[0,0])

    return posicio_estimada

def FiltreKalman2D(interval_temps, num_iteracions, estat_inicial, 
                   velocitat_real, A, B, u, Q, H, R, posicio_mesurada, I):
    """ realitza les iteracions del Filtre de Kalman per 
    sistemes de dues dimensions """
    
    posicio_estimada = []
    posicio_estimada.append(estat_inicial)
    x_act = np.matrix([[estat_inicial[0]],
                       [velocitat_real[0][0]],
                       [estat_inicial[1]],
                       [velocitat_real[0][1]]])
    P_act = I

    # iteracions
    for t in range(1,num_iteracions):
        z = np.matrix([[posicio_mesurada[t][0]],
                       [velocitat_real[t][0]],
                       [posicio_mesurada[t][1]],
                       [velocitat_real[t][1]]])
        x_act, P_act = KF_iteracio(x_act, P_act, A, B, u, Q, H, R, z, I)
        posicio_estimada.append([x_act[0,0], x_act[2,0]])

    return posicio_estimada

def KF_grafic_1D(vector_real, vector_mesures, 
                 vector_estimacions, limits):
    """ mostra el gràfics de trajectòria real, mesurada i estimada
    per sistemes d'una dimensió """

    plt.rcParams['figure.figsize'] = (10, 8)
    plt.close()
    plt.plot(vector_real, 'b-', label='Real')
    plt.plot(vector_mesures, 'g+', label='Mesurat')
    plt.plot(vector_estimacions, 'r--', label='Estimat')
    plt.legend()
    plt.axis(limits)
    plt.title('VALOR A CADA INSTANT\n', fontweight='bold')
    plt.xlabel('Temps')
    plt.ylabel('Valor')
    plt.show()

def KF_grafic_2D(vector_real, vector_mesures, 
                 vector_estimacions, limits):
    """ mostra el gràfics de trajectòria real, mesurada i estimada
    per sistemes de dues dimensions """
        
    x_real, y_real = zip(*vector_real) 
    x_mesura, y_mesura = zip(*vector_mesures) 
    x_estimada, y_estimada = zip(*vector_estimacions) 
    
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.close()
    plt.plot(x_real, y_real, 'b-', label='Real')
    plt.plot(x_mesura, y_mesura, 'g+', label='Mesurada')
    plt.plot(x_estimada, y_estimada, 'r--', label='Estimada')
    pylab.legend()
    plt.axis(limits)
    pylab.title('TRAJECTORIES\n', fontweight='bold')
    pylab.xlabel('Component X')
    pylab.ylabel('Component Y')
    plt.grid()
    plt.show()
    
    return


#--------------------------VALOR CONSTANT--------------------------

def ValorConstant():
    
    # parametres generals
    interval_temps = 0.1
    num_iteracions = 400

    # parametres sistema dinamic
    valor_constant = 9.324
    sigma_soroll = 3

    # estat inicial
    posicio_real = []
    velocitat_real = []
    posicio_mesurada = []
    estat_inicial = 0
    
    # simulacio de trajectoria i mesures
    posicio_real.append([valor_constant])
    velocitat_real.append([0])
    posicio_mesurada.append(
        [random.gauss(posicio_real[0][0],sigma_soroll)])
    for i in range(1,num_iteracions):
        velocitat_real.append([velocitat_real[i-1][0]])
        posicio_real.append([posicio_real[i-1][0]] )
        posicio_mesurada.append(
            [random.gauss(posicio_real[i][0],sigma_soroll)])        
        
    # matrius del model
    A = np.eye(2)
    B = np.zeros(2)
    u = np.matrix([[0],[0]])
    Q = np.zeros(2)
    H = np.eye(2)
    R = np.eye(2)*0.2
    I = np.eye(2)

    # FILTRE DE KALMAN
    posicio_estimada = FiltreKalman1D(interval_temps, num_iteracions, 
                                      estat_inicial, velocitat_real, A, B, 
                                      u, Q, H, R, posicio_mesurada, I)

    # GRÀFICS
    limits = [ -10 , 410 , 0 , 20 ]
    KF_grafic_1D(posicio_real, posicio_mesurada, posicio_estimada, limits)

    print posicio_real[-1], posicio_estimada[-1]
    
    return


#------------------------TRAJECTÒRIA LINEAL------------------------

def TrajectoriaLineal():
    
    # parametres generals
    interval_temps = 0.1
    num_iteracions = 175

    # parametres sistema dinamic
    angle = 330 
    velocitat_sortida = 50 
    sigma_soroll = 50

    # estat inicial
    posicio_real = []
    velocitat_real = []
    posicio_mesurada = []
    estat_inicial = [0,0]
    
    # simulacio de trajectoria i mesures
    posicio_real.append([0,200])
    velocitat_real.append([velocitat_sortida*math.cos(angle*math.pi/180), 
                           velocitat_sortida*math.sin(angle*math.pi/180)])
    posicio_mesurada.append([random.gauss(posicio_real[0][0],
                                          sigma_soroll), 
                             random.gauss(posicio_real[0][1],
                                          sigma_soroll)])
    for i in range(1,num_iteracions):
        velocitat_real.append([velocitat_real[i-1][0], 
                                velocitat_real[i-1][1]] )
        posicio_real.append([posicio_real[i-1][0]
                             +velocitat_real[i][0]*interval_temps, 
                             posicio_real[i-1][1]
                             +velocitat_real[i][1]*interval_temps] )
        posicio_mesurada.append([random.gauss(posicio_real[i][0],
                                              sigma_soroll), 
                                  random.gauss(posicio_real[i][1],
                                               sigma_soroll)])        
        
    # matrius del model
    A = np.matrix([[1,interval_temps,0,0],
                   [0,1,0,0],
                   [0,0,1,interval_temps],
                   [0,0,0,1]])
    B = np.zeros(4)
    u = np.matrix([[0],[0],[0],[0]])
    Q = np.zeros(4)
    H = np.eye(4)
    R = np.eye(4)*0.5
    I = np.eye(4)

    # FILTRE DE KALMAN
    posicio_estimada = FiltreKalman2D(
        interval_temps, num_iteracions, estat_inicial, 
        velocitat_real, A, B, u, Q, H, R, posicio_mesurada, I)

    # GRÀFICS
    limits = [ -50 , 800 , -300 , 300 ]
    KF_grafic_2D(posicio_real, posicio_mesurada, posicio_estimada, limits)
    
    print posicio_real[-1], posicio_estimada[-1]

    return


#--------------------------TIR PARABÒLIC---------------------------

def TirParabolic():
    
    # parametres generals
    interval_temps = 0.1
    num_iteracions = 175

    # parametres sistema dinamic
    angle = 60 
    velocitat_sortida = 100 
    gravetat = -9.81
    sigma_soroll = 30

    # estat inicial
    posicio_real = []
    velocitat_real = []
    posicio_mesurada = []
    estat_inicial = [0,500]
    
    # simulacio de trajectoria i mesures
    posicio_real.append([0,0])
    velocitat_real.append([velocitat_sortida*math.cos(angle*math.pi/180), 
                           velocitat_sortida*math.sin(angle*math.pi/180)])
    posicio_mesurada.append([random.gauss(posicio_real[0][0],sigma_soroll), 
                             random.gauss(posicio_real[0][1],sigma_soroll)])
    for i in range(1,num_iteracions):
        velocitat_real.append([velocitat_real[i-1][0], 
                               velocitat_real[i-1][1]+gravetat*interval_temps])
        posicio_real.append([posicio_real[i-1][0]+
                             velocitat_real[i][0]*interval_temps, 
                             posicio_real[i-1][1]+
                             velocitat_real[i][1]*interval_temps])
        posicio_mesurada.append([random.gauss(posicio_real[i][0],sigma_soroll), 
                                 random.gauss(posicio_real[i][1],sigma_soroll)])

    # matrius del model
    A = np.matrix([[1,interval_temps,0,0],
                   [0,1,0,0],
                   [0,0,1,interval_temps],
                   [0,0,0,1]])
    B = np.matrix([[0,0,0,0],
                   [0,0,0,0],
                   [0,0,1,0],
                   [0,0,0,1]])
    u = np.matrix([[0],
                   [0],
                   [0.5*gravetat*interval_temps*interval_temps],
                   [gravetat*interval_temps]])
    Q = np.zeros(4)
    H = np.eye(4)
    R = np.eye(4)*0.4
    I = np.eye(4)

    # FILTRE DE KALMAN
    posicio_estimada = FiltreKalman2D(
        interval_temps, num_iteracions, estat_inicial, 
        velocitat_real, A, B, u, Q, H, R, posicio_mesurada, I)

    # GRÀFICS
    limits = [ -50 , 950 , -50 , 550 ]
    KF_grafic_2D(posicio_real, posicio_mesurada, posicio_estimada, limits)
    
    print posicio_real[-1], posicio_estimada[-1]

    return