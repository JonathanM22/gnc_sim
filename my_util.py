from orbit import *
from body import *
import numpy as np
from astropy import constants as const
from astropy import units as u


# Sets up single Runge Kutta 4 Step
def RK4_single_step(fun, dt, t0, y0, fun_arg: list):
    k1 = fun(t0, y0, fun_arg)
    k2 = fun((t0 + (dt/2)), (y0 + ((dt.value/2)*k1)), fun_arg)
    k3 = fun((t0 + (dt/2)), (y0 + ((dt.value/2)*k2)), fun_arg)
    k4 = fun((t0 + dt), (y0 + (dt.value*k3)), fun_arg)

    y1 = y0 + (dt.value/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y1


def skew_mtx(x_vec):
    x1 = x_vec[0]
    x2 = x_vec[1]
    x3 = x_vec[2]

    return np.array([[0, -x3,  x2],
                     [x3,   0, -x1],
                     [-x2,  x1,   0]
                     ])

def euler_321_attitude(phi, theta, psi):
    # Appendix B 3-2-1 matrix OR sec 2.9 pg55
    A11 = np.cos(theta)*np.cos(phi)
    A12 = np.cos(theta)*np.sin(phi)
    A13 = -np.sin(theta)
    A21 = (-np.cos(psi)*np.sin(phi)) + (np.sin(psi)*np.sin(theta)*np.cos(phi))
    A22 = (np.cos(psi)*np.cos(phi)) + (np.sin(psi)*np.sin(theta)*np.sin(phi))
    A23 = np.sin(psi)*np.cos(theta)
    A31 = (np.sin(psi)*np.sin(phi)) + (np.cos(psi)*np.sin(theta)*np.cos(phi))
    A32 = (-np.sin(psi)*np.cos(phi)) + (np.cos(psi)*np.sin(theta)*np.sin(phi))
    A33 = np.cos(psi)*np.cos(theta)

    return np.array([[A11, A12, A13],
                     [A21, A22, A23],
                     [A31, A32, A33]])


def attitude_to_euler321(attitude):
    # https://en.wikiversity.org/wiki/PlanetPhysics/Direction_Cosine_Matrix_to_Euler_321_Angles
    # phi_1, theta_2, psi_3
    A11 = attitude[0][0]
    A12 = attitude[0][1]
    A13 = attitude[0][2]
    A23 = attitude[1][2]
    A33 = attitude[2][2]

    theta = np.arcsin(-A13)
    phi = np.atan2(A12, A11)
    psi = np.atan2(A23, A33)

    return np.array([phi, theta, psi])

def orb_2_pqw(r, f, e, p, mu):
    """
    Transforms orbital frame to perifocal frame
    """
    r_pqw = np.array([r*np.cos(f), r*np.sin(f), r*0])
    v_pqw = np.array(
        [-np.sqrt(mu/p)*np.sin(f), np.sqrt(mu/p)*(e + np.cos(f)), 0])

    return r_pqw, v_pqw


def perif_2_eci(r_pqw, v_pqw, inc, raan, aop):
    """
    Transforms perifocal fram to ECI frame
    """
    # Rotation matrices
    R1 = np.array([  # Third axis rotation about raan
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan),  np.cos(raan), 0],
        [0,             0,            1]
    ])
    R2 = np.array([  # First axis rotation about inc
        [1, 0,              0],
        [0, np.cos(inc), -np.sin(inc)],
        [0, np.sin(inc),  np.cos(inc)]
    ])
    R3 = np.array([  # Third axis rotation about aop
        [np.cos(aop), -np.sin(aop), 0],
        [np.sin(aop),  np.cos(aop), 0],
        [0,            0,           1]
    ])
    perif_2_eci_DCM = R1 @ R2 @ R3
    r_eci = perif_2_eci_DCM @ r_pqw
    v_eci = perif_2_eci_DCM @ v_pqw

    return r_eci, v_eci

def rv_2_orb_elm(r, v, mu):
    """
    Given an r, v vectors and mu of a orbit, calculates all orbital elements. 
    Based on 458 notes 2/10
    """

    I = [1, 0, 0]
    J = [0, 1, 0]
    K = [0, 0, 1]

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    energy = ((v_mag**2) / 2) - (mu/r_mag)
    a = -mu/(2*energy)

    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)

    e_vec = (np.cross(v, h_vec)/mu) - (r/r_mag)
    e = np.linalg.norm(e_vec)

    cos_i = np.dot(K, h_vec) / h
    i = np.arccos(cos_i)

    # Numpy has a special atan2, not using because we do a quad check?
    Kxh = np.cross(K, h_vec)
    n = (Kxh)/(np.linalg.norm(Kxh))

    raan = np.arctan((np.dot(J, n) / np.dot(I, n)))

    rann_quad_check = np.dot(n, I)
    if rann_quad_check < 0:
        raan += np.pi

    # aop quad check
    aop = np.arccos((np.dot(e_vec, n))/e)

    aop_quad_check = np.dot(e_vec, K)
    if aop_quad_check < 0:
        aop = (2*np.pi) - aop

    f = np.arccos(np.dot(e_vec, r)/(e*r_mag))

    radial_v = np.dot(r, (v/r_mag))
    if radial_v < 0:
        f = (2*np.pi) - f

    return a, e, e_vec, i, raan, aop, f
