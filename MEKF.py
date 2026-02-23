# Custom libs
from orbit import *
from my_util import *
from body import *
from quaternion import *
from reactionwh import *

# Standard libs
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import time
from astropy.time import Time
from astropy.time import TimeDelta
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
from astropy.coordinates import get_body_barycentric


def F_mtx(sat_w_pos):

    return np.block([
        [-skew_mtx(sat_w_pos.flatten()), -np.eye(3)],
        [np.zeros((3, 3)), np.zeros((3, 3))]
    ])


def P_dyanmics(t0, y0, fun_arg):
    P = y0
    F = fun_arg[0]
    G = fun_arg[1]
    Q = fun_arg[2]
    P_dot = (F@P) + (P@F.T) + (G@Q@G.T)
    return P_dot


def star_tracker_measurment(q_true):
    rng = np.random.default_rng()

    # Model from:
    # https://s3vi.ndc.nasa.gov/ssri-kb/static/resources/Attitude%20Determination%20&%20Control%20System%20Design%20and%20Implementation.pdf

    sigma_roll = 40 * (np.pi / (180*3600))  # arcsec to rad
    sigma_bs = 6 * (np.pi / (180*3600))  # arcsec to rad

    phi_roll = rng.normal(0.0, sigma_roll)
    phi_bs = rng.normal(0.0, sigma_bs)

    # Error in roll
    e_r = np.array([1, 0, 0])
    q_str_roll = Quaternion(np.array([
        np.sin(phi_roll/2)*e_r[0],
        np.sin(phi_roll/2)*e_r[1],
        np.sin(phi_roll/2)*e_r[2],
        np.cos(phi_roll/2)]).reshape(4, 1))

    # Error in boresight
    theta = rng.uniform(-np.deg2rad(1), np.deg2rad(1))
    e_bs = np.array([0, np.cos(theta), np.sin(theta)])

    q_str_bs = Quaternion(np.array([
        np.sin(phi_bs/2)*e_bs[0],
        np.sin(phi_bs/2)*e_bs[1],
        np.sin(phi_bs/2)*e_bs[2],
        np.cos(phi_bs/2)]).reshape(4, 1))

    # Star tracker measurment
    q_str = q_true.cross(q_str_bs.cross(q_str_roll))

    return q_str


def y_dot_nadir(t, y, fun_arg: list):

    # Unpack state
    r, v, q_sat, sat_w = unpack_state_ar(y)

    # Unpack fun_arg
    central_body = fun_arg[0]
    bodies = fun_arg[1]
    JB = fun_arg[2]
    Lwh_b = fun_arg[3]

    # Calc r_dot, v_dot
    r_mag = np.linalg.norm(r)
    r_c = get_body_barycentric(central_body.label, t).xyz.to(u.km).value
    m_c = central_body.mass.value
    a = ((central_body.mu.value)/(r_mag**3)) * -r

    for body in bodies:
        r_k = get_body_barycentric(body.label, t).xyz.to(u.km).value
        m_k = body.mass.value
        r_ck = r_k - r_c
        r_sk = r_ck - r
        r_sk_mag = np.linalg.norm(r_sk)
        r_ck_mag = np.linalg.norm(r_ck)
        a_k = ((body.mu.value)/(r_sk_mag**3)) * r_sk
        a_cb_k = ((body.mu.value)/(r_ck_mag**3)) * r_ck
        a += a_k - a_cb_k

    # Calc sat_w_dot
    total_wh_h = wh1.Hwh_body(
        sat_w) + wh2.Hwh_body(sat_w) + wh3.Hwh_body(sat_w)
    sat_h = (JB@sat_w) + total_wh_h
    sat_w_dot = sat_dynamics(sat_h, sat_w, Lwh_b)

    # Calc q_sat_dot
    q_sat_dot = q_sat.kinamatics(sat_w)

    return np.concatenate([v, a, q_sat_dot.value.flatten(), sat_w_dot.flatten()])


def unpack_state_ar(yi):
    # Unpacks a state to components for calculations
    r = yi[0:3]
    v = yi[3:6]
    q_sat = Quaternion(yi[6:10].reshape(4, 1)).normalized()
    sat_w = yi[10:].reshape(3, 1)

    return r, v, q_sat, sat_w


def rv_to_lvlh(r, v):
    # Define LVLH frame
    O1 = -r/np.linalg.norm(r)
    O2 = -np.cross(r, v)/np.linalg.norm(np.cross(r, v))
    O3 = np.cross(O1, O2)

    return np.stack([O1, O2, O3], axis=1)


def sat_dynamics(sat_h, sat_w, Lwh_b):
    # EOM
    return np.linalg.inv(JB)@(-Lwh_b-(np.cross(sat_w, sat_h, axis=0)))


# Define Earth and Parking Orbit
epoch = Time("2026-11-08")
EARTH_MASS = const.M_earth
EARTH_RAD = const.R_earth
earth = Body(EARTH_MASS, epoch, celestial_body="earth", color="green")
EARTH_MU = const.GM_earth.to(u.km**3 / u.s**2)
earth.mu = EARTH_MU
earth_parking = Orbit(mu=EARTH_MU,
                      a=(EARTH_RAD.to(u.km)+925*u.km),
                      e=0.80*u.km/u.km,
                      f0=(180*u.deg).to(u.rad),
                      inc=(28*u.deg).to(u.rad),
                      raan=(175*u.deg).to(u.rad),
                      aop=(240*u.deg).to(u.rad)
                      )

# Define Moon
MOON_MASS = (7.34 * 10**22) * u.kg
moon = Body(MOON_MASS, epoch, celestial_body="moon", color='grey')


# Intialize SAT
SAT_MASS = 3*u.kg
sat = Spacecraft(SAT_MASS, epoch, label="sat", color="purple")
sat.inertia = np.array([[7.1685e-1, -1.9186e-5, -6.4410e-4,],
                        [-1.9186e-5, 6.4006e-2, -6.6529e-6],
                        [-6.4410e-4, -6.6529e-6, 2.0095e-2]
                        ])
sat_orbit = earth_parking

# Getting intial position & vel. This r is wrt to the central body: earth
r = sat_orbit.r_at_true_anomaly(sat_orbit.f0)

r_pqw, v_pqw = orb_2_pqw(r.value,
                         sat_orbit.f0.value, sat_orbit.e.value,
                         sat_orbit.p.value, sat_orbit.mu.value)

r_eci, v_eci = perif_2_eci(r_pqw, v_pqw, sat_orbit.inc.value,
                           sat_orbit.raan.value, sat_orbit.aop.value)

# both of these are relative to the CB: earth
sat.r0 = r_eci * u.km
sat.v0 = v_eci * (u.km/u.s)


# Define Reaction wheels
# https://nanoavionics.com/cubesat-components/cubesat-reaction-wheels-control-system-satbus-4rw/
# https://scienceworld.wolfram.com/physics/MomentofInertiaCylinder.html
max_L = 3.2e-3  # Nm
wh_rad = 0.0435/2  # m
wh_h = 0.024  # m
wh_mass = 0.137  # kg
J_spin = 0.5*wh_mass*(wh_rad**2)
J_perp = ((1/12)*wh_mass*(wh_h**2)) + 0.25*wh_mass*(wh_rad**2)

wh1 = ReactionWh(0.137, 3e-5, 1.5e-5, np.array([1, 0, 0]).reshape(3, 1))
wh2 = ReactionWh(0.137, 3e-5, 1.5e-5, np.array([0, 1, 0]).reshape(3, 1))
wh3 = ReactionWh(0.137, 3e-5, 1.5e-5, np.array([0, 0, 1]).reshape(3, 1))

# JB = inertia of S/C with Jwh_body
JB = sat.inertia + wh1.Jwh_body_perp() + wh2.Jwh_body_perp() + \
    wh3.Jwh_body_perp()

# Control Law constants
kp = 0.064*4
kd = 0.032*6

# Sim central and kth bodies
central_body = earth
bodies = [moon]

# Setting Up Time series
epoch = Time("2026-11-08")
dt = TimeDelta(0.1, format='sec')
# dt_steps = np.round((sat_orbit.period(sat_orbit.a, sat_orbit.mu).value/40)/dt.value)
dt_steps = np.round((60)/dt.value)
t0 = epoch - dt*dt_steps + dt
tf = epoch
ts_sim = np.arange(t0, tf+dt, dt)
n_steps = len(ts_sim)
ys_sim = np.zeros((n_steps, 13))
ts_ekf = np.arange(t0, tf+dt, dt)
ys_ekf = np.zeros((n_steps, 13))

# Simulation Intial oritentation and angular velocity
phi0 = 60
theat0 = 45
psi0 = 30
q_sat0_sim = Quaternion.from_euler321(np.deg2rad(
    phi0), np.deg2rad(theat0), np.deg2rad(psi0))

w1 = -0.1
w2 = -0.05
w3 = 0.08
sat_w0_sim = np.array([w1, w2, w3]).reshape(3, 1)
y0_sim = np.concatenate([sat.r0.value, sat.v0.value,
                         q_sat0_sim.value.flatten(), sat_w0_sim.flatten()])
ys_sim[0] = y0_sim
ts_sim[0] = t0

# EKF Intial oritentation and angular velocity
rng = np.random.default_rng()
low = -45
high = 45
phi0 = 70 + rng.uniform(low=low, high=high)
theat0 = 35 + rng.uniform(low=low, high=high)
psi0 = 40 + rng.uniform(low=low, high=high)
q_sat0_ekf = Quaternion.from_euler321(np.deg2rad(
    phi0), np.deg2rad(theat0), np.deg2rad(psi0))


low = -0.2
high = 0.2
w1 = w1 + rng.uniform(low=low, high=high)
w2 = w2 + rng.uniform(low=low, high=high)
w3 = w3 + rng.uniform(low=low, high=high)
sat_w0_ekf = np.array([w1, w2, w3]).reshape(3, 1)
y0_ekf = np.concatenate([sat.r0.value, sat.v0.value,
                         q_sat0_ekf.value.flatten(), sat_w0_ekf.flatten()])
ys_ekf[0] = y0_ekf
ts_ekf[0] = t0

# Gyro bias
sigma_v = np.sqrt(10) * 10**-10  # rad/(s^3/2)
sigma_u = np.sqrt(10) * 10**-7

eta_v = rng.normal(0.0, sigma_v**2)
eta_u = rng.normal(0.0, sigma_u**2)

beta = 0.01 * (np.pi/180) * (1/3600)  # rad/s
beta_true = np.array([beta, beta, beta]).reshape(3, 1)
beta_hat_prior = np.zeros((3, 1))

# Intial error Covariance
# From textbook pg252
q_state_cov = (((6/3600)*(np.pi/180))**2)*np.eye(3)  # rad^2
bias_cov = (2e-5)**2 * np.eye(3)
P_pri = np.block([
    [q_state_cov, np.zeros((3, 3))],
    [np.zeros((3, 3)), bias_cov]
])
H = np.hstack((np.eye(3), np.zeros((3, 3))))
R = 36*((np.pi / (180*3600))**2)*np.eye(3)

G = np.block([
    [-np.eye(3), np.zeros((3, 3))],
    [np.zeros((3, 3)), np.eye(3)]
])

Q = np.block([
    [(sigma_v**2)*np.eye(3), np.zeros((3, 3))],
    [np.zeros((3, 3)), np.zeros((3, 3))]
])

# For Plotting
q_c_hist_sim = np.zeros((n_steps, 4))
q_error_hist_sim = np.zeros((n_steps, 4))
L_hist_sim = np.zeros((n_steps, 3))

q_error_hist_ekf = np.zeros((n_steps, 4))
L_hist_ekf = np.zeros((n_steps, 3))
beta_hist = np.zeros((n_steps, 3))
P_hist = np.zeros((n_steps, 6, 6))
innovation_hist = np.zeros((n_steps, 3))

lvlh_attitude_hist = np.zeros((n_steps, 3, 3))

# Propgation
start_time = time.perf_counter()
print("Started SIM")
for i in range(n_steps-1):

    """
    SIM
    """
    # Unpack simlutoin state
    r, v, q_sat_sim, sat_w_sim = unpack_state_ar(ys_sim[i])

    # Calc Control Moment
    lvlh_attitude = rv_to_lvlh(r, v)
    q_c = Quaternion.from_attitude(lvlh_attitude)
    q_error_sim = q_sat_sim.cross(q_c.inverse())
    Lwh_b_sim = (-kp*np.sign(q_error_sim.q4) *
                 q_error_sim.vector - kd*sat_w_sim)*-1

    # Propgate one step w/ Control Moment
    ys_sim[i+1] = RK4_single_step(y_dot_nadir, dt, ts_sim[i],
                                  ys_sim[i], fun_arg=[central_body, bodies, JB, Lwh_b_sim])

    # For plotting
    q_c_hist_sim[i] = q_c.value.flatten()
    q_error_hist_sim[i] = q_error_sim.value.flatten()
    L_hist_sim[i] = Lwh_b_sim.flatten()
    lvlh_attitude_hist[i, :, :] = lvlh_attitude

    """
    MEKF
    """
    # Unpack EKF prior state values
    r, v, q_sat_pri, sat_w_pri = unpack_state_ar(ys_ekf[i])

    # Calc Kalman Gain
    K = P_pri @ H.T @ np.linalg.inv(H @ P_pri @ H.T + R)

    # Calculate posterior state Cov
    P_pos = (np.eye(6) - K@H) @ P_pri

    # True quaternion comes from simulation
    q_str = star_tracker_measurment(q_sat_sim)

    y = 2*((q_str.cross(q_sat_pri.inverse()).vector) /
           (q_str.cross(q_sat_pri.inverse()).scalor))

    ekf_x_hat_post = K @ y

    v_hat_post = ekf_x_hat_post[0:3].reshape(3, 1)
    beta_hat_post = beta_hat_prior + ekf_x_hat_post[3:6].reshape(3, 1)

    q_star = q_sat_pri + Quaternion(0.5*Quaternion.eps(q_sat_pri)@v_hat_post)

    q_sat_pos = q_star.normalized()

    noise = rng.normal(0, sigma_v, (3, 1))
    w_measured = sat_w_sim + beta_true + noise
    sat_w_pos = w_measured - beta_hat_post

    # Calc Control Moment for estimated state
    q_error_ekf = q_sat_pos.cross(q_c.inverse())
    Lwh_b_ekf = (-kp*np.sign(q_error_ekf.q4) *
                 q_error_ekf.vector - kd*sat_w_sim)*-1

    # Updates state with posterior estimate values
    ys_ekf[i] = np.concatenate([r, v,
                                q_sat_pos.value.flatten(), sat_w_pos.flatten()])

    # EKF Propgate Posterior State w/ new Control Moment
    ys_ekf[i+1] = RK4_single_step(y_dot_nadir, dt, ts_ekf[i],
                                  ys_ekf[i], fun_arg=[central_body, bodies, JB, Lwh_b_ekf])

    # Progagate state coveriance dyanmics and make that the prior for next step.
    F = F_mtx(sat_w_pos)
    P_pri = RK4_single_step(P_dyanmics, dt, ts_ekf[i],
                            P_pos, fun_arg=[F, G, Q])

    print(f"{i} of {n_steps}")
    # print("K gain:", K)
    # print("beta_priior:", beta_hat_prior.flatten())
    # print("beta_hat:", beta_hat_post.flatten())
    # print("innovation y:", y.flatten())
    # print("--------------------------------------------------------------")
    beta_hat_prior = beta_hat_post

    # For plotting
    beta_hist[i] = beta_hat_post.flatten()
    innovation_hist[i] = y.flatten()
    q_error_hist_ekf[i] = q_error_ekf.value.flatten()
    L_hist_ekf[i] = Lwh_b_ekf.flatten()


end_time = time.perf_counter()

# For plotting
q_c_hist_sim[i+1] = q_c.value.flatten()
innovation_hist[i+1] = y.flatten()
q_error_hist_sim[i+1] = q_error_sim.value.flatten()
L_hist_sim[i+1] = Lwh_b_sim.flatten()
lvlh_attitude_hist[i+1, :, :] = lvlh_attitude
beta_hist[i+1] = beta_hat_post.flatten()
q_error_hist_ekf[i+1] = q_error_ekf.value.flatten()
L_hist_ekf[i+1] = Lwh_b_ekf.flatten()

r_hist = np.zeros((n_steps, 3))
v_hist = np.zeros((n_steps, 3))
q_sat_hist_sim = np.zeros((n_steps, 4))
sat_w_hist_sim = np.zeros((n_steps, 3))

for i, state in enumerate(ys_sim):
    r, v, q_sat, sat_w = unpack_state_ar(state)
    r_hist[i] = r
    v_hist[i] = v
    q_sat_hist_sim[i] = q_sat.value.flatten()
    sat_w_hist_sim[i] = sat_w.flatten()

q_sat_hist_ekf = np.zeros((n_steps, 4))
sat_w_hist_ekf = np.zeros((n_steps, 3))

for i, state in enumerate(ys_ekf):
    r, v, q_sat, sat_w = unpack_state_ar(state)
    q_sat_hist_ekf[i] = q_sat.value.flatten()
    sat_w_hist_ekf[i] = sat_w.flatten()


data_dict = {
    "central_body": central_body,
    "sat_orbit": sat_orbit,
    "bodies": bodies,
    "sat": sat,
    "reaction_wheels": [wh1, wh2, wh3],
    "epoch": epoch,
    "dt": dt,
    "n_steps": n_steps,
    "t0": t0,
    "tf": tf,
    "y0_sim": y0_sim,
    "y0_ekf": y0_ekf,
    "ts_sim": ts_sim,
    "ys_sim": ys_sim,
    "q_c_hist_sim": q_c_hist_sim,
    "q_error_hist_sim": q_error_hist_sim,
    "q_error_hist_ekf": q_error_hist_ekf,
    "L_hist_sim": L_hist_sim,
    "L_hist_ekf": L_hist_ekf,
    "q_sat_hist_sim": q_sat_hist_sim,
    "q_sat_hist_ekf": q_sat_hist_ekf,
    "sat_w_hist_sim": sat_w_hist_sim,
    "sat_w_hist_ekf": sat_w_hist_ekf,
    "beta_true": beta_true,
    "beta_hist": beta_hist,
    "innovation_hist": innovation_hist,
    "r_hist": r_hist,
    "v_hist": v_hist,
    "lvlh_attitude_hist": lvlh_attitude_hist,
    "kp": kp,
    "kd": kd,
    "JB": JB
}

nadir_control_sim_data = np.save("nadir_control_sim_data", np.array(data_dict))

print("Error History SIM")
print(q_error_hist_sim[-10:-1])
print("--------------------------------------------------------------")
print("Error History EKF")
print(q_error_hist_ekf[-10:-1])
print("--------------------------------------------------------------")
print(f"kp: {kp} | kd: {kd}")
print(f"dt: {dt}")
print(f"t0: {t0}")
print(f"tf: {tf}")
print(f"tf-t0: {(tf-t0).to(u.s)}")
print(f"q_sat0_sim: {q_sat0_sim}")
print(f"q_sat0_ekf: {q_sat0_ekf}")

print(DEBUG)

# print(F_mtx(sat_w_pos))
# print(G)
# print(H)
# print(Q)
# print(P_pos)

# OUTPUT
