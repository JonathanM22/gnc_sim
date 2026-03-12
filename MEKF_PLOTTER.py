# Custom libs
from orbit import *
from my_util import *
from body import *
from quaternion import *
from reactionwh import *

# Standard libs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib import gridspec
import time

""" ######### """
""" LOAD DATA """
""" ######### """
data = np.load("nadir_control_sim_data.npy", allow_pickle=True)[()]

r_hist = data["r_hist"]
v_hist = data["v_hist"]
lvlh_attitude_hist = data["lvlh_attitude_hist"]
q_c_hist_sim = data["q_c_hist_sim"]

q_sat_hist = data["q_sat_hist_sim"]
sat_w_hist = data["sat_w_hist_sim"]
q_error_hist = data["q_error_hist_sim"]
L_hist = data["L_hist_sim"]

q_sat_hist2 = data["q_sat_hist_ekf"]
sat_w_hist2 = data["sat_w_hist_ekf"]
q_error_hist2 = data["q_error_hist_ekf"]
L_hist2 = data["L_hist_ekf"]
beta_true = data["beta_true"]
beta_hist = data["beta_hist"]
innovation_hist = data["innovation_hist"]


""" ###################### """
""" PLOTTING ERROR CONTROL """
""" ###################### """

plot_error_control = True
max_L = 3.2e-3
if plot_error_control:
    sti = 0
    eni = round(len(q_error_hist)/3)
    eni = -1
    """PLOT SIM Q_ERROR"""
    fig, ax = plt.subplots(3, 2, figsize=(14.2, 10), sharex=True)
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(q_error_hist[sti:eni, 0], color='tab:red', label=r'$q_1$')
    ax1.plot(q_error_hist[sti:eni, 1], color='tab:green', label=r'$q_2$')
    ax1.plot(q_error_hist[sti:eni, 2], color='tab:blue', label=r'$q_3$')
    ax1.plot(q_error_hist[sti:eni, 3], color='tab:orange', label=r'$q_4$')
    ax1.set_xlabel("steps")
    ax1.set_ylabel("Quaternion Error")
    ax1.set_title(r"[SIM] $\mathbf{q}_{err}$")
    ax1.legend(loc='lower right')
    ax1.grid(True)

    """PLOT SIM SAT_W"""
    ax1 = plt.subplot(3, 2, 3)
    ax1.plot(sat_w_hist[sti:eni, 0], color='tab:red',
             label=r'$\omega^{sat}_1$')
    ax1.plot(sat_w_hist[sti:eni, 1], color='tab:green',
             label=r'$\omega^{sat}_2$')
    ax1.plot(sat_w_hist[sti:eni, 2], color='tab:blue',
             label=r'$\omega^{sat}_3$')
    ax1.set_xlabel("steps")
    ax1.set_ylabel("sat_w")
    ax1.set_title(r"[SIM] $\mathbf{\omega}^{sat}$")
    ax1.legend(loc='upper right')
    ax1.grid(True)

    """PLOT SIM CONTROL MOMENT"""
    ax1 = plt.subplot(3, 2, 5)
    ax1.axhline(y=max_L, color="tab:purple",
                linestyle="--", label=r'$L_{max}$')
    ax1.axhline(y=-max_L, color="tab:purple", linestyle="--")
    ax1.plot(L_hist[sti:eni, 0], color='tab:red', label=r'$L_1$')
    ax1.plot(L_hist[sti:eni, 1], color='tab:green', label=r'$L_2$')
    ax1.plot(L_hist[sti:eni, 2], color='tab:blue', label=r'$L_3$')
    ax1.set_xlabel("steps")
    ax1.set_ylabel("L (Nm)")
    ax1.set_title(r"[SIM] $\mathbf{L}_{sat}$")
    ax1.legend(loc='lower right')
    ax1.grid(True)

    """PLOT EKF Q_ERROR"""
    ax1 = plt.subplot(3, 2, 2)
    ax1.plot(q_error_hist2[sti:eni, 0], color='tab:red', label=r'$q_1$')
    ax1.plot(q_error_hist2[sti:eni, 1], color='tab:green', label=r'$q_2$')
    ax1.plot(q_error_hist2[sti:eni, 2], color='tab:blue', label=r'$q_3$')
    ax1.plot(q_error_hist2[sti:eni, 3], color='tab:orange', label=r'$q_4$')
    ax1.set_xlabel("steps")
    ax1.set_ylabel("Quaternion Error")
    ax1.set_title(r"[MEKF] $\mathbf{q}_{err}$")
    ax1.legend(loc='lower right')
    ax1.grid(True)

    """PLOT EKF SAT_W"""
    ax1 = plt.subplot(3, 2, 4)
    ax1.plot(sat_w_hist2[sti:eni, 0], color='tab:red',
             label=r'$\omega^{sat}_1$')
    ax1.plot(sat_w_hist2[sti:eni, 1], color='tab:green',
             label=r'$\omega^{sat}_2$')
    ax1.plot(sat_w_hist2[sti:eni, 2], color='tab:blue',
             label=r'$\omega^{sat}_3$')
    ax1.set_xlabel("steps")
    ax1.set_ylabel("sat_w")
    ax1.set_title(r"[MEKF] $\mathbf{\omega}^{sat}$")
    ax1.legend(loc='upper right')
    ax1.grid(True)

    """PLOT EKF CONTROL MOMENT"""
    ax1 = plt.subplot(3, 2, 6)
    ax1.axhline(y=max_L, color="tab:purple",
                linestyle="--", label=r'$L_{max}$')
    ax1.axhline(y=-max_L, color="tab:purple", linestyle="--")
    ax1.plot(L_hist2[sti:eni, 0], color='tab:red', label=r'$L_1$')
    ax1.plot(L_hist2[sti:eni, 1], color='tab:green', label=r'$L_1$')
    ax1.plot(L_hist2[sti:eni, 2], color='tab:blue', label=r'$L_1$')
    ax1.set_xlabel("steps")
    ax1.set_ylabel("L (Nm)")
    ax1.set_title(r"[MEKF] $\mathbf{L}_{sat}$")
    ax1.legend(loc='lower right')
    ax1.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.show()


""" ######################## """
""" PLOTTING SAT ORIENTATION """
""" ######################## """
plot_sat_orientation = True
if plot_sat_orientation:
    sti = 0
    eni = round(len(q_sat_hist)/3)
    eni = -1
    fig, ax = plt.subplots(4, 1, figsize=(14.2, 10), sharex=True)
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(q_c_hist_sim[sti:eni, 0], color='tab:orange',
             linestyle='--', label=r'$q_{1,c}$')
    ax1.plot(q_sat_hist[sti:eni, 0], color='tab:blue',
             linestyle='solid', label=r'$q_{1,sim}$')
    ax1.plot(q_sat_hist2[sti:eni, 0], color='tab:green',
             linestyle='solid', label=r'$q_{1,ekf}$')
    ax1.set_xlabel("steps")
    ax1.set_ylabel(r"q_1")
    ax1.set_title(r"$q_{1,c}$ vs $q_{1,sim}$ vs $q_{1,ekf}$")
    ax1.legend(loc='right')
    ax1.grid(True)

    ax1 = plt.subplot(4, 1, 2)
    ax1.plot(q_c_hist_sim[sti:eni, 1], color='tab:orange',
             linestyle='--', label=r'$q_{2,c}$')
    ax1.plot(q_sat_hist[sti:eni, 1], color='tab:blue',
             linestyle='solid', label=r'$q_{2,sim}$')
    ax1.plot(q_sat_hist2[sti:eni, 1], color='tab:green',
             linestyle='solid', label=r'$q_{2,ekf}$')
    ax1.set_xlabel("steps")
    ax1.set_ylabel(r"q_2")
    ax1.set_title(r"$q_{2,c}$ vs $q_{2,sim}$ vs $q_{2,ekf}$")
    ax1.legend(loc='right')
    ax1.grid(True)

    ax1 = plt.subplot(4, 1, 3)
    ax1.plot(q_c_hist_sim[sti:eni, 2], color='tab:orange',
             linestyle='--', label=r'$q_{3,c}$')
    ax1.plot(q_sat_hist[sti:eni, 2], color='tab:blue',
             linestyle='solid', label=r'$q_{3,sim}$')
    ax1.plot(q_sat_hist2[sti:eni, 2], color='tab:green',
             linestyle='solid', label=r'$q_{3,ekf}$')
    ax1.set_xlabel("steps")
    ax1.set_ylabel(r"q_3")
    ax1.set_title(r"$q_{3,c}$ vs $q_{3,sim}$ vs $q_{3,ekf}$")
    ax1.legend(loc='right')
    ax1.grid(True)

    ax1 = plt.subplot(4, 1, 4)
    ax1.plot(q_c_hist_sim[sti:eni, 3], color='tab:orange',
             linestyle='--', label=r'$q_{4,c}$')
    ax1.plot(q_sat_hist[sti:eni, 3], color='tab:blue',
             linestyle='solid', label=r'$q_{4,sim}$')
    ax1.plot(q_sat_hist2[sti:eni, 3], color='tab:green',
             linestyle='solid', label=r'$q_{4,ekf}$')
    ax1.set_xlabel("steps")
    ax1.set_ylabel(r"q_4")
    ax1.set_title(r"$q_{4,c}$ vs $q_{4,sim}$ vs $q_{4,ekf}$")
    ax1.legend(loc='right')
    ax1.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.show()


""" ################## """
""" PLOTTING GYRO BIAS """
""" ################## """
plot_beta = True
radsec_2_deghr = (180/np.pi)*3600
if plot_beta:
    sti = 0
    eni = -1
    fig, ax = plt.subplots(3, 1, figsize=(14.2, 10), sharex=True)
    ax1 = plt.subplot(3, 1, 1)
    ax1.axhline(y=beta_true[0], color="orange", label=r'$\beta_{1,true}$')
    ax1.plot(beta_hist[sti:eni, 0]*radsec_2_deghr,
             color='tab:blue', label=r'$\beta_1$')
    ax1.set_ylabel(r"$\beta_1 \ \frac{\deg}{hr}$")
    ax1.set_title(r"$\beta_1$ vs step")
    ax1.legend(loc='right')
    ax1.grid(True)

    ax1 = plt.subplot(3, 1, 2)
    ax1.axhline(y=beta_true[1], color="orange", label=r'$\beta_{2,true}$')
    ax1.plot(beta_hist[sti:eni, 1]*radsec_2_deghr,
             color='tab:blue', label=r'$\beta_2$')
    ax1.set_ylabel(r"$\beta_2 \ \frac{\deg}{hr}$")
    ax1.set_title(r"$\beta_2$ vs step")
    ax1.legend(loc='right')
    ax1.grid(True)

    ax1 = plt.subplot(3, 1, 3)
    ax1.axhline(y=beta_true[2], color="orange", label=r'$\beta_{3,true}$')
    ax1.plot(beta_hist[sti:eni, 2]*radsec_2_deghr,
             color='tab:blue', label=r'$\beta_3$')
    ax1.set_ylabel(r"$\beta_3 \ \frac{\deg}{hr}$")
    ax1.set_title(r"$\beta_3$ vs step")
    ax1.legend(loc='right')
    ax1.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.show()

""" ######################### """
""" PLOTTING Q NORM CONTRAINT """
""" ######################### """
plot_norm_constraint = True
if plot_norm_constraint:

    q_sat_hist_norm = np.linalg.norm(q_sat_hist, axis=1)
    q_sat_hist_norm2 = np.linalg.norm(q_sat_hist2, axis=1)

    sti = 0
    eni = -1
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1 = plt.subplot(2, 1, 1)
    ax1.axhline(y=1, color="orange", linestyle="--")
    ax1.plot(q_sat_hist_norm, color='tab:blue',
             label=r'$\parallel\mathbf{q}_{sim}\parallel$')
    ax1.set_ylabel("norm")
    ax1.set_title(r'$\parallel\mathbf{q}_{sim}\parallel$')
    ax1.set_ylim(0.99, 1.01)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax1 = plt.subplot(2, 1, 2)
    ax1.axhline(y=1, color="orange", linestyle="--")
    ax1.plot(q_sat_hist_norm2, color='tab:red',
             label=r'$\parallel\mathbf{q}_{mekf}\parallel$')
    ax1.set_ylabel("norm")
    ax1.set_title(r'$\parallel\mathbf{q}_{mekf}\parallel$')
    ax1.set_ylim(0.99, 1.01)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.show()
