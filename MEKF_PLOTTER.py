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

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

""" ################### """
"""    PLOTTING VIDEO   """
""" ################### """
plot_vid = False
""" ----------------- PLOT CHANGING FRAME ----------------- """
if plot_vid:
    # Animation driver
    # Change to reflect your file location!
    plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\jonam\Desktop\Aero-Project\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

    # Vars
    sti = 0
    eni = 100
    stp = 1
    dpi = 150
    fps = 24
    pause = 2

    # Setup Ani
    metadata = dict(title='sat_orientation', artist='jonamat03@gmail.com')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    frames = range(sti, eni, stp)

    """NEED TO SAVE IT OUTSIDE OF REPO TOO BIG FOR GIT"""
    save_path = r"C:\Users\jonam\Desktop\Aero-Project\nadir_control_sim.mp4"
    with writer.saving(fig, "nadir_control_sim.mp4", dpi):
        for i, f in enumerate(frames):
            plt.cla()

            lvlh_attitude = lvlh_attitude_hist[f]
            q_sat_attitude = Quaternion(
                q_sat_hist[f].reshape(4, 1)).to_attitude()

            """ PLOT LVLH FRAME """
            origin = np.array([0, 0, 0])
            O1 = lvlh_attitude[:, 0]
            O2 = lvlh_attitude[:, 1]
            O3 = lvlh_attitude[:, 2]
            lvlh_color = "tab:blue"

            ax.quiver(origin[0], origin[1], origin[2],
                      O1[0], O1[1], O1[2],
                      color=lvlh_color, label='O1')

            ax.quiver(origin[0], origin[1], origin[2],
                      O2[0], O2[1], O2[2],
                      color=lvlh_color, label='O2')

            ax.quiver(origin[0], origin[1], origin[2],
                      O3[0], O3[1], O3[2],
                      color=lvlh_color, label='O3')

            """ PLOT SAT ORIENTATION"""
            S1 = q_sat_attitude[:, 0]
            S2 = q_sat_attitude[:, 1]
            S3 = q_sat_attitude[:, 2]
            q_sat_color = "tab:orange"

            ax.quiver(origin[0], origin[1], origin[2],
                      S1[0], S1[1], S1[2],
                      color=q_sat_color, label='S1')

            ax.quiver(origin[0], origin[1], origin[2],
                      S2[0], S2[1], S2[2],
                      color=q_sat_color, label='S2')

            ax.quiver(origin[0], origin[1], origin[2],
                      S3[0], S3[1], S3[2],
                      color=q_sat_color, label='S3')

            ax.set_title(f"Quaternion Control", fontsize=14, pad=10)
            ax.set_aspect('equal')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            plt.tight_layout()

            # ax.view_init(elev=45, azim=azim_angle[i], roll=0)
            ax.legend(loc='best')
            writer.grab_frame()
            print(f'Generated Leg 1 frame {i} of {len(frames)}')

    print("Done generating Video")

""" ---------------------- CHAT GPT HELPED WITH GRID SPEC ----------------------  """
# Figure and gridspec
sti, eni, stp = 0, 1000, 2
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1.2], height_ratios=[
                       1, 1, 1], wspace=0.3, hspace=0.4)

# Left column: stacked plots
ax_qerror = fig.add_subplot(gs[0, 0])
ax_satw = fig.add_subplot(gs[1, 0])
ax_L = fig.add_subplot(gs[2, 0])

# Right column: 3D animation
ax_3d = fig.add_subplot(gs[:, 1], projection='3d')  # spans all 3 rows

plot_vid = False
if plot_vid:
    plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\jonam\Desktop\Aero-Project\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

    metadata = dict(title='sat_orientation', artist='jonamat03@gmail.com')
    writer = FFMpegWriter(fps=24, metadata=metadata)

    frames = range(sti, eni, stp)
    azim_angle = np.linspace(0, 90, len(frames))
    origin = np.array([0, 0, 0])
    L_arrow = 1.0

    """NEED TO SAVE IT OUTSIDE OF REPO TOO BIG FOR GIT"""
    save_path = r"C:\Users\jonam\Desktop\Aero-Project\nadir_control_sim_combined.mp4"
    with writer.saving(fig, save_path, dpi=150):
        for i, f in enumerate(frames):
            ax_3d.cla()  # clear 3D axes
            ax_qerror.cla()
            ax_satw.cla()
            ax_L.cla()

            # Q_ERROR
            ax_qerror.plot(q_error_hist[sti:f, 0], label='q1')
            ax_qerror.plot(q_error_hist[sti:f, 1], label='q2')
            ax_qerror.plot(q_error_hist[sti:f, 2], label='q3')
            ax_qerror.plot(q_error_hist[sti:f, 3], label='q4')
            ax_qerror.set_ylabel("Quaternion Error")
            ax_qerror.set_title("Quaternion Error vs steps")
            ax_qerror.legend()
            ax_qerror.grid(True)

            # SAT_W
            ax_satw.plot(sat_w_hist[sti:f, 0], label='w1')
            ax_satw.plot(sat_w_hist[sti:f, 1], label='w2')
            ax_satw.plot(sat_w_hist[sti:f, 2], label='w3')
            ax_satw.set_ylabel("sat_w")
            ax_satw.set_title("Satellite Angular Velocity")
            ax_satw.legend()
            ax_satw.grid(True)

            # CONTROL MOMENT
            ax_L.plot(L_hist[sti:f, 0], label='L1')
            ax_L.plot(L_hist[sti:f, 1], label='L2')
            ax_L.plot(L_hist[sti:f, 2], label='L3')
            ax_L.set_xlabel("steps")
            ax_L.set_ylabel("L")
            ax_L.set_title("Control Moment")
            ax_L.legend()
            ax_L.grid(True)

            # LVLH frame
            lvlh_attitude = lvlh_attitude_hist[f]
            O1, O2, O3 = lvlh_attitude[:,
                                       0], lvlh_attitude[:, 1], lvlh_attitude[:, 2]
            ax_3d.quiver(*origin, *O1, color='tab:blue',
                         length=L_arrow, normalize=True)
            ax_3d.quiver(*origin, *O2, color='tab:blue',
                         length=L_arrow, normalize=True)
            ax_3d.quiver(*origin, *O3, color='tab:blue',
                         length=L_arrow, normalize=True)

            # Satellite frame
            q_sat_attitude = Quaternion(
                q_sat_hist[f].reshape(4, 1)).to_attitude()
            S1, S2, S3 = q_sat_attitude[:,
                                        0], q_sat_attitude[:, 1], q_sat_attitude[:, 2]
            ax_3d.quiver(*origin, *S1, color='tab:orange',
                         length=L_arrow, normalize=True)
            ax_3d.quiver(*origin, *S2, color='tab:orange',
                         length=L_arrow, normalize=True)
            ax_3d.quiver(*origin, *S3, color='tab:orange',
                         length=L_arrow, normalize=True)

            ax_3d.set_xlim([-1.5, 1.5])
            ax_3d.set_ylim([-1.5, 1.5])
            ax_3d.set_zlim([-1.5, 1.5])
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Z')
            ax_3d.set_title('Quaternion Control', fontsize=12)
            ax_3d.set_box_aspect([1, 1, 1])
            ax_3d.view_init(elev=20, azim=azim_angle[i])

            writer.grab_frame()
            print(f'Frame {i+1}/{len(frames)} generated')
