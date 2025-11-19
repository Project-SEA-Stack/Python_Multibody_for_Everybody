import pandas as pd
import numpy as np
import os

# Load the tidy CSV produced by your C++ loop by absolute path
folder      = "yaml_rotor_pendulum"  # Change to your folder name
abs_path    = os.getcwd() + "\Validation2externalSoft\\results"
base_dir    = os.path.join(abs_path, folder, "trajectory.csv")

df = pd.read_csv(base_dir)

# Keep things sorted and consistent
df = df.sort_values(["time", "body_index"]).reset_index(drop=True)

# Build wide matrices (time x bodies) for x,y of position and velocity
t  = df.pivot(index="time", columns="body_index", values="time").to_numpy()[:,0]
rx = df.pivot(index="time", columns="body_index", values="pos_x").to_numpy()
ry = df.pivot(index="time", columns="body_index", values="pos_z").to_numpy()
vx = df.pivot(index="time", columns="body_index", values="vel_x").to_numpy()
vy = df.pivot(index="time", columns="body_index", values="vel_z").to_numpy()

# Optional sanity check: ensure no missing samples
assert not (np.isnan(rx) | np.isnan(ry) | np.isnan(vx) | np.isnan(vy)).any(), "NaNs found—check sampling!"

# Stack into (time_steps, bodies, dof=2)
r = np.stack([rx, ry], axis=-1)   # positions
v = np.stack([vx, vy], axis=-1)   # velocities

# Save as requested
dir2save = abs_path + "\\" + folder
np.save(dir2save + "\\t_chrono.npy", t)
np.save(dir2save + "\\r_chrono.npy", r)
np.save(dir2save + "\\v_chrono.npy", v)

print("Saved:\n",
      folder + "\\r_chrono.npy", r.shape, "\n",
      folder + "\\v_chrono.npy", v.shape)

