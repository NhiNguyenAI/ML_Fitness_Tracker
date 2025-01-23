import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/data_process.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"]) 

# this case dont tell us, how much sample we have  -> reset_index, drop = true: covert timestamp to index
plt.plot(set_df["acc_y"].reset_index(drop = True))


# --------------------------------------------------------------
# Plot all exercises base the label
# --------------------------------------------------------------
# Changed the type object of label to unique


for label in df["label"].unique():
    subset = df[df["label"] == label]
    # display(subset.head(2))

    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop = True), label= label)
    plt.legend()
    plt.show()

# Just take a look on first 100 index
for label in df["label"].unique():
    subset = df[df["label"] == label]
    # display(subset.head(2))

    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop = True), label= label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams['figure.figsize'] = (20,5)
mpl.rcParams['figure.dpi'] = 100


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
categogy_df = df.query("label =='squat'").query("participant == 'A'").reset_index()
# plot both the variablen of category heavy and medium in one diagramm
fig, ax = plt.subplots()
categogy_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
# If the diagramm get messy -> use sort_values
participant_df = df.query("label =='bench'").sort_values("participant").reset_index()
# plot both the variablen of category heavy and medium in one diagramm
fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = "spuat"
participant = "A"
all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index
fig, ax = plt.subplots()

all_axis_df[["acc_x","acc_y","acc_x"]].plot(ax=ax) #all_axis_df["acc_x","acc_y","acc_x","acc_z","gyr_x","gyr_y","gyr_z"] 
ax.set_ylabel("acc_axis")
ax.set_xlabel("samples")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------