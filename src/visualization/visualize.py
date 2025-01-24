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
label = "squat"
participant = "A"
all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)  # Pass the axis object
ax.set_ylabel("Acceleration (g)")
ax.set_xlabel("Samples")
plt.legend(["acc_x", "acc_y", "acc_z"])  # Explicitly set legend labels
plt.title(f"Acceleration Data for Label: {label}, Participant: {participant}")
plt.show()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
       all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()

    fig, ax = plt.subplots()
    all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)  # Pass the axis object
    ax.set_ylabel("Acceleration (g)")
    ax.set_xlabel("Samples")
    plt.legend(["acc_x", "acc_y", "acc_z"])  # Explicitly set legend labels
    plt.title(f"{label}({participant})".title())
    plt.legend()
    plt.show() 

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
label = "row"
participant = "A"
combined_plot_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index(drop =True)

fig, ax= plt.subplots(nrows = 2, sharex = True, figsize = (20,10))
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax = ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax = ax[1])

ax[0].legend(loc = "upper center", bbox_to_anchor=(0.5,1.15), ncol = 3, fancybox = True, shadow=True)

ax[1].legend(loc = "upper center", bbox_to_anchor=(0.5,1.15), ncol = 3, fancybox = True, shadow=True)


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df_01 = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
        if len(combined_plot_df_01) > 0:
            ig, ax= plt.subplots(nrows = 2, sharex = True, figsize = (20,10))
            combined_plot_df_01[["acc_x", "acc_y", "acc_z"]].plot(ax = ax[0])
            combined_plot_df_01[["gyr_x", "gyr_y", "gyr_z"]].plot(ax = ax[1])

              # Add legends
            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            
            # Set axis labels
            ax[1].set_xlabel("Samples")
            ax[0].set_ylabel("Accelerometer Data")
            ax[1].set_ylabel("Gyroscope Data")

            save_path = f"../../reports/figures/{label.title()}_{participant}.png"
            plt.savefig(save_path)
            plt.show()
