## Process for developing a Machine Learning Project

### Step 0:
To make it easier to export and classify the data for analysis, the data should be saved in the following structure: **participant/label/category/time_pattern_name_of_machine**
-> after exporting the data, we finally have 4 parameters to indent in the Data table: Participant, Label, Category and Set

### Step 1:
- Read the data file

**Important Note:**

- **data_merged.resample(ms).apply(sampling)**: Resampling in Pandas will process every 200ms interval across the entire dataset.
- If your data is not continuous ( exercises only 2 hours/day, 3 days/week), this may create unnecessary intervals and consume significant resources.
Solution:
- Resample only the periods Day 'D' where data exists, using dropna() to filter valid rows first **day = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]** and **pd.concat([df.resample("").apply(sampling).dropna() for df in day])**

## Step 2: Data visualisation
The process for visualizing data follows the steps in the diagram [Workflow_Data_Visualation](/document/image/Work_Flow_Data_Visualation.png):

At the Step 1: 3 Steps have done: Define Objective, colect raw data, data process
In this step: Focus in the remaining steps

**Step 1**: Load the pickel file

**Step 2**: Try to plot the digram of each **set** following the **index** with reset_index(drop=true)

**Step 3.1**: Try to plot the diagram for all data for  each **label** following **index**. Don't forget that label at the begin is object, muss change the type of the label to unique()
**Step 3.2**: Try plot all data for each **label** following **index** with **first 100 index**

**Step4**: Adjust plot settings with mpl

**Step5**: Plot multiple axis: Display all variables (e.g.: heavy and medium) of a parameter (e.g.: category) in a diagram.
**Note1**: If the diagram get messy, then use sort_values -> participant_df = df.query("label =='bench'").**sort_values**("participant").reset_index()
**Note2**: for the multiple axis: all_axis_df **[** "acc_x","acc_y","acc_x","acc_z","gyr_x","gyr_y","gyr_z" **]** is Series, can't plot -> all_axis_df **[[** "acc_x","acc_y","acc_x","acc_z","gyr_x","gyr_y","gyr_z" **]]**

**Step6**: Export the data for each label following each participant in PNG file

## Step 4: Detecting outliers in sensor data


