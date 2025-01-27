# Process for Developing a Machine Learning Project

## Step 0: Data Preparation
To facilitate exporting and classifying data for analysis, organize your data using the following structure:
**participant/label/category/time_pattern_name_of_machine**

After exporting the data, ensure the following parameters are included in the Data Table:
- **Participant**
- **Label**
- **Category**
- **Set**

## Step 1: Reading the Data File

### Important Notes:
1. **Resampling Data:**
   - Use the Pandas function `data_merged.resample(ms).apply(sampling)` to resample data at specific intervals (e.g., 200ms).
   - **Caution:** If your data is not continuous (e.g., exercises only 2 hours/day, 3 days/week), this method may create unnecessary intervals and consume significant resources.

   **Solution:**
   - Resample only periods (days) where data exists:
     ```python
     day = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]
     pd.concat([df.resample("200ms").apply(sampling).dropna() for df in day])
     ```

-----------------------------------------------------------------------------------------------------------------------------------------------------------

## Step 2: Data Visualization
Follow the steps outlined in the workflow diagram: **[Workflow_Data_Visualization](/document/image/Work_Flow_Data_Visualation.png)**

### **Context:**
- In **Step 1**, you have already:
  1. Defined Objectives
  2. Collected Raw Data
  3. Processed the Data
- Focus on the remaining steps in this stage.

### **Steps:**
1. **Load the Pickle File**:
   Load the processed data stored in pickle format.

2. **Plot Individual Diagrams**:
   - Plot diagrams for each **set** based on the **index**.
   - Reset the index with `reset_index(drop=True)` for clarity.

3. **Label-wise Visualization**:
   - **Step 3.1:** Plot diagrams for all data grouped by each **label** following the **index**. 
     - Convert the `label` column from object to unique categories if necessary.
   - **Step 3.2:** Plot all data for the first 100 indices for each **label**.

4. **Adjust Plot Settings**:
   Configure settings using `matplotlib` (mpl) for cleaner visuals.

5. **Multiple Axis Visualization**:
   - Display all variables (e.g., heavy, medium) of a parameter (e.g., category) in one diagram.
   - **Note 1:** If the diagram appears cluttered, use `sort_values`:
     ```python
     participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()
     ```
   - **Note 2:** When plotting multiple axes:
     - **Incorrect:** `all_axis_df["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]` (returns Series).
     - **Correct:** `all_axis_df[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]]` (returns DataFrame).

6. **Export Visualizations**:
   Save the plotted data as PNG files for each **label** and **participant**.

-----------------------------------------------------------------------------------------------------------------------------------------------------------


## Step 3: Detecting Outliers in Sensor Data

### **Step 1: Plotting Outliers with plotbox()**
- Create boxplots for accelerometer (`acc_x`, `acc_y`, `acc_z`) and gyroscope (`gyr_x`, `gyr_y`, `gyr_z`) data:
  - **Individual Columns**: Plot one column at a time.
  - **All Columns**: Combine all columns into a single plot.
- **Note:** Configure the plot settings:
  ```python
  plt.style.use("seaborn-v0_8-deep")
  plt.rcParams["figure.figsize"] = (20, 5)
  plt.rcParams["figure.dpi"] = 100


### Step 2: Interquartile Range (IQR) with mark_outliers_iqr() and plot_binary_outliers()

#### Important Notes:
1. **DataFrame vs Series:**
   - `type(df[["acc_x"]])` returns a **DataFrame**.
   - `type(df["acc_x"])` returns a **Series**.
   - Use this distinction carefully when plotting boxplots to ensure compatibility with plotting functions.

2. **Handling `plot_binary_outliers`:**
   - **Be mindful of the `reset_index` parameter:**
     - **Incorrect:** Setting `reset_index=False` will use the **timestamp** on the x-axis, which can clutter the visualization.
     - **Correct:** Set `reset_index=True` to improve the clarity of the plot and make the visualization more interpretable.

### Step 3: Chauvenets criteron (distribution based)
Problem: Looking the data on basically own one big pile. Some data in IQR is extrem higher oder lower. For Example in this case is ar the break time,  people can do everthing, that result the data of the watch sometime not like in the exercise like squat

Normal distribution
It's important to note that Chauvenet's criterion is only applicable to datasets that are normally distributed. If your dataset is not normally distributed, this method may not be suitable for identifying outliers.

Histogram — Do you see a bell shaped curve?
Boxplot — Is the box symmetrical?
1. check the data is normal distributed? Use plot.hist in the step 1
2. Use the Chauvenet's function

### Step 4: Local outlier factor
**Document for outlier method:**  In this File will be explain, was is different between 3 outliers methode IQR, Schauvenet and Local outlier Factor [Outlier_method](\Outliers_Methods.pdf)


### Step 5: Compare 3 method

### Step 6: Choose methof and deal with outliers
- Test on single column
- Create a loop for remove all outliers value in the dataset
- Export new dataframe
**Note:**  # update the column in the original dataframe
       **outliers_removed_df.loc[(outliers_removed_df["label"] == label), col] = mark_outliers_chavenet_dataset[col]**

-----------------------------------------------------------------------------------------------------------------------------------------------------------


## Step 4: Feature Engineering

*** Step1** Load data from the step 3




