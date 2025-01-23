## Process for developing a Machine Learning Project

### Step 1: 
- Read the data file

**Important Note:**

- **data_merged.resample(ms).apply(sampling)**: Resampling in Pandas will process every 200ms interval across the entire dataset.
- If your data is not continuous ( exercises only 2 hours/day, 3 days/week), this may create unnecessary intervals and consume significant resources.
Solution:
- Resample only the periods Day 'D' where data exists, using dropna() to filter valid rows first **day = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]** and **pd.concat([df.resample("").apply(sampling).dropna() for df in day])**

## Step 2: Data visualisation
The process for visualizing data follows the steps in the diagram [Workflow_Data_Visualation](image/Work_Flow_Data_Visualation.png): 