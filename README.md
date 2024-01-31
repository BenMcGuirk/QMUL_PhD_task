Your objective is to investigate the relationship between mammographic density and risk of breast cancer in this case-control study from the IBIS-I trial. In particular:

•	Summarise each variable in the data using relevant measures.
•	Fit a logistic regression model with age (continuous), body mass index (continuous) and breast density (continuous), and report results.
•	Apply one further machine learning or statistical method to model the same data (age, body mass index, breast density and breast cancer status). 
•	Outline why you chose this method, and report the model fit and describe the results.
Compare your model results with those from logistic regression, if there are any differences, and if so potential reasons.
 
Your analysis should be presented either as 

(1) an analysis notebook (eg. a jupyter notebook, or Rmarkdown); or
(2) in a document (PDF) with your analysis code provided as an appendix.

The text in your report should not be more than 1,000 words. You may include up to 4 figures / tables. The figures/tables should be clear and consistently labelled with appropriate, intelligible, legends.


# Project Structure

- **data/:**
  - raw/                  # Raw data files (e.g., CSV, Excel)
  - processed/            # Cleaned and preprocessed data

- **notebooks/:**
  - preprocessing/        # Jupyter notebooks for data preprocessing
  - modeling/             # Jupyter notebooks for model development

- **src/:**
  - data_preprocessing/   # Python scripts/modules for data preprocessing

- **reports/:**
  - figures/              # Visualizations and plots
  - documentation/              # Project summary reports or documentation

- **requirements.txt:**
  - Python dependencies for the project

- **README.md:**
  - Project documentation