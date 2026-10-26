# What sets a loan's interest rate? A regression analysis

Can a borrower's attributes predict their loan's interest rate?

**The short answer: Only weakly.** Models using the observable features only reached an R²-score of about 0.27 to 0.30. The rate is almost entirely captured by loan grade and grade itself is only partly recoverable from said features.

**Limitation:** The dataset is simulated, so the findings describe the simulation and not real borrowers, and real-world relationships may differ.

## Findings

- Linear, Ridge and Lasso all reach R² = 0.27 (MAE 2.21 vs 2.68 baseline). A GBM reaches 0.30, so the limit most likely lies in the features, and not the model.
- The models essentially rely on one feature: a prior default on file adds 4.1 percentage points to the rate. This creates two clusters in the prediction. Home ownership and loan size make small adjustments, but not relevant ones.
- Predicting the rate from grade alone: R² = 0.87, MAE 0.90. Adding the other observable features: MAE unchanged at 0.90. So features don't add measurable information.
- Logistic regression recovers grade with 47% accuracy from the observables, compared to 33% for predicting the most common grade. So some information lies in the observables, but not all.

## Results

| model         |      r2 |   r2_std |   rmse |    mae |
|:--------------|--------:|---------:|-------:|-------:|
| mean baseline | -0.0003 |   0.0002 | 3.2295 | 2.6815 |
| linear        |  0.2739 |   0.0067 | 2.7516 | 2.2109 |
| ridge         |  0.2739 |   0.0067 | 2.7516 | 2.2109 |
| lasso         |  0.2739 |   0.0066 | 2.7516 | 2.2109 |
| gbm           |  0.3024 |   0.0111 | 2.6969 | 2.1673 |

5-fold cross-validation, loan grade excluded

*See `notebooks/03_grade.ipynb` for the models that include grade.*

## Data

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) from Kaggle:
32,581 loans of simulated credit-bureau data in the style of US consumer lending.

**Cleaning removes 3,949 rows (about 12%):** Rows with missing interest rates and employment length (both missingnesses look random), rows with impossible ages (kept `age <= 100`) or employment lengths (kept `emp_length <= age - 14`, which assumes nobody works before age 14, using 16 does not change the results).
Income and loan amount are log-transformed, which is common practice for skewed monetary values.

Grade and rate carry very similar information and the data can't show which determines which. In the real-world lending, the grade is assigned first and the rate follows from it, so I am treating grade as an input to the rate.

## About this project

**Approach.** The project started as a single-notebook analysis (`notebooks/00_linear_regression.ipynb`), with a simple linear regression of interest rate on borrower features. Weak results led to more follow-up questions (is the model the problem, or the features? What drives the rate?). To answer them, I compared regularized (Ridge, Lasso) and gradient-boosted models, and analysed loan grade separately. The original notebook 00 is kept, as a baseline, so the full path it visible. The later notebooks supersede the numbers where the notebooks differ. 


**Process note.** The overall analysis is my own work. I used Claude to help with the code and restructuring into a multi-notebook project as a learning exercise. 
I wrote up and checked the findings myself, where each branch (e.g. employment length vs age) is a note of interest i wanted to pursue further.

## Structure

```
notebooks/
  00_linear_regression.ipynb   starting point, kept as a starting point, retrospective note was added at the end
  01_eda.ipynb               cleaning decisions, target and feature distributions
  02_models.ipynb            baseline, linear, Ridge, Lasso, GBM; coefficients; residuals
  03_grade.ipynb             how much grade explains, and whether it can be predicted
src/
  data.py                    load and clean
  features.py                preprocessing pipeline
  evaluate.py                cross-validation report
scripts/download_data.sh     fetches the dataset via the Kaggle API
```

## How to run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# needs a Kaggle API token in ~/.kaggle/kaggle.json
./scripts/download_data.sh
jupyter lab
```