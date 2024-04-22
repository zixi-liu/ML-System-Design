### Data Structures

-- type system

### Data Preprocessing

1. Dealing with Missing Values

- Missing Data Types
   - Missing Completely at Random (MCAR) -> deleting rows or columns
   - Missing at Random (MAR) -> imputation of data
   - Missing not at Random (MNAR) -> improve dataset to find data
- Types of Imputations
   - Univariate imputation -> mean, median, mode imputation
   - Multivariate imputation -> LR based imputation
   - Single imputation
   - Numerous imputations

2. Data Standardization/Feature Scaling
   - standardization
   - normalization
   - min-max scaling
   - Why feature scaling? Because gradient descent require data to be scaled. Distance-base algorithms rely on distances between data points to determine similarity.
   - Normalize or Standardize?
     - Normalize is relatively sensitive to outliers, useful is distribution is unknown or not Gaussian, retains the shape of original distribution
     - Normalization equation: (x-min)/(max-min)
     - Standardization equation: (x-mean)/standard deviation

 3. Handling Noisy Data
    - Binning
      - Smoothing by bin mean/median/bin boundary

### Gradient Descent for Linear Regression

- Loss Function
  - Sum of squares error
 
- Gradient Descent
  - 
