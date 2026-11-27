# Machine Learning

A collection of machine learning notebooks covering regression, multiclass classification and text-based classification with scikit-learn. Each notebook goes from data exploration to model evaluation.

## Projects

| Notebook | Task | Models | Main test result |
|---|---|---|---|
| [House price prediction](house_price_prediction.ipynb) | Regression | Linear Regression, Random Forest, SVR | R² 0.944, RMSE 17.93 k€ |
| [Credit decision](bank.ipynb) | 3-class classification | Decision Tree, Random Forest | Accuracy 97.8% |
| [Spam detection](spam.ipynb) | Binary classification | Multinomial Naive Bayes | Accuracy 92.6% |

## 1. House Price Prediction

Predicts a house price (in thousands of euros) from surface area, number of rooms, age, distance to the city centre and garage availability (150 samples).

**Features**

| Column | Description |
|---|---|
| `surface_m2` | Surface area in square metres |
| `nb_pieces` | Number of rooms |
| `age_annees` | Age of the house in years |
| `distance_centre_km` | Distance to the city centre in km |
| `garage` | Garage availability (1 = yes, 0 = no) |
| `prix_keuros` | Target: price in thousands of euros |

**Workflow**
- Exploratory analysis: correlation matrix, price distribution, surface vs. price
- Outlier inspection with the IQR method and boxplots
- 80/20 train/test split
- Preprocessing inside scikit-learn pipelines (median imputation, plus scaling for Linear Regression and SVR)
- 5-fold cross-validation and hyperparameter tuning with `GridSearchCV`
- Coefficient analysis, prediction on a new house, and model saving with `joblib`

**Cross-validation RMSE after tuning (k€, lower is better)**

| Model | CV RMSE |
|---|---|
| Linear Regression | 18.72 |
| Random Forest | 25.96 |
| Support Vector Regression | 27.05 |

Linear Regression was selected. On the test set it reaches an MAE of 13.56 k€, an RMSE of 17.93 k€ and an R² of 0.944. Tuning made a large difference for SVR, whose CV RMSE went from 73.95 to 27.05.

**Coefficients of the final model** (features are standardized, so each value is the price change in k€ for a one standard deviation increase, with the others held fixed)

| Feature | Coefficient |
|---|---|
| `surface_m2` | +78.44 |
| `garage` | +7.98 |
| `nb_pieces` | +2.26 |
| `age_annees` | -11.04 |
| `distance_centre_km` | -21.35 |

- Surface area has by far the largest effect. Price falls with distance from the centre and with age.
- The IQR rule found no price outliers.
- The split gives 120 training and 30 test samples.
- As an example, a 100 m², 4-room, 10-year-old house with a garage, 5 km from the centre, is predicted at about 271.57 k€.
- The final pipeline is saved as `house_price_model.pkl`, then reloaded to confirm it gives the same prediction.

## 2. Credit Decision Classification

Classifies a credit application as `Acceptation` (accepted), `Refus` (rejected) or `Risque élevé` (high risk), using 300 clients. The column names and class labels are in French.

**Features**

| Column | Description |
|---|---|
| `age` | Client age |
| `revenu_annuel` | Annual income |
| `anciennete_emploi` | Years of employment |
| `dette_totale` | Total debt |
| `ratio_dette_revenu` | Debt-to-income ratio |
| `nb_credits_en_cours` | Number of active loans |
| `retards_paiement` | Number of past late payments |
| `revenu_stable` | Stable income (`Oui` / `Non`) |
| `decision` | Target |

The `id_client` column is an identifier and is excluded from training.

**Class distribution:** `Risque élevé` 146 (48.7%), `Acceptation` 108 (36.0%), `Refus` 46 (15.3%). Because of this imbalance, the split is stratified.

**Data cleaning**

| Column | Missing | Treatment |
|---|---|---|
| `revenu_annuel` | 9.00% | Median |
| `anciennete_emploi` | 7.33% | Median |
| `revenu_stable` | 6.67% | Mode |
| `ratio_dette_revenu` | 5.00% | Recalculated for 13 of 15 rows, median for the other 2 |
| `age` | 4.00% | Median |

The income of 128,900 is also flagged by the IQR rule, but it is kept because it is plausible. Decision trees need no feature scaling. `revenu_stable` is encoded as 0/1.

**Workflow**
- Variable typing, class balance and descriptive statistics
- Missing value analysis and treatment of implausible values (an age of 140 and an income of 950,000)
- Imputation with median and mode; the missing debt-to-income ratios are recalculated from debt and income where possible
- Stratified 70/30 train/test split
- Decision trees (Gini vs. entropy), decision rules and a manual prediction walkthrough
- Overfitting analysis across tree depths, feature importance and `GridSearchCV`
- Comparison with a Random Forest

**Results**

| Model | Train accuracy | Test accuracy |
|---|---|---|
| Decision Tree (Gini, depth 3) | 94.8% | 97.8% |
| Tuned Decision Tree (depth 2) | 93.8% | 97.8% |
| Random Forest (100 trees) | 100% | 97.8% |

- The debt-to-income ratio and income stability account for about 96% of the depth-3 tree's feature importance.
- The tuned tree's 5-fold CV accuracy (93.8%) is a more cautious estimate than the score on the 90-client test set.
- Imputation statistics were computed on the full dataset before the split. A stricter setup would fit them on the training data only.

- On the test set, the depth-3 tree makes only 2 errors: two `Acceptation` clients predicted as `Risque élevé`. The Random Forest has an identical classification report.
- The root split is `ratio_dette_revenu <= 0.41` with both Gini and entropy.
- In the depth comparison, depths 3 and 4 give the best test accuracy. Training accuracy keeps rising with deeper trees while test accuracy drops, a sign of overfitting.
- The grid search covers 70 combinations of `criterion`, `max_depth` and `min_samples_leaf`. The best is Gini, depth 2, with at least 3 samples per leaf.
- The notebook describes the dataset as synthetic and easy to separate.

## 3. Spam Detection with Naive Bayes

Classifies emails as `Spam` or `Legitime` (legitimate) from word counts, the number of links and whether the sender is known (900 emails). Some feature names are in French.

**Features**
- `nb_liens`: number of links
- `expediteur_connu`: whether the sender is known
- 14 word counts: `gratuit`, `offre`, `urgent`, `gagner`, `cliquer`, `facture`, `reunion`, `projet`, `rapport`, `cordialement`, `loterie`, `heritage`, `seminaire`, `soutenance`
- `type_mail`: target (`Spam` or `Legitime`)

The `id_mail` column is excluded. The split gives 630 training and 270 test emails.

**Workflow**
- Exploratory analysis and class priors (64.1% legitimate, 35.9% spam)
- Stratified 70/30 train/test split
- The zero-frequency problem and the effect of Laplace smoothing, shown on a hand-built "trap" email
- Analysis of different `alpha` values
- Evaluation, most discriminative features and a check of the naive independence assumption
- Hyperparameter tuning and a manual verification of the model's scoring

**Results (alpha = 1)**

| Metric | Score |
|---|---|
| Train accuracy | 95.9% |
| Test accuracy | 92.6% |
| Spam precision | 87.4% |
| Spam recall | 92.8% |

- Smoothing leaves test accuracy unchanged compared with almost no smoothing, but it changes the prediction on the trap email, where a single unseen word would otherwise dominate the result.
- Cross-validated accuracy varies very little across the tested `alpha` values (94.9% to 95.2%).
- Flagging a legitimate email as spam is considered the more costly error, so spam precision is the main metric to improve.

**Confusion matrix (test set)**

| | Predicted legitimate | Predicted spam |
|---|---|---|
| **Actual legitimate** | 160 | 13 |
| **Actual spam** | 7 | 90 |

**Most discriminative features** (`log P(w | Spam) - log P(w | Legitime)`)

| Points to spam | Score | Points to legitimate | Score |
|---|---|---|---|
| `loterie` | +3.34 | `seminaire` | -3.13 |
| `heritage` | +3.34 | `soutenance` | -2.69 |
| `gratuit` | +1.82 | `rapport` | -2.47 |
| `gagner` | +1.63 | `reunion` | -2.37 |
| `cliquer` | +1.31 | `projet` | -1.89 |

- `facture` (+0.12) barely separates the two classes.
- `loterie` and `heritage` appear only in spam, while `seminaire` and `soutenance` appear only in legitimate emails. These words are rare, and their scores depend on the smoothing value.
- In the trap email, which contains `loterie` among professional words from a known sender, the nearly unsmoothed model predicts spam and the Laplace-smoothed model predicts legitimate.
- Grid search selects `alpha = 2`, but its advantage over `alpha = 1` amounts to about one email.
- The score `log P(C) + sum(x_i * log P(w_i | C))` is computed by hand for a new email and agrees with scikit-learn's prediction and probabilities.

## Getting Started

### Requirements

Python 3 with the following packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### Run the notebooks

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
jupyter notebook
```

Each notebook expects its dataset in the same folder:

| Notebook | Dataset file |
|---|---|
| `house_price_prediction.ipynb` | `houses.csv` |
| `credit_decision.ipynb` | `dataset_credit_TP3.csv` |
| `spam_detection.ipynb` | `dataset_spam_tp4.csv` |

## Note on the Results

These are learning projects on small datasets. The scores describe performance on these particular splits and should not be read as real-world performance.

## Author

Laouari Basmala
