
## ABSTRACT ##

Class imbalance can significantly impact the performance of learning algorithms, often leading to prediction bias toward the majority class. 
This challenge is particularly critical in healthcare-related domains, as medical datasets are often imbalanced, hindering the accurate prediction of the minority class, which is commonly the class of interest. 
As such, this work introduces a novel resampling algorithm, designated Genetic Beta Oversampling, which integrates user-defined preferences into the synthetic data generation process, allowing fine control over the model’s inclination towards false negatives or false positives. 
These user preferences are encoded in the form of a parameter, $\beta$, which dictates the trade-off between recall and precision that the method should seek to achieve. 
This flexibility is particularly relevant in clinical settings, where prioritizing recall can enhance patient care by reducing missed diagnoses. 
We evaluate the proposed approach on the EMPA and AI4PAIN datasets for pain classification, a recall-critical task in which undetected pain episodes must be minimized. Experimental results show that our method consistently surpasses SMOTE, SMOTE-IPF, and four cost-sensitive classifiers in terms of F$\beta$-score across a range of $\beta$ values and synthetically induced imbalance ratios. These results highlight the adaptability of the method to recall-sensitive applications, including pain assessment and broader clinical decision-support scenarios.

# Genetic Beta Oversampling

This repository contains code for **genetic optimization of data resampling strategies** in highly imbalanced **physiological pain classification**, evaluated on two datasets:

- **AI4PAIN**: Multimodal physiological signals (BVP, EDA, respiration, SpO₂) for pain detection.
- **EMPA**: A cold pressor test (CPT) dataset with baseline and nociceptive stimulation windows.

At the core is the **Genetic_BETA_resampling framework**, a simplex-based oversampling / hybrid resampling method that optimizes the **F-β score** via a genetic algorithm, and plugs into the `smote_variants` oversampling API.

This work is built upon the `smote_variants` library ([https://github.com/analyticalmindsltd/smote\_variants](https://github.com/analyticalmindsltd/smote_variants)), and we extend our gratitude to the original authors for their excellent work.

-----


## 1. Overview

Class imbalance is a central challenge in pain prediction: clinically relevant pain states tend to be rare compared to “no pain” or baseline segments. This repo addresses that by:

1. **Learning sample-group–specific resampling distributions** over:
   - *Instance hardness / local neighbourhood structure* (via `pmf_local_neighbourhood`).
   - *Proximity to majority examples* (ProWSyn-style, via `pmf_prowsyn`).

2. **Optimizing these distributions with a genetic algorithm** to directly maximize cross-validated **F-β** on the training set.

3. **Evaluating against standard resampling and classifier baselines**:
   - No resampling, SMOTE, SMOTE-IPF.
   - Random Forest, SVC, Extra Trees, AdaBoost, etc., with class-weighting.

The framework is implemented as a subclass of `OverSamplingSimplex` from `smote_variants`, and is fully compatible with scikit-learn pipelines.

---

## 2. Repository Structure

```text
.
├── AI4PAIN_analysis.py        # Experiments on the AI4PAIN multimodal dataset
├── EMPA_analysis.py           # Experiments on the EMPA CPT dataset
├── smote_variants/
│   └── oversampling/
│       └── _genetic_beta_resampling.py   # Genetic_BETA_resampling framework
└── data/
    ├── bvp_*train*.xlsx / bvp_*val*.xlsx
    ├── eda_*train*.xlsx / eda_*val*.xlsx
    ├── resp_*train*.xlsx / resp_*val*.xlsx
    ├── spo2_*train*.xlsx / spo2_*val*.xlsx
    └── Dataset_EMPA.xlsx
````

**Note:** Folder layout may be slightly different in your repo; adjust the paths in the README accordingly.

---

## 3. Core Method: Genetic βeta Oversampling

The main method is implemented in:

* `smote_variants/oversampling/_genetic_beta_resampling.py`

  * `Genetic_BETA_resampling_framework`
  * `Genetic_BETA_resampling_framework_v2`
  * `pmf_local_neighbourhood`, `pmf_prowsyn`
  * `generate_samples_local_neighboorhood`, `generate_samples_prowsyn`

### 3.1 Intuition

1. **Partition minority samples into groups**

   * `pmf_local_neighbourhood`: groups each minority sample by the number of majority neighbours in its *k*-NN neighborhood (local hardness).
   * `pmf_prowsyn`: captures how often each minority sample is selected as a neighbour of majority samples (ProWSyn-inspired proximity).

   Each function returns a vector of **group IDs** over the minority class plus additional bookkeeping variables.

2. **Define a gene for each group**

   For each PMF, a set of groups (with enough samples) is detected. Each group corresponds to one gene controlling:

   * The **relative amount of synthetic data** to generate from that group; or
   * In “removal mode”, how many majority examples associated with that group to remove.

3. **Genetic optimization**

   A `pygad` GA searches over:

   * Distribution over PMF1 groups.
   * Distribution over PMF2 groups.
   * A **main/secondary distribution ratio** (how much weight to give PMF1 vs PMF2).
   * Optionally, a parameter for **minority outlier removal**.

   For each chromosome:

   * Cross-validated resampling (StratifiedKFold).
   * Train a `RandomForestClassifier` on the resampled training set.
   * Evaluate with **F-β** on held-out folds.
   * Fitness = mean F-β across folds.

4. **Final resampling**

   Once the best chromosome is found, the framework uses it to:

   * Optionally remove some majority instances (if `func2_toremove=True`).
   * Generate synthetic minority samples from the learned group distributions.
   * Return a resampled `(X_resampled, y_resampled)` ready for downstream modeling.

### 3.2 Usage as an Oversampler

The method is used through a simple helper in the analysis scripts:

```python
from smote_variants.oversampling import SMOTE, SMOTE_IPF
from smote_variants.oversampling._genetic_beta_resampling import (
    Genetic_BETA_resampling_framework_v2,
    pmf_local_neighbourhood,
    pmf_prowsyn,
    generate_samples_local_neighboorhood,
    generate_samples_prowsyn,
)

def resample_data(X_train, y_train, method, beta=None):
    if method is None:
        return X_train, y_train, None
    if method == Genetic_BETA_resampling_framework_v2:
        sol, X_resampled, y_resampled = method(
            Beta=beta,
            func1_toremove=False,
            func2_toremove=False,
            func1_pmf=pmf_local_neighbourhood,
            func2_pmf=pmf_prowsyn,
            func1_generate_samples=generate_samples_local_neighboorhood,
            func2_generate_samples=generate_samples_prowsyn,
        ).resample(X_train, y_train)
        return X_resampled, y_resampled, sol
    X_resampled, y_resampled = method(n_jobs=16).fit_resample(X_train, y_train)
    return X_resampled, y_resampled, None
```

You can swap in your own `pmf_*` and `generate_samples_*` functions as long as they follow the documented interface.

---

## 4. Experimental Pipelines

### 4.1 Common Preprocessing

Both AI4PAIN and EMPA experiments use a scikit-learn `Pipeline` that:

1. Encodes categorical features using:

   * `SimpleImputer(strategy="most_frequent")` + `OrdinalEncoder` on categorical columns.
2. Drops high-missingness columns (`DropHighNaNs(threshold=0.2)`).
3. Imputes numeric features (`KNNImputer`).
4. Removes low-variance features (`VarianceThreshold`).
5. Normalizes features (`StandardScaler`).
6. Performs embedded feature selection with a `RandomForestClassifier` via `SelectFromModel` (max ~100 features).

This produces a dense feature matrix `X` and binary labels `y` (0 = majority, 1 = minority).

### 4.2 AI4PAIN Experiments (`AI4PAIN_analysis.py`)

**Data loading**

* Expects **per-modality** `.xlsx` files in `./data`:

  * `bvp_*train*.xlsx`, `bvp_*val*.xlsx`
  * `eda_*train*.xlsx`, `eda_*val*.xlsx`
  * `resp_*train*.xlsx`, `resp_*val*.xlsx`
  * `spo2_*train*.xlsx`, `spo2_*val*.xlsx`
* For each modality:

  * Concatenates train + val.
  * Uses `class` as label, `participant_id` as subject ID.
  * Drops a known outlier row (index 1093) in non-SpO₂ modalities.
* All modality feature matrices are horizontally concatenated to form a multimodal design matrix.

**Label handling**

* Labels are taken from the `class` column.
* One label (with value 2) is excluded to form a **binary** classification (majority vs minority).

**Class imbalance**

* Imbalance is created via `create_class_imbalance(X, y, minority_ratio)` with:

  * `minority_class = 1`
  * `majority_class = 0`
* Default `--imbalance_ratios`: `0.05 0.1 0.2`.

**Evaluation**

* For each imbalance ratio and each β in `--betas` (default `1 2.5 5 7.5 10`):

  * Stratified K-fold CV (`--n_splits`, default 10).
  * Resampling methods:

    * `NoResampling`
    * `SMOTE`
    * `SMOTE_IPF`
    * `Genetic_BETA_resampling_framework_v2`
  * Classifiers:

    * `RandomForestClassifier`
    * `SVC`
    * `ExtraTreesClassifier`
    * `AdaBoostClassifier` (with a depth-5 `DecisionTreeClassifier` base learner).

* Metrics:

  * Primary: **F-β** (symmetric and cost-sensitive variants via β).
  * Secondary: **precision**, **recall**, **TPR/TNR**.

All per-ratio, per-β results are summarised and written to a CSV (`--output_file`).

### 4.3 EMPA Experiments (`EMPA_analysis.py`)

**Data loading & selection**

* Expects `Dataset_EMPA.xlsx` under `./data/`.
* Filters the data to:

  * Baseline windows (`Label == "Baseline"`) and
  * CPT windows (`Label == "CPT"`) whose time window starts *after* the reported pain onset (`(Window * 5) * 1000 > Time_Pain`).
* Drops `Time_Pain` and `Time_Tolerance` columns.
* Encodes labels using `LabelEncoder`, then applies the same preprocessing pipeline as in AI4PAIN.

**Class imbalance & evaluation**

* Same workflow as AI4PAIN:

  * `create_class_imbalance` with a specified minority ratio (default: `0.33, 0.05, 0.1, 0.2`).
  * CV, resampling methods, classifiers, and metrics identical to AI4PAIN.
  * Results are aggregated and written to CSV.

---

## 5. Command-Line Interface

Both analysis scripts share the same CLI signature:

```bash
python AI4PAIN_analysis.py \
  --imbalance_ratios 0.05 0.1 0.2 \
  --betas 1 2.5 5 7.5 10 \
  --n_splits 10 \
  --output_file ai4pain_results.csv

python EMPA_analysis.py \
  --imbalance_ratios 0.33 0.05 0.1 0.2 \
  --betas 1 2.5 5 7.5 10 \
  --n_splits 10 \
  --output_file empa_results.csv
```

### 5.1 Arguments

* `--imbalance_ratios`: list of target **minority class fractions** (e.g., `0.05` = 5% minority).
* `--betas`: list of β values for the F-β score.
* `--n_splits`: number of CV folds (StratifiedKFold).
* `--output_file`: destination CSV file for aggregated results.

---

## 6. Installation & Dependencies

Create an environment (e.g., with `conda` or `venv`) and install:

```bash
pip install numpy pandas scikit-learn smote-variants pygad scipy openpyxl
```

**Core dependencies**

* Python 3.x
* `numpy`, `pandas`
* `scikit-learn`
* `smote_variants`
* `pygad`
* `scipy`
* `openpyxl` (for `.xlsx` IO)

If this repository is part of a larger `smote_variants` installation, ensure it is on your `PYTHONPATH` or installed in editable mode:

```bash
pip install -e .
```

---

## 7. Extending the Framework

### 7.1 Plug in your own PMFs

To experiment with alternative sample grouping strategies:

1. Implement a PMF function:

```python
def my_pmf(instance, X, y, to_remove, **kwargs):
    # Return:
    #   groups: (n_minority,) array of ints
    #   aux_vars: list/tuple of any extra structures required by your generator
    return groups, aux_vars
```

2. Implement a compatible sample generator:

```python
def my_generate_samples(instance, X, y, pmf, aux_vars, sample_group, n_to_sample):
    # Return:
    #   samples: (n_to_sample, n_features) numpy array
    return samples
```

3. Construct the oversampler:

```python
my_oversampler = Genetic_BETA_resampling_framework_v2(
    Beta=beta,
    func1_pmf=my_pmf,
    func1_generate_samples=my_generate_samples,
    # optionally set func2_* as well
)
X_res, y_res = my_oversampler.resample(X, y)[1:]
```

### 7.2 Changing the fitness model or metric

The GA currently uses:

* `RandomForestClassifier(n_estimators=10, max_depth=5)` as the model.
* `fbeta_score` with the user-specified β as the objective.

You can modify `fitness_function` in `_genetic_beta_resampling.py` to:

* Swap in a different classifier (e.g., SVC, XGBoost).
* Optimize a different metric (AUPRC, balanced accuracy, etc.).
* Use a multi-objective strategy (e.g., combining F-β with calibration).

---

## 8. Reproducibility Notes

* Random seeds are set in:

  * `create_class_imbalance` (`random_seed=342`).
  * `StratifiedKFold(shuffle=True, random_state=1)` in the resampling framework.
  * RandomForest and other classifiers via `random_state=1` where applicable.
* The GA uses `parallel_processing=16`. You may want to adjust this to match your hardware.

For strict reproducibility, ensure:

* Fixed library versions (e.g., via `requirements.txt` or `conda env`).
* Single-threaded BLAS / OpenMP if necessary for bit-level determinism.

---

## 9. Citation

WIP

---

## 10. License

Specify your license here (e.g., MIT, Apache-2.0). For example:

> This project is licensed under the MIT License – see the `LICENSE` file for details.



