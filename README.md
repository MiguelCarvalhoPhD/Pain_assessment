## Genetic Beta Resampling Framework (GBRF)

This repository contains the official implementation of the **Genetic Beta Resampling Framework (GBRF)** and the experiments conducted for the paper. GBRF is a novel, customizable, and computationally efficient resampling framework that integrates user preferences directly into the synthetic data generation process.

This work is built upon the `smote_variants` library ([https://github.com/analyticalmindsltd/smote\_variants](https://github.com/analyticalmindsltd/smote_variants)), and we extend our gratitude to the original authors for their excellent work.

-----

### Abstract

Resampling techniques are widely used by researchers and practitioners to address class imbalance due to their adaptability across diverse classification tasks. However, they inherently lack the ability to enforce user-defined preferences regarding model behavior after training, a feature typically exclusive to cost-sensitive learning frameworks or prediction post-processing techniques. This limitation is particularly critical in high-stakes applications, such as in medical applications, where maximizing minority class accuracy while minimizing false negatives is essential. To overcome this constraint, we introduce the Genetic Beta Resampling Framework (GBRF), a novel, customizable and computationally efficient resampling framework that integrates user preferences into the process of synthetic data generation. GBRF leverages Genetic Algorithms to optimize two probability mass functions (PMFs) that govern the sampling probabilities of different instance groups, enabling synthetic data generation and/or instance removal. Consequently, GBRF can function as an hybrid sampling, oversampling or undersampling technique. User preferences are encoded through a parameter, $ \\beta $, which controls the trade-off between precision and recall. Comprehensive experiments on 60 OpenML datasets demonstrate that GBRF effectively embeds user preferences into data distributions, thus shaping model behavior accordingly. It consistently outperforms state-of-the-art resampling techniques, such as SMOTE-IPF and ProWSyn, as well as cost-sensitive classifiers, even when integrated with various classification models. Furthermore, by employing a non-instance-wise genetic optimization approach, GBRF significantly reduces the search space, achieving faster convergence to optimal solutions. Finally, since synthetic data generation is governed by two PMFs, the framework provides an intuitive and transparent mechanism for understanding how data is generated.

-----

### 🚀 Features

  * **Preference-Driven Resampling**: Embeds user-defined preferences (e.g., balancing precision and recall) directly into the data generation process using the $\\beta$ parameter.
  * **Flexible**: Can be configured to perform oversampling, undersampling, or hybrid sampling.
  * **Efficient**: Utilizes a non-instance-wise genetic optimization to find optimal sampling strategies quickly.
  * **Interpretable**: The use of two Probability Mass Functions (PMFs) provides a clear view of how synthetic data is generated.
  * **Customizable**: Allows for user-defined sample generation and instance grouping functions.

-----

### 📂 Repository Structure

    .
    ├── smote_variants/              # Core GBRF implementation and competing methods
    ├── Example_usage.py             # Simple demo on synthetic data
    ├── Performance_testing.py       # Main script to run all benchmarks
    ├── Results_analysis.py          # Script to process results and generate tables
    ├── requirements.txt             # Required Python packages
    ├── LICENSE                      # Open-source license
    └── README.md                    # You are here!

-----

### 🛠️ Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/GBRF.git
    cd GBRF
    ```

2.  **Create and Activate a Virtual Environment (Recommended)**

    ```bash
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

-----

### 📈 Usage

#### 1\. Replicating Paper Experiments

To replicate the benchmarking experiments from the paper, you can use the `Performance_testing.py` script. The script requires a directory an .npy file containing all datasets in a 1D array (name beginning with dataset), and a .npy file containing all target variable for all datasets in a 1D array (name beginning with target). 

The OpenML datasets used in the paper are listed below. You will need to download them and place them in the specified `--data-dir` under the aforementioned formatting.

  * **Dataset Information**:

      * **Binary Datasets (OpenML IDs)**: `sick` (38), `hepatitis` (55), `oil_spill` (311), `scene` (312), `yeast_ml8` (316), `SPECT` (336), `SyskillWebertSheep` (376), `analcatdata` (450, 463, 465, 479, 728, 747, 757, 760, 764, 765, 767, 852, 865, 867, 875), `backache` (463), `balloon` (914), `socmob` (934), `water-treatment` (940), `arsenic` (947, 949, 950, 951), `spectrometer` (954), `braziltourism` (957), `segment` (958), `mfeatmorphological` (962).
      * **Multiclass Datasets (OpenML IDs)**: `page-blocks` (30.0–30.4), `abalone` (183.1–183.10, 183.12–183.15, 183.21–183.27). Note: IDs with a "." indicate specific binary decompositions of the multiclass datasets.

  * **Running the Script**:

    ```bash
    # Create a directory for results
    mkdir -p ~/gbrf_results

    # Run the experiment script
    python Performance_testing.py \
        --betas 0.2 1.0 5.0 \
        --data-dir /path/to/your/datasets/ \
        --output-dir ~/gbrf_results/
    ```

      * `--betas`: A space-separated list of $\\beta$ values to evaluate.
      * `--data-dir`: The directory containing your dataset files (e.g., `dataset_X.npy`, `dataset_y.npy`).
      * `--output-dir`: The directory where results (Excel and pickle files) will be saved.

#### 2\. Using GBRF on Your Own Datasets

You can easily apply GBRF to your own projects. The `Example_usage.py` script provides a simple demonstration.

The core of the framework is the `GBRF` class. Here's a minimal example:

```python
from smote_variants.oversampling import *
from smote_variants._logger import logger
from sklearn.datasets import make_classification

# 1. Generate some imbalanced data
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                           n_redundant=0, n_classes=2, n_clusters_per_class=1,
                           weights=[0.95, 0.05], flip_y=0, random_state=42)

# 2. Instantiate GBRF
# We set beta > 1 to prioritize recall
beta = 5
gbrf = Genetic_BETA_resampling_framework(Beta=beta, # determining user preference
  func1_toremove=False, 
  func2_toremove=True, # func1_toremove args dictate that the method should work as hybrid sampling
  func1_pmf=pmf_local_neighbourhood, # defining sample grouping 1
  func2_pmf=pmf_prowsyn, # defining sample grouping 2
  func1_generate_samples=generate_samples_local_neighboorhood, # defining sample generation function 1
  func2_generate_samples=generate_samples_prowsyn, # defining sample generation function 2
  outlier_removal=False) # defining whether to do outlier removal

# 3. Resample the data
solution, X_resampled, y_resampled = gbrf.resample(X, y)

```

#### 3\. Customizing GBRF

GBRF is highly customizable. You can define your own functions for **sample grouping** and **sample generation**.

  * **Custom Sample Grouping Function**: This function must return a 1D integer array where each element is the group index for a minority class sample.

      * **Example**: For 5 minority samples split into 5 individual groups, the output should be `[0, 1, 2, 3, 4]`.

  * **Custom Sample Generation Function**: This function must return a NumPy array of shape `(n_samples, n_features)` containing the newly generated synthetic samples.

See the functions `generate_samples_prowsyn`, `pmf_prowsyn`, `generate_samples_local_neighborhood`, and `pmf_local_neighborhood` in the `smote_variants/` directory for implementation examples.

-----

### 📄 Citation

If you use GBRF in your research, please cite our paper:

```bibtex
To be finished
}
```

-----

### 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.