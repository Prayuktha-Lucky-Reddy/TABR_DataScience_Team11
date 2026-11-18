# TabR: Tabular Deep Learning Meets Nearest Neighbors

## Overview

This project presents an implementation and experimental evaluation of **TabR (Tabular Deep Learning Meets Nearest Neighbors)**, a retrieval-augmented deep learning model for tabular data.

TabR combines neural networks with nearest-neighbor retrieval. Instead of making predictions using only the learned representation of an input sample, the model retrieves similar samples from the training dataset and uses their information to improve its predictions.

The project focuses on reproducing the baseline performance of TabR across multiple tabular datasets and studying the effects of different retrieval context sizes and input noise.

**Reference Paper:**  
*TabR: Tabular Deep Learning Meets Nearest Neighbors* – ICLR 2024.

## Key Features

- Implementation of a retrieval-augmented neural network for tabular data.
- Support for regression, binary classification, and multiclass classification.
- Nearest-neighbor retrieval using negative squared L2 distance.
- Comparison of model performance with results reported in the original paper.
- Experiments with different retrieval context sizes.
- Robustness evaluation under Gaussian input noise.
- Evaluation across eight tabular datasets.

## Model Architecture

TabR combines neural representation learning with nearest-neighbor retrieval.

The main components are:

1. **Input Representation:** Numerical and categorical features are transformed into a representation suitable for the model.
2. **Query and Key Representations:** The model generates representations used to measure similarity between samples.
3. **Nearest-Neighbor Retrieval:** For each input sample, the model retrieves the most similar samples from the training dataset.
4. **Value Information:** Information from the retrieved samples is used as additional context.
5. **Prediction Module:** The model combines the input representation and retrieved context to generate the final prediction.

The implementation uses a single-head attention-like retrieval mechanism rather than a conventional multi-layer, multi-head Transformer architecture.

### Model Configuration

| Parameter | Value |
|---|---:|
| Representation dimension | 265 |
| Encoder residual blocks | 0 |
| Predictor blocks | 1 |
| Numerical feature embeddings | Disabled |
| Attention dropout | 0.38 |
| T-module dropout | 0.38 |
| Retrieval context size (m) | 96 |

## Datasets

The implementation was evaluated on eight datasets covering different tabular learning tasks.

| Dataset | Task | Evaluation Metric |
|---|---|---|
| Churn | Binary Classification | Accuracy |
| California Housing | Regression | RMSE |
| House | Regression | RMSE |
| Adult | Binary Classification | Accuracy |
| Diamonds | Regression | RMSE |
| Otto | Multiclass Classification | Accuracy |
| Higgs-small | Binary Classification | Accuracy |
| Black-Friday | Regression | RMSE |

## Data Preprocessing

The following preprocessing steps were applied:

- **Missing Values:** Numerical missing and infinite values were replaced using the corresponding training-set means.
- **Numerical Features:** Quantile transformation to a normal distribution was applied, with the transformation fitted only on the training split.
- **Categorical Features:** One-hot encoding was used. The encoder was fitted on the training data, and unseen categories were handled using all-zero encodings.
- **Regression Targets:** Targets were standardized using statistics calculated from the training split.
- **Classification Labels:** Class labels were converted into integer representations.

The preprocessing transformations were fitted using training data to avoid information leakage from the test set.

## Implementation Details

The project was implemented using **PyTorch**.

### Retrieval Mechanism

- Similarity between query and candidate samples is calculated using negative squared L2 distance.
- The model retrieves the top-m nearest training samples.
- The default retrieval context size is 96.
- Exact brute-force retrieval is used instead of approximate nearest-neighbor search.
- Candidate samples are processed in chunks of 65,536 to manage memory usage.
- Candidate key representations are refreshed using the current model during training.
- A query sample is excluded from retrieving itself as part of its context.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 0.0003 |
| Weight Decay | 0.00001 |
| Maximum Epochs | 200 |
| Early Stopping Patience | 16 |
| Random Seeds | 3 |
| Batch Sizes | 256, 512, 1024 (dataset-dependent) |
| Environment | Google Colab |
| GPU | NVIDIA Tesla T4 |

### Loss Functions

Different loss functions were used depending on the prediction task:

- **Regression:** Mean Squared Error (MSE)
- **Binary Classification:** Binary Cross-Entropy with Logits
- **Multiclass Classification:** Cross-Entropy Loss

## Experiments and Results

Three major experiments were conducted.

### Experiment 1: Baseline Performance

The baseline implementation was evaluated on eight datasets. Results were averaged across three random seeds and compared with the results reported in the original TabR paper.

| Dataset | Metric | Our Result | Paper Result |
|---|---|---:|---:|
| Churn | Accuracy | 0.8440 | 0.860 |
| California Housing | RMSE | 0.4048 | 0.403 |
| House | RMSE (×10⁴) | 3.234 | 3.067 |
| Adult | Accuracy | 0.8587 | 0.865 |
| Diamonds | RMSE | 0.1336 | 0.133 |
| Otto | Accuracy | 0.8130 | 0.818 |
| Higgs-small | Accuracy | 0.7208 | 0.722 |
| Black-Friday | RMSE | 0.6920 | 0.690 |

The implementation achieved results close to the original paper across most of the evaluated datasets.

### Experiment 2: Effect of Retrieval Context Size

The effect of the number of retrieved neighbors was studied using three context sizes:

- m = 48
- m = 96
- m = 144

The observations were:

- Higgs-small showed improvement with larger retrieval context sizes.
- Black-Friday RMSE decreased as the context size increased.
- Adult improved when the context size increased from 48 to 96, with limited additional improvement at 144.
- Churn performance remained relatively stable.
- Other datasets showed smaller or inconsistent changes.

This experiment demonstrates that the retrieval context size can influence model performance differently across datasets.

### Experiment 3: Robustness to Gaussian Noise

The robustness of TabR was evaluated by introducing Gaussian noise into test inputs.

The experiment compared TabR with an MLP trained on clean data using three datasets:

- California Housing
- House
- Churn

Noise levels of 0%, 5%, 10%, 20%, and 30% were considered.

**Observations:**

- On California Housing, TabR achieved better performance than the MLP on clean test data.
- As noise increased, TabR's RMSE deteriorated more rapidly than the MLP's on California Housing and House.
- On Churn, TabR and MLP showed comparable accuracy across the noise levels.
- The results indicate that TabR's robustness to input noise depends on the dataset and is not consistently better than that of the MLP.

## Project Structure

```text
TABR_DataScience_Team11/
│
├── Code/                         # Model implementation and experiment code
│
├── data/                         # Dataset directory
│
├── results/                      # Experimental results
│
├── requirements.txt              # Python dependencies
│
├── TabR_FInalReport_Team11.pdf   # Detailed project report
│
└── README.md                     # Project documentation
```

## Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Data Processing:** NumPy, Pandas
- **Machine Learning:** Scikit-learn
- **Scientific Computing:** SciPy
- **Visualization:** Matplotlib
- **Development Environment:** Google Colab

## Installation and Execution

The experiments were conducted using **Google Colab**.

### 1. Open Google Colab

Visit [Google Colab](https://colab.research.google.com/) and create a new notebook.

### 2. Clone the Repository

Run the following command in a notebook cell:

```python
!git clone https://github.com/Prayuktha-Lucky-Reddy/TABR_DataScience_Team11.git
```

### 3. Navigate to the Project Directory

```python
%cd TABR_DataScience_Team11
```

### 4. Install Dependencies

```python
!pip install -r requirements.txt
```

The project dependencies include:

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy

### 5. Run the Implementation

Open the `Code/` directory and execute the appropriate implementation or experiment file in Google Colab.

For GPU execution, select **Runtime → Change runtime type → GPU**. GPU availability depends on the Colab account and available resources.

Refer to the project report for the experimental configurations and evaluation details.

## Limitations

- Only eight of the eleven datasets considered in the benchmark were completed due to computational constraints.
- Weather-big, Covtype, and Microsoft datasets were not evaluated.
- Three random seeds were used instead of the fifteen seeds used in the original paper.
- Extensive dataset-specific hyperparameter tuning was not performed.
- GBDT and extensively tuned MLP baselines were not included in the main benchmark comparison.
- Context freezing and ensembling were not implemented.
- The robustness experiment was limited to three datasets and Gaussian input noise.

## Conclusion

This project implements and evaluates TabR, a retrieval-augmented deep learning approach for tabular data.

The baseline experiments produced results close to those reported in the original paper across eight datasets. Additional experiments explored the influence of retrieval context size and the effect of Gaussian noise on model performance.

Overall, the project provided an opportunity to study how nearest-neighbor retrieval can be integrated with neural networks and evaluated across different tabular learning tasks.

## Reference

Gorishniy et al., *TabR: Tabular Deep Learning Meets Nearest Neighbors*, International Conference on Learning Representations (ICLR), 2024.
