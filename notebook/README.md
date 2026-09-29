# Notebook

## Final experimental notebook

`Project_final_(1).ipynb` contains the complete executable workflow:

1. Environment setup
2. Dataset loading
3. Train-test split
4. Model definitions
5. Feature-noise generation
6. Label-noise generation
7. Baseline evaluation
8. Feature-noise experiment
9. Label-noise experiment
10. Summary statistics
11. F1 degradation analysis
12. Feature-versus-label comparison
13. Transition analysis
14. Confusion-matrix analysis
15. Final figures
16. Result-file generation

### Conditions

Feature noise: 0%, 2%, 5%, 10%, 15%, 20%, 30%.

Label noise: 0%, 2%, 4%, 6%, 8%, 10%.

Transition diagnostic: 6%, 8%, 8.5%, 9%, 9.5%, 10%.

Each condition is repeated 10 times using different random seeds.

### Models

- Logistic Regression
- KNN with `n_neighbors=5`

Both use standardized features through a scikit-learn pipeline.

### Evaluation

The test set remains clean and fixed. The notebook records Accuracy, Precision, Recall, F1-score and ROC-AUC.

### Execution

```bash
pip install -r ../requirements.txt
```

Open the notebook in Jupyter or Google Colab and run all cells from a fresh runtime.

The notebook is intentionally compact. Detailed theory, critical evaluation, limitations, conclusion and the formal AI declaration belong in the written report.

