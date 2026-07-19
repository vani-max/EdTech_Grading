# Automated EdTech Grading Assistant

## Project Overview
This project focuses on automating the grading of handwritten worksheets by designing an image classification pipeline[cite: 1]. By utilizing the MNIST Handwritten Digits Database, we implement and evaluate machine learning architectures to recognize and categorize digits from 0 to 9[cite: 1].

## Key Learning Outcomes
*   **Image Processing**: Learned to ingest, reshape, and visualize high-dimensional pixel matrices[cite: 1].
*   **Data Normalization**: Implemented Min-Max scaling to prepare pixel intensity features for distance-based models[cite: 1].
*   **Model Implementation**: Built and compared Linear and RBF-kernel Support Vector Machines (SVM)[cite: 1].
*   **Performance Metrics**: Evaluated models using multi-class confusion matrices and macro-averaged F1-scores[cite: 1].

## Progress Roadmap (Weeks 1–5)

### Week 1: Data Ingestion & Visualization
*   Successfully loaded the MNIST dataset using `fetch_openml('mnist_784')`.
*   Reshaped 784-element feature vectors into $28 \times 28$ pixel matrices.
*   Visualized sample digits to verify data integrity against target labels.

### Week 2: Preprocessing & Scaling
*   Flattened image matrices into 1D feature vectors suitable for SVM models.
*   Applied Min-Max scaling to normalize pixel intensities from the range $(0 \dots 255)$ to $(0 \dots 1)$, ensuring efficient model convergence.

### Week 3: Baseline Classification
*   Partitioned the data using a 20,000-row training subset to maintain fast training times.
*   Established a performance baseline using a `LinearSVC` model, achieving an initial accuracy of 0.90.

### Week 4: Training SVM RBF Kernel Models
*   Implemented `SVC` with an RBF kernel to map features into high-dimensional space.
*   Demonstrated that non-linear decision boundaries improve classification accuracy compared to the linear baseline.

### Week 5: Analyzing Errors with Confusion Matrices
*   Generated multi-class confusion matrices to visualize classification performance.
*   Identified specific digit pairs (e.g., '4' vs '9') that exhibit the highest confusion, providing insights for future hyperparameter tuning.

## Technical Stack
*   **Language**: Python 3.10+
*   **Libraries**: NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn
*   **Environment**: Jupyter Lab / VS Code

## Next Steps
*   **Week 6**: Perform hyperparameter optimization using `GridSearchCV`.
*   **Week 7**: Compare SVM results against tree-based models like `RandomForestClassifier`.
*   **Week 8**: Package the pipeline into a modular script for interactive inference.
