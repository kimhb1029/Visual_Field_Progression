# Labeling Matters: A Multicenter Machine Learning Study on Visual Field Progression in Glaucoma
This is the code used in

'Labeling Matters: A Multicenter Machine Learning Study on Visual Field Progression in Glaucoma'

Hyobeen Kim, EunAh Kim, Sangwoo Moon, Sang Wook Jin, Jung Lim Kim, Seung Uk Lee, Jeong Rye Park and Jiwoong Lee

[Paper](#) (Coming soon)

## Data

The original dataset cannot be publicly released.  
We only provide the preprocessed **train** and **test** data.

To illustrate the preprocessing pipeline, we include a small sample dataset that does not contain any personally identifiable information.  
See `Example_for_sample.ipynb` for details.

## Abstract

Purpose: To compare machine learning (ML) performance for detecting visual field (VF) progression across different labeling strategies using a large multicenter dataset.

Methods: In this multicenter retrospective study, VF data were collected from five tertiary referral hospitals. Two labeling approaches were evaluated: an inclusive Consensus label, defined as progression detected by at least one of five conventional algorithms (mean deviation slope, Visual Field Index slope, Advanced Glaucoma Intervention Study, Collaborative Initial Glaucoma Treatment Study, and pointwise linear regression), and a conservative Wiggs’ label, based on a region-based event–threshold rule. Four ML classifiers, support vector machine, random forest, logistic regression, and extreme gradient boosting, were trained using each labeling strategy. Model performance was assessed using area under the receiver operating characteristic curve (AUC), sensitivity, specificity, and precision–recall analysis summarized by average precision (AP).

Results: Using the Consensus label, all models demonstrated excellent discrimination (AUC, 0.92–0.95), with high sensitivity (0.82–0.85) and near-perfect specificity (0.99–1.00). Precision–recall analysis showed consistently high reliability of progression predictions, with AP values ranging from 0.93 to 0.94. In contrast, models trained with the Wiggs’ label exhibited lower AUCs (0.88–0.89) and reduced sensitivity (0.63–0.72), while maintaining moderate-to-high specificity (0.87–0.92) and lower AP values (0.84–0.85), reflecting a stricter, region-based progression definition.

Conclusions: In this multicenter study, labeling strategy was a major determinant of ML performance in VF progression detection. The Consensus label enabled sensitive and reliable identification of progression with minimal false positives across heterogeneous clinical settings, whereas the Wiggs’ label provided conservative, spatially consistent confirmation. These findings underscore that careful definition of ground truth is critical for developing robust and generalizable glaucoma AI systems.
