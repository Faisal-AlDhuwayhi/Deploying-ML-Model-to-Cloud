# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a `RandomForestClassifier` trained to predict the salary category (`<=50K` or `>50K`) based on various demographic and employment features from the census dataset. The model was trained using the scikit-learn library

## Intended Use
The primary use case for this model is to predict individuals' income levels based on census features. It can help determine if a person qualifies for facilities or services like loan approval by predicting whether they earn over 50k.

## Training Data
The model was trained on a dataset obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/20/census+income). The training data consists of a **cleaned version** of the census dataset, which includes features such as:
- age, 
- workclass, 
- education, 
- marital status, 
- occupation, 
- relationship, 
- race,
- sex, 
- hours per week, 
- and native country. 

The dataset was split into training and testing sets with an *80-20* split.

Cleaning data include remove trailing spaces in columns names and values, also change the value `?` to `Unknown` for categorical columns for better readability.

As an inference pipeline, it includes `One-hot-encoding` of categorical features and `Label-binarizer` for the target feature.

## Evaluation Data
The evaluation data is the 20% split of the original census dataset that was not used for training. This data was used to evaluate the performance of the model

## Metrics
The model's performance was evaluated using **precision, recall, and F1-score (beta=1).** Below are the metrics for the model:
- Precision: 0.73
- Recall: 0.62
- F1-score: 0.67

**Note**: The focus of the project is to demonstrate the concept of deploying an end-to-end ML model to cloud platform and using DVC, not on the model performance.

## Ethical Considerations
- **Bias**: The model may inherit biases present in the training data, such as biases related to race, gender, or socioeconomic status.
- **Fairness**: The model's predictions should be interpreted with caution, especially when used in contexts that could impact individuals' lives.
- **Privacy**: The data used for training the model should be handled with care to ensure the privacy of individuals.

## Caveats and Recommendations
- The model is trained on a specific dataset and may not generalize well to other datasets or real-world scenarios.
- The model should not be used for making critical decisions without further validation and consideration of ethical implications.
- Users should be aware of the potential biases and limitations of the model and use it responsibly.
