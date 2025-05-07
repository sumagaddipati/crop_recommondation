# crop_recommondation
## Soil Type Prediction Using Random Forest
This project builds a machine learning model to predict the type of soil based on various environmental and chemical parameters using the Random Forest Classifier. The model is trained on a dataset and includes steps such as data preprocessing, hyperparameter tuning, and evaluation.

## Project Overview
The goal is to predict the soil type based on the following features:

Nitrogen (N)

Phosphorus (P)

Potassium (K)

Temperature

Humidity

pH

Rainfall

The model is built using Random Forest Classifier and incorporates techniques like class balancing, hyperparameter tuning, and cross-validation to ensure robust predictions.


## Requirements
To run this project, ensure you have Python installed, along with the following libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

joblib

You can install all required libraries by running:

pip install pandas numpy matplotlib seaborn scikit-learn joblib
Additionally, you need the dataset soildata.csv. 
Ensure it is placed in the same directory as the script.


Train the Random Forest model on the provided dataset.

Evaluate the model using accuracy, F1-score, and confusion matrix.

Save the trained model and scaler to files (rf_soil_model.pkl and scaler.pkl).

## How to Use
Once the model is trained, you can use it to make predictions by providing new input values (soil parameters).

Prediction Workflow:
Modify Input Values:

Edit the input_values array with new soil data for prediction.

The array should include the following features in this order: N, P, K, temperature, humidity, ph, and rainfall.

Example:

python
Copy
Edit
input_values = [30, 31, 20, 8, 57, 5.8, 101]
Run the Script:

The script will scale the input values, use the trained model to predict the soil type, and display the prediction.

## Example Output:

Predicted Soil Type: class_1
Re-use the Model:

The model and scaler are saved as rf_soil_model.pkl and scaler.pkl.

## Model Evaluation
The model's performance is evaluated using various metrics:

Accuracy: Percentage of correctly predicted soil types.

F1-Score: Harmonic mean of precision and recall, useful for imbalanced datasets.

Confusion Matrix: A matrix that shows the true vs predicted classifications.

Cross-Validation: Evaluates the model on different splits to ensure it generalizes well.

## Example of Model Evaluation:

Initial Model Accuracy: 92.50%

F1 Score: 0.91

Classification Report:

              precision    recall  f1-score   support
              
     class_0       0.93      0.94      0.94       200
    
     class_1       0.91      0.90      0.90       180
     
## Prediction Example
## Example Input:
You can input the following sample values for soil prediction:

input_values = [30, 31, 20, 8, 57, 5.8, 101]  # New soil data
Output:

Predicted Soil Type: class_1
This output indicates that the soil is predicted to belong to class_1 based on the input values.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Notes
The script uses Random Forest Classifier with class_weight='balanced' to handle imbalanced classes.

GridSearchCV is used for hyperparameter tuning to find the best model parameters (e.g., n_estimators, max_depth, etc.).

The trained model and scaler are saved as rf_soil_model.pkl and scaler.pkl for future use.

Feel free to reach out if you need help or have further questions!
