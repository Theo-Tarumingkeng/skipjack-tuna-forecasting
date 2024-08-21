# Skipjack-Tuna-Forecasting

An end to end forecasting skipjack tuna catches in the Sulawesi Sea project using Linear Regression & Recurrent Neural Network models, with performance comparison of the two models.

## Frameworks & Libraries
Frameworks & Libraries used are as follows:
* Scikit-learn : for the Linear Regression model
* Tensorflow Keras : for the Recurrent Neural Network model
* Flask : for the model deployment into simple web application

## Dataset
The dataset used in this project is a merged data of daily catches of Skipjack Tuna with daily measurements of environmental variables in Sulawesi Sea from 2018 to 2022 (5 years). The amount of data is 1770 rows.

The environmental variables contained in the dataset are:   
1. **Sea Surface Temperatures** (m/s) [NUMERIC]   
2. **Chlorophyll Levels** (mg/m<sup>3</sup>) [NUMERIC] 
3. **Northward & Eastward Wind Velocities** (m/s) [NUMERIC] 
4. **Northward & Eastward Seawater Velocities** (m/s) [NUMERIC]
5. **Wind & Seawater Magnitude** without considering direction (m/s) [NUMERIC]
6. **Wind & Seawater Direction** [CATEGORICAL]
7. **Fishing season** [CATEGORICAL]   

### NOTE : **In this project, I only used the numeric variables, hence the categorical variabel ignored**

You can see the full dataset [HERE](https://drive.google.com/file/d/1wMW3ljotmVUdqro7FFo9lgt8AQp015mC/view?usp=sharing) (the dataset is in CSV format)

## Train & Test
The proportion used for train & test data division is 40:60, means the amount of data for training is 2 from 5 years of data. The division of data for training & testing is carried out based on the row index.
|   Train data   |   Test data    |
|----------------|----------------|
|    708 rows    |   1062 rows    |

## Prediction results
This is the visualization of prediction results from the two models compared to the test data
<p align="center">
  <img src="Prediksi Cakalang/figures/hasil prediksi LINREG.png" width="500" alt="accessibility text">
  <img src="Prediksi Cakalang/figures/hasil prediksi RNN.png" width="500" alt="accessibility text">
  <br>
  <em>Prediction results from Linear Regression and RNN models</em>
</p>

## Model Evaluation
And this is the performance comparison between the models using these evaluation metrics: 
* *Coefficient of Determination* (R<sup>2</sup>)
* *Mean Absolute Error* (MAE) *in kilogram
* *Mean Squared Error* (MSE) *in kilogram
* *Root Mean Squared Error* (RMSE) *in kilogram

|     *Metrics*     | *Linear Regression* | *Recurrent Neural Network* |
|-------------------|---------------------|----------------------------|
|   R<sup>2</sup>   |  -0.0048899335214   |      -0.0055417886416      |
|       *MAE*       |      6805.790       |          6216.985          |
|       *MSE*       |     76330724.1      |         76380240.0         |
|       *RMSE*      |      8736.746       |          8739.579          |


