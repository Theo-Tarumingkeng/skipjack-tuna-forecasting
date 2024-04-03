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
1. **Sea Surface Temperatures** (m/s)   
2. **Chlorophyll Levels** (mg/m<sup>3</sup>) 
3. **Northward & Eastward Wind Velocities** (m/s) 
4. **Northward & Eastward Seawater Velocities** (m/s)
5. **Wind & Seawater Magnitude** without considering direction (m/s)
6. **Wind & Seawater Direction**
7. **Fishing season**   

### NOTE : **In this project, I only use the numerical variables, hence the categorical variabel ignored**

You can see the full dataset [HERE](https://drive.google.com/file/d/1wMW3ljotmVUdqro7FFo9lgt8AQp015mC/view?usp=sharing) (the dataset is in CSV format)

## Train & Test
The proportion used for train & test data division is 40:60, means the amount of data for training is 2 from 5 years of data. The division of data for training & testing is carried out based on the row index.
|   Train data   |   Test data    |
|----------------|----------------|
|    708 rows    |   1062 rows    |

## Prediction results
This is the visualization of prediction results from the two models compared to the test data
<p align="center">
  <img src="pics/mmu2/b0_loss.png" width="430" alt="accessibility text">
  <img src="pics/mmu2/b0_acc.png" width="430" alt="accessibility text">
  <br>
  <em>Loss and Accuracy curve of EfficientNet-b0 on the MMU2 dataset</em>
</p>



