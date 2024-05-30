# Team10
- Gautham Mohan (st184914)
- Utsav Panchal (st184584)

# Human Activity Recognition 
Human Activity Recognition (HAR) is a problem that is an active research field in pervasive
computing. An HAR system has the main goal of analyzing human activities by observing
and interpreting ongoing events successfully

## About the project
In this project we have used Deep learning techniques to analyse the human activities. To do this we have trained our deep learning model on two datasets. 
1) **Human Activities and Postural Transitions Dataset (HAPT)**  
The dataset contains data from the tri-axial accelerometer and gyroscope of a smartphone, both captured at a frequency of 50 Hz. The dataset consists of six basic activities (static and dynamic) and six postural transitions between the static  activities. The static activities include standing, sitting and lying.

2) **Real World (HAR) Dataset**  
The dataset contains signals from various smartphone embedded sensors like the accelerometer, Global-Positioning-System (GPS), gyroscope, light, magnetic field and sound. The activities that are covered by this dataset are climbing downstairs, climbing upstairs, jumping, lying, standing, sitting, running/jogging, and walking. Smartphones were placed at different
body positions: chest, head, shin, thigh, upper arm, waist, and a smartwatch at the forearm that were recorded simultaneously.


## How to run the project 
In the [batch.sh](/human_activity_recognition/batch.sh) file you can set various flags.  
To run the code you can type the following command in terminal. **The train and evaluate flags are set to off by default. Make sure to provide the flags.** Details below. 

```
python3 main.py --train --eval --model_name "model1_LSTM" --hapt
```
## Datasets  
The model can be trained on two datasets described above. You can choose between the datasets HAPT or HAR by providing the flags **--hapt** or **--har**

## Real World (HAR) Dataset: Important Note
If you are training the model using Real World (HAR) dataset, additionally you need to provide the specific body part. To do 
this you must change the bodypart parameter in [main.py](/human_activity_recognition/main.py) line no. 54. 
The data is available for the following body parts: ["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"]

## Dataset Visualisation
The HAPT dataset can be visualised by setting the experiment number to be visualised in the **configs/config.gin** file. 
```
make_tfrecords.visualise_expt = 1
```
The visualisation of experiment number 1 is shown below.

 ![visualise_har](https://media.github.tik.uni-stuttgart.de/user/7666/files/76819a11-8bb9-46c5-b7da-b15c0795887f)


## Models  

You can change the model name inside the [**batch.sh**](/human_activity_recognition/batch.sh) file under **'--model_name'** flag.  

Currently four models are available. 
1) **model1_LSTM**: LSTM model
2) **model_bidirectional_LSTM**: Bidirectional LSTM
3) **model1_GRU**: GRU model
4) **model1D_Conv**: 1-dimensional Convolutional Model



## Training and Evaluation
The training and evaluation flags are off by default.Here you can train and evaluate different models. 
```
python3 main.py --train --eval --model_name "model1_LSTM" --hapt
```

## Evaluation using Checkpoint
If you want to evaluate a pretrained model using a particular checkpoint. You can provide the checkpoint path in **configs/config.gin** and run the following command
```
python3 main.py --eval --model_name "model1_LSTM" --hapt
```
## Ensemble learning
You can create an ensemble of the available pretrained models by specifying the model type and their checkpoints in the **configs/config.gin**  file.
The voting can be done in both hard and soft methods. This can also be specified in the **configs/config.gin**  file. 

```
python3 main.py --eval --model_name "ensemble_model" --hapt
```

## Wandb sweep: Bayesian Hyperparameter Optimization
You can a run a sweep configuration for a particular model inside **'wandb_sweep.py'**. You can select various parameters predefined in the file for any of the above models.
The sweep method by default uses Bayesian Hyperparameter Optimization method.  

```
python3 wandb_sweep.py
```


# Results - HAPT dataset
We ran each model 5 times to see the variation in accuracy. The metric shown here is Sparse Categorical Accuracy(%).  

|  | LSTM | Bi-directional LSTM | GRU | 1-D Convolutional Model | 
| :---: | :---: | :---: | :---: | :---: | 
| Run 1 | 94.51 | 93.05 | 92.19 | 96.03 | 
| Run 2 | 92.95 | 93.71 | 94.21 | 96.39 | 
| Run 3 | 93.88 | 94.57 | 91.20 | 96.33 | 
| Run 4 | 94.48 | 92.92 | 94.74 | 95.96 | 
| Run 5 | 94.11 | 95.10 | 89.94 | 96.59 |

Ensemble learning was done using all the 4 models. The Sparce Categorical Accuracy is shown below for two schemes of voting.
| Ensemble Type | Accuracy |
| :---: | :---: |
| Hard Voting | 95.73 |
| Soft Voting | 96.06 |


# Results - HAR dataset
The files for these results are stored in **results/har/** directory.   
The results particularly in this dataset is hard to reproduce. We have seen 15% of difference in the results.  

|  | LSTM | Bi-directional LSTM | GRU | 1-D Convolutional Model | 
| :---: | :---: | :---: | :---: | :---: | 
| Chest | 66.74 | 92.03 | 90.98 | 75.33 | 
| Forearm | 73.29 | 69.46 | 57.84 | 72.09 | 
| Head | 68.91 | 66.97 | 65.10 | 68.68 | 
| Shin | 82.90 | 70.16 | 71.71 | 72.70 | 
| Thigh | 65.88 | 62.61 | 64.89 | 66.45 | 
| Upperarm | 68.16 | 63.10 | 65.99 | 71.60 |
| Waist | 65.33 | 72.08 | 73.27 | 72.45 |


## Converting the model into TFlite
You can set *--tflite* flag to convert the trained model into tflite format which can be further used in any android application. This will generate a .tflite model and .pb file.  

```
python3 main.py --train --eval --model_name "model1_LSTM" --hapt --tflite
```

## Android Application

We have also implemented an Android Application which detects the real time activites. The dataset used to develop this application is HAR dataset. The source code of the project can be found in [Har App ](/har_app/har2/) directory.  
We've used the model trained on Real World (HAR) dataset specifically for the chest bodypart which gave us the highest result.    
Here's a screenshot of the application.  
<img src = "https://github.tik.uni-stuttgart.de/iss/dl-lab-23w-team10/blob/develop_utsav/human_activity_recognition/android_ss2.jpg" width = "250" height="500" />





 



