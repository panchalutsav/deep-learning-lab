## Team10
- Gautham Mohan (st184914)
- Utsav Panchal (st184584)

## Diabetic Retinopathy Detection

To run the code you can add different flags inside the batch.sh file. To simply train and evaluate a cnn_se model.

```
python3 main.py --train --eval --model_name "cnn_se" 
```

You can change the model name inside the **'batch.sh** file under **'--model_name'** flag.  
Currently three models are available. 
1) **cnn_1**: Custom CNN model
2) **cnn_se**: Custom CNN model with squeeze and excitation blocks
3) **transfer_model**: Transfer learning model

Note: If you are using transfer learning model, make sure to update the base model name which you want to train in the **'configs/config.gin'** file under the parameter name **transfer_model.base_model_name = 'DenseNet121'** .
Currently three transfer learning models are available: 
1) **DenseNet121**: Dense Net model
2) **InceptionV3**: Inception Net V3 model 
3) **InceptionResnet**: Inception Resnet model 


## Image Preprocessing
The model was trained using different types of image processing methods.  
List of preprocessing methods we tried. 
1) Original (RGB) Images with cropped boundries
2) Bens preprocessing method: Add gaussian blur and subtract the local average
3) Enhance Local Contrast (CLAHE filtering)
5) Bens preprocessing + CLAHE

Out of all the preprocessing methods Original(RGB) and Bens preprocessing methods shown us promising accuracy.  

Starting from left: 1-2-3-4  

![](canvas.jpg)  

The type of image processing methods can be configured in **'configs/config.gin'**. 
```
preprocess_image.with_clahe = False
preprocess_image.with_bens = False
```


## Training and Evaluation
Here you can train different models as described above. 
```
python3 main.py --train --eval --model_name "cnn_se" 
```

## Evaluation
Make sure to provide evaluation checkpoint in **'configs/config.gin'**
```
python3 main.py --eval
```
## Ensemble learning
You can create an ensemble of the available pretrained models by specifying the model type and their checkpoints in the **configs/config.gin**  file.
The voting can be done in both hard and soft methods. This can also be specified in the **configs/config.gin**  file. 

```
python3 main.py --eval --model_name "ensemble_model" 
```

## Wandb sweep
You can a run a sweep configuration for a particular model.  
Inside the **'sweep_configs/'** directory, different sweep configurations are available. Copy and paste the sweep into **'wandb_sweep.py'**. 
```
python3 wandb_sweep.py
```

## Deep Visualization
You can a run different deep visualization methods by activating the flag. This will create a single image which consists of 3 different deep visualization methods. An example image is attached at the end.  

```
python3 main.py --train --eval --model_name "cnn_se" --deep_visu
```

# Results
We ran each model 10 times to see the variation in accuracy. The metric shown here is Sparse Categorical Accuracy(%).  
The logs are stored in **'good_performances/'** directory. 

|  | CNN_1 | CNN_SE | DenseNet121 | InceptionV3 | InceptionResnet |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Run 1 | 82.52 | 82.52 | 82.52 | 82.52 | 78.64 |
| Run 2 | 78.64 | 79.61| 82.52 | 80.58 | 76.70 |
| Run 3 | 77.67 | 84.47 | 83.50 | 77.67 | 77.67 |
| Run 4 | 84.47 | 83.50 | 83.50 | 79.61 | 79.61 |
| Run 5 | 77.67 | 85.44 | 81.55 | 79.61 | 79.61 |
| Run 6 | 82.52 | 83.50 | 82.52 | 78.64 | 76.70 |
| Run 7 | 81.55 | 76.70 | 81.55 | 79.61 | 80.58 |
| Run 8 | 78.64 | 84.47 | 82.52 | 79.61 | 74.76 |
| Run 9 | 79.61 | 69.90 | 82.52 | 79.61 | 81.55 |
| Run 10 | 82.52 | 80.58 | 82.52 | 78.64 | 77.67 |



The best performances metrics are shown below.  
Please note that it is difficult to reprocude the same accuracy because of limited dataset as seen above in 10 runs.   


|  | Accuracy (%) | Balanced Accuracy (%) | Sensitivity (%) | Specificity (%) | Recall (%) | Precision(%) | F1 Score(%) | AUC (%) | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN_1 | 84.47 | 84.35 |  72.55 | 96.15 | 72.55 | 94.87 | 82.22 | 86.00 |
| CNN_SE | 85.44 | 84.44 | 80.00 | 88.89 | 80.00 | 82.05 | 81.01 | 85.00 |
| DenseNet121 | 83.50 | 82.36 | 76.19 | 88.52 | 76.19 | 82.05 | 79.01 | 83.00 |
| InceptionNet V3 | 82.52 | 82.34 | 81.82 | 82.86 | 81.82 | 69.23 | 75.00 | 80.00 |
| InceptionResnet | 81.55 | 80.68 | 77.78 | 83.58 | 77.78 | 71.79 | 74.67 | 80.00 |


## Deep Visualisation
An implementation of grad-CAM is done for visualisation. The following are to be specified in **'configs/config.gin'**:
1) List of indices of the images to be visualised from the train and test set.
2) The layer of the neural network at which grad-CAM is applied. 
3) The class label for which visualisation is to be done, if this is not specified the gradCAM is performed for the predicted class.

```
DeepVisualize.image_list_train = [5,6,7,8]
DeepVisualize.image_list_test = [15,57,87,32]
DeepVisualize.layer_name = 'activation_3'
GradCam.class_idx = 1
```
![](deep_viz_8.png)

In the image above we can see how the model has learned to identify hard exudates to classify the image as RDR.




