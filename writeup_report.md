# **Behavioral Cloning** 


In this project we built a convolutional neural network with Keras and tensorflow with the goal to create a model that can predict the correct steering angle of a car given a single input image. The idea is to use an end-to-end pure deep learning approach in order to let the model learn the relevant features of the road while driving and recording the steering angle. A regression model can then be used to predict the steering angle feeding an image from a front camera.

The project consisted in the following steps:

* Use of a simulator to collect data (composed of images and steering angle) of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set extracted from the collected data
* Test that the model successfully drives around the tracks without leaving the road

[//]: # (Image References)

[angle_distr_1]: ./images/angles_distribution_1.png "Recorded steering angle distribution"
[angle_distr_2]: ./images/angles_distribution_2.png "Steering angle distribution (with left/right camera)"
[angle_distr_3]: ./images/angles_distribution_3.png "Steering angle distribution (with mirrored images)"
[angle_distr_4]: ./images/angles_distribution_4.png "Normalized steering angle distribution"
[train_distr]: ./images/train_distribution.png "Training set angle distribution"
[valid_distr]: ./images/valid_distribution.png "Validation set angle distribution"
[train_gen_distr]: ./images/train_gen_distribution.png "Training distribution from the generator"
[valid_gen_distr]: ./images/valid_distribution.png "Validation distribution from the generator"
[nvidia_architecture]: ./images/nvidia_architecture.png "End-to-End Nvidia Architecture"
[model_architecture]: ./images/model_architecture.png "Model architecture"
[model_log]: ./images/model_log.png "Training log"
[model_log_overfitting]: ./images/model_log_overfitting.png "Training Overfitting"
[model_log_nobn]: ./images/model_log_nobn.png "Training without batch normalization"
[model_log_high_learning_rate]: ./images/model_log_high_learning_rate.png "Training with a high learning rate"
[tricky_turn]: ./images/tricky_turn.jpg "The model thinks there is a road straight right ahead"
[camera_center_left_right]: ./images/camera_center_left_right.png "Center, left and right cameras view"
[camera_corrected_angles]: ./images/camera_corrected_angles.png "Center, left and right cameras with corrected angles"
[camera_mirrored]: ./images/camera_mirrored.png "Center, left and right cameras with mirrored view"
[camera_processing]: ./images/camera_processing.png "Image preprocessing"
[camera_rnd_transform]: ./images/camera_rnd_transform.png "Image random transformations"


## Rubric Points
### In the following I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

The project includes the following files:
* **[app_args.py](./app_args.py)** Container for the command line arguments supported by the application
* **[data_loader.py](./data_loader.py)** Data loader/generator that is used to create the dataset that is fed to the model for training
* **[image_processor.py](image_processor.py)** Contains a shared pre-processing pipeline to be used on the images that are fed to the model. The same is used in the [drive.py](./drive.py)
* **[model.py](model.py)** Contains the code that creates and trains the model
* **[plots.py](plots.py)** Contains utilities to create plots from the data
* **[drive.py](drive.py)** For driving the car in autonomous mode and recording the front camera output
* **[video.py](video.py)** For creating a video from the [drive.py](./drive.py) output when recording, allows to create a version with the images pre-processed as being fed to the model
* **[process_fmap.py](process_fmap.py)** Utility to create a video of the feature maps from the various layers of the model
* **[model.h5](model.h5)** Contains the final trained model
* **[video_track_1.mp4](./video_track_1.mp4)** A video recording of the car driving on the first track in autonomous mode that includes all the views (top, front, processing, feature map)
* **[video_track_1_front_camera.mp4](./video_track_1_front_camera.mp4)** A video recording from the front camera of the car driving on the first track in autonomous mode
* **[video_track_2.mp4](./video_track_2.mp4)** A video recording of the car driving on the second track in autonomous mode
that includes all the views (top, front, processing, feature map)
* **[video_track_2_front_camera.mp4](./video_track_2_front_camera.mp4)** A video recording from the front camera of the car driving on the first track in autonomous mode 

#### 2. Submission includes functional code
Using the Udacity provided simulator and the  [drive.py](./drive.py) file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The  [model.py](./model.py) file contains the code for training and saving the convolution neural network. I split the code in different modules to make it more readable. In particular the  [data_loader.py](./data_loader.py) module provides a class that generates the data from the driving log, augment it and supplies a generator for training, moreover it saves into a pickle file the result of the processing so that it can be reused for training with different parameters. Additionally I split the image processing needed for the model into a separate [module](./image_processor.py) so that it can be reused from the [drive.py](./drive.py).

The code is commented when needed to explain how the pipeline works. To make my life easier I added several command line parameters that can be used to tune both the dataset and the model when training, in particular the [model.py](./model.py) accepts the following parameters (can be used as ```python model.py --parameter_name=parameter_value```:

* **epochs** The number of epochs; defaults to 50
* **batch_size** The mini batch size; defaults to 64
* **learning_rate** The learning rate for the adam optimizer; defaults to 0.0001
* **loss** The name of the loss function to use; defaults to "mse"
* **dropout** The dropout probability; defaults to 0.1
* **activation** The name of the activation function to use; defaults to "relu"
* **batch_norm** The batch normalization momentum, if 0 batch norm is not applied; defaults to 0.9')
* **angle_correction** The correction angle to add to right and left camera images; defaults to 0.15
* **mirror_min_angle** The min steering angle needed in order to discriminate if an image should be mirrored or not; defaults to 0.0 (e.g. all images are mirrored)
* **normalize_factor** The factor applied for normalization when dropping values over the mean; defaults to 1.5 (e.g. 1.5 times the mean)
* **normalize_bins** The number of bins to divide the data when normalizing; defaults to 100
* **regenerate** True if the dataset should be regenerated from the log file, rather than loaded from the pickle file; defaults to False
* **preprocess** True if the generator should preprocess all the images in memory, speeds up the training but requires more memory; defaults to True
* **clahe** True if histogram equalization should be applied during preprocessing; defaults to False
* **blur** True if blurring should be applied during preprocessing; defaults to True
* **random_transform** True if random image transformations should be applied during training; defaults to True

Additionally I created a simple script [process_fmap.py](./process_fmap.py) to output a video of the feature maps from the images generated by the [drive.py](./drive.py). This allows to see if the model is learning the road and environment features.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Given the problem at end I decided to start right away from a well known model, taking inspiration from the Nvidia [End-to-End learning for self-driving cars](https://arxiv.org/pdf/1604.07316v1.pdf) paper.

I initially implemented the Nvidia architecture as described in the paper:

![alt text][nvidia_architecture]

I then added a few modifications that were not clarified in the paper, in particular I used a **RELU** activation in all the layers (but the output) and added **batch normalization** for the fully connected layers. As the Nvidia paper I added a **normalization** layer for the input to speed up training. I also introduced a **dropout** layer after the first fully connected layer to reduce overfitting (even though batch normalization provided a regularization effect):

![alt text][model_architecture]

The batch normalization was introduced to speed training but had only a regularization effect. I also experimented with a variation of the model using max pooling instead of subsampling using the stride in the convolutions, this led to a bit more complex model that was giving similar results but the training speed tripled so I decided to stick to the original architecture.

#### 2. Attempts to reduce overfitting in the model

I initially recorded a limited set of data with around 10k measurements, at this stage the model was slightly overfitting as depicted from the image below:

![alt text][model_log_overfitting]

Note that at this point I already introduced a 0.2 probability dropout (but without batch normalization) in the first fully connected layer, increasing the dropout led to worse performance while driving on the second track so I decided to keep the dropout to a minimum and instead introduce more data recording an additional back and forth laps on both tracks using the mouse for steering, this led to more complex data that reduced overfitting:

![alt text][model_log]

Note that the validation loss is actually lower than the training loss but I decided to stop training as the loss was not a good indicator for how well the car would drive. There are various reasons for this behavior: as explained in the keras documentation the loss for training is an average among all the batches, while the validation loss is computed at the end of the epoch. Since I have a relative big among of data (around 30k samples per epoch) this can lead to a validation loss which is lower than training and that slowly catches up after some epochs (since the gap between the initial batch loss and the final batch is smaller). I also introduced batch normalization that provides a regularization effect. Training for more epochs would have led to the two losses to converge but the final result was satisfactory so I decided to stop there. The following shows the training of the model without batch normalization:

![alt text][model_log_nobn]

To make sure that the model was generalizing enough and not having at my disposal more tracks to try, I tested the model running the simulator with different graphics settings (e.g. on the lower level shadows are not present) and also in different directions, the final model was able to keep the car in the track in all the different scenarios.

#### 3. Model parameter tuning

I had several different parameters to tune for the model, in particular:

* Batch size
* Learning rate
* Batch normalization momentum
* Dropout probability

I did an hyperparameter search running several experiments changing one parameter at the time and taking tha one that would give me the best results, it was quite challenging since the loss is not a good indicator of how well the car would drive, but how the loss would decrease was a better indicator. I quickly settled for a batch size of 64 and a learning rate of 0.0001. While the model uses an adam optimizer the default 0.001 was too high:

![alt text][model_log_high_learning_rate]

The dropout and the batch normalization were more affected by the type of data collected, I experimented a lot with the amount of data and with its normalization in order to reach the goal. This was part of the parameter tuning and I'd say that the data collected was far more important than the actually model/parameters.

#### 4. Appropriate training data

I collected the training data driving on both track in the center of the lane and in both direction, for a total of around 4 laps and 12k measurements. I used the mouse and keyboard so that the data collected was more accurate. Given that the simulator provided also left and right camera view I augmented my data using that information and correcting the steering angle appropriately of 0.15. I also mirrored the images so to further increase the data. Moreover I "normalized' the data cutting the spikes in number of samples down to a factor of the mean of the number of samples. I did this dividing in around 100 bins the measurements and computing the mean value, then cutting the exceeding samples. The code is in the _normalize function in the [data_loader.py](./data_loader.py). To further help the model generalize I randomly applied a brightness factor to each image generated in each batch plus a slightly random translation on the x axis (with an appropriate angle correction). Given that I use a generator this provides a lot of sparse data for training. The data augmented is then split for training and validation with a factor of 0.8. For the validation set I decided to disable the random brightness and translation to avoid going off the test data (e.g. the simulator) too much.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As I explained in an earlier section I decided to spend more time on the data rather than the model and chose to use the Nvidia architecture trusting that it would probably work given the similarity of the problem. My intuition was right as I could drive the first track basically on the first try. As usual I split the data collected/augmented in a training and validation set while for testing I had to rely on trying the model each time on the track(s). The validation data set helped in tuning the parameter to see if the model was overfitting. Even though in this case it was difficult to say much about the end result. The images fed to the model are pre-processed, in particular a cropping factor is applied to remove redundant parts of the image (e.g. the hood of the car and the sky) as suggested by the Nvidia paper. I also resized the image to the same size of the Nvidia paper, this not only allowed me to stay as close as possible to their experimentation but also increased the training speed. 

As from the paper the images are converted in YUV color space, I tested with different color spaces as well but I didn't notice any particular improvement (with RGB the car was driving worse). Finally to introduce a bit of noise in the crispy images from the simulator I added a gaussian blur, in the hope that the model would learn better the road features, this turned to be true as it was less inclined to be tricked by too many details.

Initially I collected a limited set of data and the model was quickly overfitting, adding new data and augmenting the dataset helped a lot. The car was driving well on the first track but had some issues on the second track in a particular section with 2 hard turns and shadows. At that point I introduced **histogram equalization** thinking that balancing the images would help the model fight the shadow environment. An alternative would have been to collect more data using different settings of the simulator (e.g. disabling/enabling the shadows) but I didn't want to have too much data and wanted to keep my training time relatively short. The histogram equalization helped a bit but was not enough, so I decided to introduce **random brightness** in the generator itself together with **random translation** on the x axis with a correction on the angle so that the model could generalize better. This allowed me to complete the second track with a low speed. I analized the images in that particular section and quickly realized that the model was tricked into thinking that the road was going straight instead of turning due to perspective:

![alt text][tricky_turn]

This was the most challenging part of the track, it helped me get down to a good cropping factor (e.g. cut 60 pixels from the top) plus tuning the data and processing. I added some other data for that particular section that helped relatively, in the end adding random translation and brightness was the key to let the model generalize enough to complete this part of the track correctly. I pushed a bit the limit increasing the speed of the driving to 25 instead of the default 9 for the second track, for this I had to tune the collected data to the right balance. In particular I implemented a pipeline that allowed me to quickly change the distribution of the data. After several experiments I was able to complete both tracks at "sustained" speeds, the first one had no issues with the maximum speed allowed by the simulator (30), while for the second track a speed of 25 was appropriate to avoid making the car "jump" on the hills, I added a small modification to reduce the throttle to zero on hard turns so that it would slow down a bit and this helped in completing the track smoothly. I noticed that the performance of the PC mattered a lot, and my guess is that the frequency of the image sampling is not high enough and the car fails if the simulator does not have enough resources.

#### 2. Final Model Architecture

The final model architecture is very similar to the one presented in the Nvidia paper, as stated in the previous section I added RELU activation, batch normalization and dropout (note that in the image below I didn't include the normalization layer):

![alt text][model_architecture]

#### 3. Creation of the Training Set & Training Process

I collected the data driving 1 laps in each track in both direction for a total of 4 laps plus a recovery data collection in difficult parts of the tracks. The simulator captures 3 images from the center, left and right side of the car:

![alt text][camera_center_left_right]

The simulator records different measurements: steering angle, throttle and break force. For this project I decided to follow the nvidia approach and use purely the steering angle. The following image shows the steering angle distribution of the original data (only center camera):

![alt text][angle_distr_1]

I used all the 3 images for each measurement, adjusting the steering angle accordingly reaching a correction value of 0.15 after several experiments:

![alt text][camera_corrected_angles]

The use of the 3 cameras leads to triple the amount of data:

![alt text][angle_distr_2]

This was my starting point, I then mirrored the images to further gain some "free" data, accordingly I inverted the steering angles:

![alt text][camera_mirrored]

This lead me to further double the data, reaching more than 65k samples:

![alt text][angle_distr_3]

Given the amount of data collected in the near-zero angle I decided to cut off the peaks using a factor of the mean of the amount of samples in a set of bins. This allowed the model to avoid being bias toward driving always straight:

![alt text][angle_distr_4]

Finally I shuffled the data and used a generator to supply the data for training and validation using 20% of the data for validation:

![alt text][train_distr]
![alt text][valid_distr]

In the generator I also preprocess the images using the [image_processor.py](./image_processor.py) that changes the color space to YUV, crops the image (60 from the top and 25 from the bottom), resizes it to 200x66 and applies a light gaussian blur:

![alt text][camera_processing]

After processing the generator applies random transformations, in particular adjusting the brightness randomly and translating the image on the x axis correcting the angle:

![alt text][camera_rnd_transform]

The random transformations help balancing the training set further, the final distribution of the training and validation dataset coming from the generator is as follows:

![alt text][train_gen_distr]
![alt text][valid_gen_distr]

I then trained the model using the following configuration:

* **Epochs** 40
* **Batch size** 64
* **Learning rate** 0.0001
* **Batch normalization momentum** 0.9

![alt text][model_log]