# Yolo-v2 in Keras + Conversion of graph to NCS with api 2.0.8
The idea of this project is to allow training arbitrary networks and convert them to the NCS stick.
An example of pedestrian detection is included that runs at 20 fps in a raspberry pi with one NCS.
This model has obviously much less parameters than the tiny-yolo model.

# Retraining:
This part of the repo is based on https://github.com/rodrigo2019/keras-yolo2.
Follow his readme, it's very clear. I haven't changed much of the project, because I didn't have time to investigate.
The project is great and it works but you need some patience to get a network working.

### Demo/Evaluation
To see how models work I downloaded some videos from youtube similar to my target data and
just run the network on them. I found this much quicker to see if the model is learning correctly or not.
Then I can change the config based on this.

### General comments for retraining
To retrain from scratch I've used Adam and a learning rate of 1e-3. Using 1e-2 as the paper suggests makes the
optimization diverge early on.
I've found for datasets such as openimages using no_object_scale=5 worked well and had less false positives.
I've done some experiments with different combinations of these parameters and they drastically change results.
Concretely this was just running one or two epochs with debug=true, to see how the network reacted to different parameters.
But this depends on the application, so I don't think there is a general advice.
Please feel free to make issues to share your experiments.

# Steps to retrain and Convert to NCS
Please read rodrigo's readme before trying to retrain.
The NCS convertion is based on this repo: https://github.com/bastiaanv/Yolov2-tiny-tf-NCS
I've made some changes but the general idea is the same.
Basically you need to define the network in tensorflow, assign the weights and then compile to the NCS.

## Retraining steps
Create a conda environment using the environment.yml file.

### 1) Change config
`python gen_anchors.py -c config.json`

### 2) Train or fine-tune
`python train.py -c config.json`

### 3) Take a look at how it works with a video.
`python predict.py -c config.json -w ncs_darknetreference/openimages_best4.h5 -i videos/uba.mp4`

## NCS implementation
Each model that is converted to NCS has a separate folder. For example see how ncs_darknetreference/ is structured

### 1) Use the model class in each folder to assign the weights to tensorflow graph
This loads the weights from the ncs_darknetreference/config.json into the class DarknetReferenceNet.
The weights are extracted as numpy arrays and then assigned to the tensorflow graph.
Batch norm is "fused" into the conv-net weights, this seems to be working ok.
The process is quite akward but it's how I made it work.
For some reason I couldn't simplify the process but feel free to make changes here.
`python ncs_darknetreference/save_keras_graph.py`

### 2) Try model with keras predictions, see if this is correct
`python ncs_darknetreference/predict_converted.py`

### 3) Freeze Graph
Before freezing delete any previously generated models and ckpt in the folder.
`freeze_graph --input_graph=ncs_darknetreference/NN.pb \
           --input_binary=true \
           --input_checkpoint=ncs_darknetreference/NN.ckpt \
           --output_graph=ncs_darknetreference/frozen.pb \
           --output_node_name=Output;`

### 4) Compile NCS in your dev environment, I've used docker.
docker run -v /mnt/foo:/mnt/foo -i -t ncsdk
mvNCCompile -s 12 ncs_darknetreference/frozen.pb -in=Input -on=Output -o ncs_darknetreference/graph -is 256 256

### 5) After compiling the graph test it with a video
I've installed the API in a virtualenv. This setup is just because I couldn't make everything work in docker.
`source /opt/movidius/virtualenv-python/bin/activate`
`python ncs_darknetreference/ncs_predict.py`
