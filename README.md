# Modified Steps!

1.	The existing Moments in Time model repository can be found at       https://github.com/zhoubolei/moments_models#models
2.	The repo’s documentation is sparse, so be sure that your environment is configured to meet the following requirements  
- 	Python=3.6.13
- 	Pytorch=1.6.0, torchvision=0.7.0, cudatoolkit=10.1
-	Cv2=3.4.2 (latest is fine)
-	Moviepy=1.0.1 (latest)
-	Ffmpeg (latest)
-	Pyyaml (latest)
3. You should be able to run the 3D CNN model with the original repo's commands if these requirements are met.
4.	In order to run the TRN models, a couple of corrective steps have to be taken
-	Comment out line 35 of the pytorch_load.py file of the BNInception model according to this issue https://github.com/zhoubolei/TRN-pytorch/issues/10
-	(currently only BNInception model works) 
-	(GPU required) change line 107 from 
```
checkpoint = torch.load(args.weights)
```
to
```
checkpoint = torch.load(args.weights, map_location=torch.device('cpu'))
```

5.	The something-somethingv2 models seem to be absent, but changing the example command to run the something-something model should work. Also, one of the demo videos seems to be unavailable, but the other should work. 

```
python test_video.py --arch BNInception --dataset something  --weights pretrain/TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar  --video_file sample_data/juggling.mp4
```

# Running 3D-CNN model on sub-videos

1. To specifiy the region of interest while the video is being streamed and perform sub-video label prediction. Parameters can be edited at the bottom of the file. 

```
python localization_tool.py
```
2. Be careful to only drag from the top left to bottom right, as the dimensions will be incorrectly interpreted if the box is drawn another way. 


# Finetuning 3D-CNN model on new data

1. Decide hyperparameters and specify which video csv files are going to be used by fixing the path in configs/config_file.json
2. Run the training file
```
python training_main_tune.py
```

# Original ReadMe
# Pretrained models for Moments in Time Dataset

We release the pre-trained models trained on [Moments in Time](http://moments.csail.mit.edu/).

### Download the Models

* Clone the code from Github:
```
    git clone https://github.com/metalbubble/moments_models.git
    cd moments_models
```

### Models

* RGB model in PyTorch (ResNet50 pretrained on ImageNet). Run the following [script](test_model.py) to download and run the test sample. The model is tested sucessfully in PyTorch 1.0 + python36.
```
    python test_model.py
```

We provide a 3D ResNet50 (inflated from 2D RGB model) trained on 16 frame inputs at 5 fps.

The model has been recently updated with 305 classes and the following performance on the MiT-V2 dataset:

| Top-1 | Top-5 |
| :---: | :---: |
| 28.4% | 54.5% |

The 3D model can be downloaded and run using a similar command:
```
    python test_video.py --video_file path/to/video.mp4 --arch resnet3d50
```

If you use any of these files please cite our Moments paper (https://arxiv.org/abs/1801.03150).

We now include the Multi-label Moments (M-MiT) 3D Resnet50 Model, Broden dataset with action regions and loss implementations including wLSEP.  If you use any of these files please cite our Multi Moments paper (https://arxiv.org/abs/1911.00232).

The multi-label model has been recently updated with 305 classes and the following performance on the M-MiT-V2 dataset:

| Top-1 | Top-5 | micro mAP | macro mAP |
| :---: | :---: | :---: | :---: |
| 59.4% | 81.7% | 62.4 | 39.4 |

The 3D M-MiT model can be downloaded and run using the following command:
```
    python test_video.py --video_file path/to/video.mp4 --arch resnet3d50 --multi
```

We uploaded a [python file](loss_functions.py) with our pytorch implementations of the different loss functions used in our Multi Moments paper (https://arxiv.org/abs/1911.00232).

In order to [NetDissect](http://netdissect.csail.mit.edu/) Moments models, download the Broden  datasets with action regions:
- [Broden (224x224)](http://data.csail.mit.edu/soundnet/actions3/broden1_224.zip)
- [Broden (227x227)](http://data.csail.mit.edu/soundnet/actions3/broden1_227.zip)
- [Broden (384x384)](http://data.csail.mit.edu/soundnet/actions3/broden1_384.zip)
Note: these can be used with the [PyTorch NetDissect code](https://github.com/CSAILVision/NetDissect-Lite) without modification.

* Dynamic Image model in Caffe: use the [testing script](compute_prob_dynImg.py).

* TRN models is at [this repo](https://github.com/metalbubble/TRN-pytorch). To use the TRN model trained on Moments:

Clone the TRN repo and Download the pretrained TRN model

```
git clone --recursive https://github.com/metalbubble/TRN-pytorch
cd TRN-pytorch/pretrain
./download_models.sh
cd ../sample_data
./download_sample_data.sh
```

Test the pretrained model on the sample video (Bolei is juggling ;-]!)

![result](http://relation.csail.mit.edu/data/bolei_juggling.gif)

```
python test_video.py --arch InceptionV3 --dataset moments \
    --weight pretrain/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar \
    --frame_folder sample_data/bolei_juggling

RESULT ON sample_data/bolei_juggling
0.982 -> juggling
0.003 -> flipping
0.003 -> spinning

```

### Reference

Mathew Monfort, Alex Andonian, Bolei Zhou, Kandan Ramakrishnan, Sarah Adel Bargal, Tom Yan, Lisa Brown, Quanfu Fan, Dan Gutfruend, Carl Vondrick, Aude Oliva. Moments in Time Dataset: one million videos for event understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019. [pdf](https://arxiv.org/pdf/1801.03150.pdf), [bib](http://moments.csail.mit.edu/data/moments.bib)

Mathew Monfort, Kandan Ramakrishnan, Alex Andonian, Barry A McNamara, Alex Lascelles, Bowen Pan, Quanfu Fan, Dan Gutfreund, Rogerio Feris, Aude Oliva. Multi-Moments in Time: Learning and Interpreting Models for Multi-Action Video Understanding. arxiv preprint arXiv:1911.00232, 2019. [pdf](https://arxiv.org/pdf/1911.00232), [bib](http://moments.csail.mit.edu/multi_data/multi_moments.bib)


### Acknowledgements

The project is supported by MIT-IBM Watson AI Lab, IBM Research, the SystemsThatLearn@CSAIL / Ignite Grant and the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/ Interior Business Center (DOI/IBC) contract number D17PC00341.
