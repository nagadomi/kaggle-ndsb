# Kaggle-NDSB

Code for [National Data Science Bowl](https://www.kaggle.com/c/datasciencebowl) at Kaggle. Ranked 10th/1049.

# Summary

Ensemble Deep CNNs trained with real-time data augmentation.

<table>
  <tr>
    <td>
      Preprocessing
    </td>
    <td>
      centering, convert to a fixed length with padding, convert to a negative.
      <table>
        <tr>
	  <th>Source</th>
	  <th>Destination</th>
	</tr>
	<tr>
	  <td><img src="https://raw.githubusercontent.com/nagadomi/kaggle-ndsb/master/figure/preprocess_before_1.png"></td>
	  <td><img src="https://raw.githubusercontent.com/nagadomi/kaggle-ndsb/master/figure/preprocess_after_1.png"></td>
	</tr>
	<tr>
	  <td><img src="https://raw.githubusercontent.com/nagadomi/kaggle-ndsb/master/figure/preprocess_before_4.png"></td>
	  <td><img src="https://raw.githubusercontent.com/nagadomi/kaggle-ndsb/master/figure/preprocess_after_4.png"></td>
	</tr>
	<tr>
	  <td><img src="https://raw.githubusercontent.com/nagadomi/kaggle-ndsb/master/figure/preprocess_before_5.png"></td>
	  <td><img src="https://raw.githubusercontent.com/nagadomi/kaggle-ndsb/master/figure/preprocess_after_5.png"></td>
	</tr>
      </table>
    </td>
  </tr>
  <tr>
    <td>
      Data augmentation
    </td>
    <td>
      real-time data agumentation (apply the random transformation each minibatchs).
      transformation method includes translation, scaling, rotation, perspective cropping and contrast scaling.<br/>
      <img src="https://raw.githubusercontent.com/nagadomi/kaggle-ndsb/master/figure/random_transform.png">
      <img src="https://raw.githubusercontent.com/nagadomi/kaggle-ndsb/master/figure/random_transform_grid.png">
    </td>
  </tr>
  <tr>
    <td>
     Neural Network Architecture
    </td>
    <td>
      Three CNN architectures for different rescaling inputs. 
      <a href="https://github.com/nagadomi/kaggle-ndsb/blob/master/cnn_96x96.lua">cnn_96x96</a>,
      <a href="https://github.com/nagadomi/kaggle-ndsb/blob/master/cnn_72x72.lua">cnn_72x72</a>,
      <a href="https://github.com/nagadomi/kaggle-ndsb/blob/master/cnn_48x48.lua">cnn_48x48</a>
    </td>
  </tr>
  <tr>
    <td>
      Normalization
    </td>
    <td>
     Global Contrast Normalization (GCN)
    </td>
  </tr>
  <tr>
    <td>
      Optimization method
    </td>
    <td>
     minibatch-SGD with Nesterov momentum.
    </td>
  </tr>
  <tr>
    <td>
      Results
    </td>
    <td>
      <table>
        <tr>
	  <th>Model</th>
	  <th>Public LB score</th>
	</tr>
	<tr>
	  <td> cnn_48x48 single model</td>
	  <td> 0.6718 </td>
	</tr>
	<tr>
	  <td> cnn_72x72 single model</td>
	  <td> 0.6487 </td>
	</tr>
	<tr>
	  <td> cnn_96x96 single model</td>
	  <td> 0.6561 </td>
	</tr>
	<tr>
	  <td> cnn_48x48 average of 8 models</td>
	  <td> 0.6507 </td>
	</tr>
	<tr>
	  <td> cnn_72x72 average of 8 models</td>
	  <td> 0.6279 </td>
	</tr>
	<tr>
	  <td> cnn_96x96 average of 8 models</td>
	  <td> 0.6311 </td>
	</tr>
	<tr>
	  <td> ensemble (cnn_48x48(x8) * 0.2292 + cnn_72x72(x8) * 0.3494 + cnn_96x96(x8) * 0.4212 + 9.828e-6)</td>
	  <td> 0.6160 </td>
	</tr>
      </table>
    </td>
  </tr>
</table>

## Developer Environment

- Ubuntu 14.04
- 16GB RAM 
- GPU & CUDA (I used EC2 g2.2xlarge instance)
- [Torch7](http://torch.ch/)
- [NVIDIA CuDNN](https://developer.nvidia.com/cuDNN)
- [cudnn.torch](https://github.com/soumith/cudnn.torch)

## Installation

Install CUDA, Torch7, NVIDIA CuDNN, cudnn.torch.

### Checking CUDA environment

    th cuda_test.lua

Please check your Torch7/CUDA environment when this code fails.

### Convert dataset

Place the [data files](https://www.kaggle.com/c/datasciencebowl/data) into a subfolder ./data.

    ls ./data
    test  train  train.txt test.txt classess.txt
-
    th convert_data.lua

### Training, Validation, Make submission

training & validate single cnn_48x48 model.

    th train.lua -model 48 -seed 101
    ls -la models/cnn*.t7

make submission file.

    th predict.lua -model 48 -seed 101
    ls -la models/submission*.txt

when use cnn_72x72 model.

    th train.lua -model 72 -seed 101
    th predict.lua -model 72 -seed 101
    
when use cnn_96x96 model.

    th train.lua -model 96 -seed 101
    th predict.lua -model 96 -seed 101

### Ensemble

This task is very heavy. I used x20 g2.xlarge instances for this task and it's takes 4 days.

(helper tool can be found at ./appendix folder.)

    th train.lua -model 48 -seed 101
    th train.lua -model 48 -seed 102
    th train.lua -model 48 -seed 103
    th train.lua -model 48 -seed 104
    th train.lua -model 48 -seed 105
    th train.lua -model 48 -seed 106
    th train.lua -model 48 -seed 108
    th train.lua -model 72 -seed 101
    th train.lua -model 72 -seed 102
    th train.lua -model 72 -seed 103
    th train.lua -model 72 -seed 104
    th train.lua -model 72 -seed 105
    th train.lua -model 72 -seed 106
    th train.lua -model 72 -seed 108
    th train.lua -model 96 -seed 101
    th train.lua -model 96 -seed 102
    th train.lua -model 96 -seed 103
    th train.lua -model 96 -seed 104
    th train.lua -model 96 -seed 105
    th train.lua -model 96 -seed 106
    th train.lua -model 96 -seed 108
    
    th predict -model 48 -seed 101
    th predict -model 48 -seed 102
    th predict -model 48 -seed 103
    th predict -model 48 -seed 104
    th predict -model 48 -seed 105
    th predict -model 48 -seed 106
    th predict -model 48 -seed 108
    th predict -model 72 -seed 101
    th predict -model 72 -seed 102
    th predict -model 72 -seed 103
    th predict -model 72 -seed 104
    th predict -model 72 -seed 105
    th predict -model 72 -seed 106
    th predict -model 72 -seed 108
    th predict -model 96 -seed 101
    th predict -model 96 -seed 102
    th predict -model 96 -seed 103
    th predict -model 96 -seed 104
    th predict -model 96 -seed 105
    th predict -model 96 -seed 106
    th predict -model 96 -seed 108

    th ensemble.lua
