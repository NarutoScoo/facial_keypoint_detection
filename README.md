# Facial Keypoint Detection
---

The objective was to detect the landmarks/keypoints on the face of an individual using CNN.
And different transformations where used to make the model robust to detect the landmarks even when the subject is ocluded or looking away.

<div align='center'>
<a name='samples'></a>

|![sample0](./images/markdown_images/sample0.png)|![sample1](./images/markdown_images/sample1.png)|![sample2](./images/markdown_images/sample2.png)|
|:---:|:---:|:---:|
||_Sample Images with Overlayed Keypoints_||
</div>

In addition to the transforms required by the rubric such as  RandomCrop, RandomRotate and Normalize, _Random Shear_ and _RandomScale_ was also added to help the model generalize. All the transformation listed above where custom written (instead of using the pre-defined transformation by PyTorch). [Refer to data_load.py](./data_load.py).

<div align='center'>
<a name='additional_transforms'></a>

|![addition_transforms](./images/markdown_images/additional_transforms.png)|
|:---:|
|_Additional custom transforms - `RandomShear`, `RandomCrop`, `RandomScale`_|
</div>

Further more, multiple variation of the model architecture was tried out (along with different model architectures) and finalized using `NaimishNet`, a network based on LeNet's architecture. (Reference: [NaimishNet](https://arxiv.org/pdf/1710.00977.pdf)), intialized using _He Initialization_ and trained for $50$ epochs with a batch size of $32$. (Refer to [models.py](./models.py) for the network architectures).


<div align='center'>
<a name='architecture'></a>

|![naimishnet_arch](./images/markdown_images/naimishnet_arch.png)|
|:---:|
|_NaimishNet Architecture_|
</div>

The plots below illustrates the different experiments conducted to help finalize the parameters and the architecture

<div align='center'>
<a name='experiments'></a>

|![experiment1](./images/markdown_images/training_loss.png)|![experiment1](./images/markdown_images/training_n_val_loss.png)|
|:---:|:---:|
|_Experiments with different Batch Size, Initialization etc_|_Training and Validation Loss against each experiment_|
</div>


The results/predictions on the test images are shown below
<div align='center'>
<a name='predictions_multiple'></a>

|![preds](./images/markdown_images/predicted_keypoints.png)|
|:---:|
|_Facial Landmark Predictions_|
</div>


<div align='center'>
<a name='predictions'></a>

|![pred0](./images/markdown_images/pred0.png)|![pred1](./images/markdown_images/pred1.png)|
|:---:|:---:|
|_Facial Landmark Predictions_|_Facial Landmark Predictions_|
</div>

To demonstrate the use of facial keypoint detection, the trainined model was used accessories to Christopher_Walken as shown below. The keypoint location guide is used to get the required keypoints (i.e. the one around the eye) to place the glasses.

<div align='center'>
<a name='predictions_multiple'></a>

|![preds](./images/landmarks_numbered.jpg)|
|:---:|
|_Facial Landmark Positions_|
</div>

<div align='center'>
<a name='application'></a>

|![app_orig](./images/markdown_images/fun_app.png)|![pred1](./images/markdown_images/fun_app_pred.png)|
|:---:|:---:|
|_Original Image_|_Adding glasses using the predicted keypoints_|
</div>


---
The project was part of Udacity Computer Vision Nanodegree and the reviewers comments can be found [here](./facial_keypoint_detection_review.pdf).