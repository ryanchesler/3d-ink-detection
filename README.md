# 3d-ink-detection
Expanding ink detection to 3d so it can be scaled to all areas and new scrolls

In this repo I am providing the data prep, training and inference pipelines as well as a very well trained 3d unet for 3d ink detection of the vesuvius scrolls and the 3d ink detected volumes the model has already been run against. 

## Scroll 1 Ink detection video
[![Scroll 1](https://img.youtube.com/vi/eXHKyKgKAr0/0.jpg)](https://www.youtube.com/watch?v=eXHKyKgKAr0)

## Scroll 2 Ink detection video(slight truncation because inference got cancelled early)
[![Scroll 1](https://img.youtube.com/vi/WvDKq0YaoVA/0.jpg)](https://www.youtube.com/watch?v=WvDKq0YaoVA)

## Scroll 3 Ink detection video
[![Scroll 1](https://img.youtube.com/vi/TFgKJRuvXxU/0.jpg)](https://www.youtube.com/watch?v=TFgKJRuvXxU)

## Scroll 4 Ink detection video
[![Scroll 1](https://img.youtube.com/vi/bs7tYjuGDEo/0.jpg)](https://www.youtube.com/watch?v=bs7tYjuGDEo)

## Data prep
In order to train a model you will need the original scan volumes converted to an hdf5 array. This can be done with some minor modification to paths from this script https://github.com/ryanchesler/LSM/blob/main/volume_to_hdf.py
On top of that you will also need some 2d predictions to map back to the 3d volume. You can use predictions from any decent 2d model, in a lot of the experiments I used my own unetr-> segformer pipeline, but another starting point can be the labels provided in grand prize winning repository https://github.com/younader/Vesuvius-Grandprize-Winner/tree/main/all_labels. To take these 2d labels and map them back to 3d you will need the ppms for each segment in the directory. These can be downloaded from the official data server here https://scrollprize.org/data_scrolls#data-server

Now that you have these bits of data gathered up you can run `python all_labels_to_hdf.py` with some minor modifications to paths. This will initially create a new training hdf5 array to represent the labels in 3d. In order to make a validation array we will need to make some minor modifications to the all_labels_to_hdf in order to make it set aside `20230904135535`. Rerun the command and now you will have both a training and validation array.

Now final data preparation step is to run pred_find_coords.py. This will scan through the label volume and note the coordinates where it was actually labeled, we could randomly sample from the space to find it all the time but more efficient to just know the coordinates and have easy access of the valid points to train in. Run this once on the training volume and once on the validation volume with minor adjustments to the path in order to get the numpy arrays representing the valid coords to train against. 

![image](https://github.com/ryanchesler/3d-ink-detection/assets/26419230/1da29f11-483b-4099-b938-29c0fb2f2a5d)


## Training

Now that you have the hdf5 volume of the scroll scan, the hdf5 volume for training and validation labels mapped from 2d to 3d and the coordinate numpy arrays you are now ready to run training. In order to do this some paths will need to be updated and wandb will need to be configured. Update paths to your new hdf5 arrays and numpy arrays. A couple notes on the details of training. It is highly recommended to use the checkpoint from the 3d pretraining repository, in shorter training runs this can have a significant impact, if you train for a long time it seems to matter less. This model is a 3d unet that is shaped to take in a 128x128x128 input volume and output back a 128x128x128 ink detection volume. One of the major difficulties of this is that we only have sparse ribbons of labeled data from these 2d sheets that have been segmented out from the 3d. It is very important to train this in the correct way, treating the unlabeled data as unlabeled and not penalizing the model for predictions that arent no ink, but are rather just no label. In order to do this all of the labeled areas get real values and all of the unlabeled errors are masked off by -1's. These values will be excluded from the loss function so no gradient is calculated for them to update the model. Without this the model will bias towards always predicting no ink. One other crucial element is proper validation. As an extra precaution to make sure that we can truly measure the performance of the model on unseen regions of the scroll a quality check is done that will exclude any coordinates that are within 512 euclidean distance of any validation points. This is rather aggressive, taking a full 512 block away from any labeled point in the validation volume, but it is paramount that we do this in order to see how the model actually performs on new data. This code uses volumentation in order to make the model robust to many different scroll scenarios. Rotations, flips, some intensity variation and crop and rescaling, make it slightly scale invariant. 

<img width="1544" alt="Screenshot 2024-04-30 at 11 26 51 PM" src="https://github.com/ryanchesler/3d-ink-detection/assets/26419230/c10bbedd-89ca-459d-ab20-bdfc4f869c41">

Some training results

Depending on hardware this can train for many hours until convergence

## Inference


Now with a model that has been trained or reusing the shared checkpoint, you are ready to run inference on a scroll. The scroll scan of interest needs to be hdf5 formatted, if you have already done this during training then you can reuse it. If you want to use it on a new scroll scan then you will need to run the process again and update the path to the array you want. Now this process will run for a while depending on the size and resolution of the scroll. Inference is done in a sliding window manner since it is only trained on 128x128x128 at a time. A more rigourous inference can be done with a smaller stride length but it will come at the cost of greatly increased inference time. 

## 3D prediction back to 2D
Now with the 3d predictions you can do several things. One of them is use existing segment ppms in order to unroll the 3d prediction to a flat sheet again. This will yield something like this

<img width="1617" alt="Screenshot_2024-02-12_at_9 29 38_AM" src="https://github.com/ryanchesler/3d-ink-detection/assets/26419230/4944117e-a3d2-4200-b1d0-8d35c2a7bec5">

This can be done with the flatten_3d.py script. Some paths will need to be updated for the prediction volume and the ppms of your target segments

Something alternate you can do with the 3d predictions is inspect the connected components to look for new letters. The connected_components_explore.py script will try to look for reasonably sized blobs and export them as multiple views of the 3d image to quickly flip through. This is a great way to find new letters in unexplored parts of the scrolls. So far it has found new stuff in scroll 1 which it was trained on but inconclusive results on scrolls 2-4. 

<img width="745" alt="Screenshot_2024-03-07_at_1 03 53_PM" src="https://github.com/ryanchesler/3d-ink-detection/assets/26419230/e5ebc7fb-e302-40a5-b599-800a2ae89446">

Subsets of the prediction volume can also be export to jpg or other file formats again and visualized in slicer with the image stacks extension. This gives a lot of nice tooling to inspect and probe the predictions 

<img width="793" alt="Screenshot_2024-03-07_at_12 49 16_PM" src="https://github.com/ryanchesler/3d-ink-detection/assets/26419230/a7f249e6-2fa8-4cf5-af93-774076794307">

