

## Automatic Lung Parenchyma Segmentation using a Deep Convolutional Neural Network from Chest X-Rays

### Abstract
To detect and diagnosis the lungs related diseases, a Chest X-Ray (CXR) is the major tool used by the physician. In this paper, an efficient automatic CAD system is proposed to detect the boundaries using a deep convolutional neural network (DCNN) model. The DCNN is trained in an end-to-end setting to facilitate fully automatic lung segmentation from anteroposterior or posteroanterior view CXRs. It learns to predict binary masks for a given CXR, by learning to discriminate regions of organ parenchyma from regions of no organ parenchyma. The proposed model’s architecture makes use of residual connections in all the concurrent up-sampling paths from each encoder block at every level, thus facilitating collective learning within blocks through inter-sharing of all high-dimensional feature maps. To generalize the proposed model to CXRs from all data distributions, image preprocessing techniques such as Top-Hat Bottom-Hat Transform and Contrast Limited Adaptive Histogram Equalization are employed. The proposed model is trained and tested using the JSRT, NLM-MC and Shenzhen Hospital datasets. The proposed method achieved a Dice Similarity Coefficient of 0.982 ± 0.018 and a Jaccard Similarity Coefficient of 0.967 ± 0.015. The implementation results demonstrated that the proposed method has surpassed the existing methods and our model is relatively lightweight and can be easily implemented on standard GPUs.

**NOTE:**

- Link to publication - [https://doi.org/10.1016/j.bspc.2021.103398](https://doi.org/10.1016/j.bspc.2021.103398)
- The entire modeling, simulation and analysis has been done on the Google Colab environment.
- The following instructions will be given in the same order as the proposed methodology of the project, i.e., Data Acquisition, Data Pre-processing, Data Augmentation and Deep Convolutional Neural Network (DCNN) Modeling, Experimental Results, Deployment of DCNN Model.

**INSTRUCTIONS:**

_Data Acquisition:_

1. The dataset used for the DCNN model were acquired through sequential downloads from the websites of the Japanese Society of Radiology &amp; Technology (JSRT) and the National Library of Medicine (NLM). The links for both are as follows:

- JSRT: [http://db.jsrt.or.jp/eng.php](http://db.jsrt.or.jp/eng.php)

- NLM: [http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip](http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip)

2. The datasets have been downloaded and combined and the same is present in the folder titled **data**.

_Data Pre-processing:_

1. For pre-processing the Chest X-Rays (CXRs) with Top-Hat Bottom-Hat Transform (TBH) followed by Contrast Limited Adaptive Histogram Equalization (CLAHE), run the script **generate\_dataset\_tbh+clahes​.​py**, after changing the directories.
2. The dataset after pre-processing with TBH+CLAHE is also in the folder titled **tbh+clahe**.

_Data Augmentation and DCNN Modeling_:

Sequentially running all the cells of the Google Colab Jupyter notebook **train\_lungSeg\_unetplusplus\_model.ipynb** does the following:

1. Load the dataset **tbh+clahe.zip** and install efficient net.
2. Define Image Data Generators to augment the loaded Chest X-Ray datasets on-the-go.
3. Define all the mathematical loss functions required for driving the model&#39;s training.
4. Define the custom Cyclic Learning Rate Schedule and set the appropriate condition values.
5. Design and develop our proposed U-Net++ DCNN Model and subsequently compile and train the model using the augmented dataset for 250 epochs.
6. Save the trained DCNN model as **net\_lung\_seg.hdf5**
7. Plot the training history w.r.t. Jaccard Co-efficient, Dice Similarity Co-efficient, Loss Function, CLR Policy and Binary Cross-Entropy.

_Experimental Results:_

Sequentially running all the cells of **evaluate\_model.ipynb** does the following:

1. Load the trained model **net\_lung\_seg.hdf5** and install efficient net.
2. Define all the pertinent mathematical loss functions, as well as load the test dataset.
3. Define Image Data Generators to load the test Chest X-Ray dataset on-the-go.
4. Print all the evaluation metrics (Loss, Binary Cross Entropy, Dice Coefficient, Jaccard Coefficient, Binary Accuracy) for the validation and test data.

_Deployment of DCNN Model:_

Running the python script **model\_create\_mask.py** does the following:

1. Load the trained model **net\_lung\_seg.hdf5** and install efficient net.
2. Define all the pertinent mathematical loss functions, as well as load the test dataset.
3. Define Image Data Generators to load the test Chest X-Ray dataset on-the-go.
4. Create predicted masks for the test CXR dataset.

**NOTE:** For predicting the mask of any random CXR, add the corresponding CXR to the respective test directory and then run the python script **model\_create\_mask.py**. The resulting masks of the test CXRs, the mask difference between predicted test CXR masks and original test CXR masks, as well as segmented test CXRs are present in the folder titled **results**.
