# Constructing Multi-Modal Dialogue Dataset by Replacing Text with Semantically Relevant Images

We present a 45k multi-modal dialogue dataset and the dataset creation method. This dataset is meant for training and evaluating multi-modal dialogue systems. Each multi-modal dialogue instance consists of a textual response and a dialogue context with multiple text utterances and an image. The details used in our creation method can be found in the [paper](temp_link). The work was published in ACL 2021.

## Link to the dataset

The dataset can be found at [drive_link](drive_link).

## Dataset Details

TBD
<!-- There are two files in the above link. "train.json" is meant for training whereas "test.json" is meant for evaluation.
Each line in the file is a json consisting the following keys:`
| Key | Description |
|:--- |:---         |
| id  | Id of the example|
| context | Text of the post |
| cquestion | The corresponding clarification question to the post |
| answer | The answer to the post | -->

## Source Dataset Details

Our multi-modal dialogue dataset is constructed based on 3 source dialogue datasets and 2 image captioning datasets.
We provide download and paper links of all our source datasets.
| Source Dataset | Type | Download link |
|:---            |:---  |:---           |
| DailyDialog [paper](https://www.aclweb.org/anthology/I17-1099/)  | text | [http://yanran.li/dailydialog.html](http://yanran.li/dailydialog.html) |
| Persona-Chat [paper](https://www.aclweb.org/anthology/P18-1205/) | text | [https://parl.ai/about/](https://parl.ai/about/) |
| EmpatheticDialogues [paper](https://www.aclweb.org/anthology/P19-1534/) | text | [https://github.com/facebookresearch/EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) |
| MS-COCO (2014) [paper](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48) | image | [https://cocodataset.org/#download](https://cocodataset.org/#download) |
| Flickr 30k [paper](https://openaccess.thecvf.com/content_iccv_2015/html/Plummer_Flickr30k_Entities_Collecting_ICCV_2015_paper.html) | image | [https://www.kaggle.com/hsankesara/flickr-image-dataset](https://www.kaggle.com/hsankesara/flickr-image-dataset) |


## Code Details 

Before running our code, you have to create Anaconda environment using given enviroment.yaml file.

```bash
conda env create --file environment.yaml
```

we provide two source code sets, similarity-prediction and dialogue-prediction.

With similarity-prediction source code, you can calculate the similarities between source dialogue dataset and image dataset using pretrained VSRN weight.
With dialogue-prediction source code, you can run the current and next dialogue prediction task using our multi-modal dialogue dataset as in the paper.

#### 1. Similarity Prediction
To directly run our similarity-prediction code, you have to download all source dialogue, image datasets, and weight of pre-trained VSRN. Especially for image dataset, we use pre-processed image features in which bottom-up attention is applied. You can download the all image features in here, and the weight of pre-trained VSRN in here.  
After downloading all the necessary dataset and weights in to the dataset directory, then run calculating_similarity.py:

```bash
python similarity-prediction/calculating_similarity.py
```

#### 2. Dialogue Prediction
To run our current and next dialogue prediction task, you have to download our multi-modal dialogue dataset in to the dataset directory. Then, run predicting_dialogue.py:

For current turn prediction task:
```bash
python dialog-prediction/predicting_dialogue.py --model_name $MODEL_NAME --gpu_id $GPU_ID --task current
```

For next turn prediction task:
```bash
python dialog-prediction/predicting_dialogue.py --model_name $MODEL_NAME --gpu_id $GPU_ID --task next
```

## References
If you find the data useful and use it for your work, please consider citing the following:

TBD

<!-- ```
@misc{kumar2020clarq,
    title={ClarQ: A large-scale and diverse dataset for Clarification Question Generation},
    author={Vaibhav Kumar and Alan W. black},
    year={2020},
    eprint={2006.05986},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
``` -->
