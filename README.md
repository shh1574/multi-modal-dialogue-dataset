# Constructing Multi-Modal Dialogue Dataset by Replacing Text with Semantically Relevant Images

We present a 45k multi-modal dialogue dataset and the dataset creation method. This dataset is meant for training and evaluating multi-modal dialogue systems. Each multi-modal dialogue instance consists of a textual response and a dialogue context with multiple text utterances and an image. The details used in our creation method can be found in the [paper](TBD). The work was published in ACL 2021.

## Link to the dataset

The dataset can be found at [drive_link](https://drive.google.com/drive/folders/12-Zz4MJTASJVlbncpSWvBVqLDe5_m5QU?usp=sharing).

## Dataset Details

There are 3 files in the above link. Each zip(or egg) file compressed json and npy format files for training and evaluation.
Each line in the json file is a json consisting the following keys:`
| Key | Description |
|:--- |:---         |
| dialog  | Dialogue context and response |
| replaced_idx | Index(turn) of dialogue context to be replaced |
| img_idx | Index of image tensor to replace in the npy file |
| score | The similarity score between |
| dialog_dataset | Source dialogue dataset |
| dialog_file | Used file name in the source dialogue dataset |
| img_dataset | Source image dataset |
| img_file | Used file name in the source image dataset |

{"dialog": ["hello , how are you tonight ?", "i am tired . i played all day long .", "what did you play ? i walked dogs for nine miles total .", "i played with my toys in my room .", "do you like dogs ? i have three .", "yes . i like dogs . what kind are they ?", "i have a lab and a mixed breed .", "my best friend has a cat . she lives next door to me .", "i like cats . do you play with her a lot ?", "yes . we play with our dolls together . how old are you ?", "i am 26 . do you need a babysitter ?", "i am only four years old . i do not like my babysitter .", "tell your parents . you should always like the one who is watching you .", "she gets mad at me when i do not clean my room ."], "replaced_idx": 4, "img_idx": 62, "score": 0.5801908360436532, "dialog_dataset": "persona", "dialog_file": "chat10005.txt", "img_dataset": "coco", "img_file": "COCO_val2014_000000283210.jpg"}

## Source Dataset Details

Our multi-modal dialogue dataset is constructed based on 3 source dialogue datasets and 2 image captioning datasets.
We provide download and paper links of all our source datasets.
| Source Dataset | Paper | Type | Download link |
|:---            |:---   |:---  |:---           |
| DailyDialog | [paper](https://www.aclweb.org/anthology/I17-1099/)  | text | [http://yanran.li/dailydialog.html](http://yanran.li/dailydialog.html) |
| Persona-Chat | [paper](https://www.aclweb.org/anthology/P18-1205/) | text | [https://parl.ai/about/](https://parl.ai/about/) |
| EmpatheticDialogues | [paper](https://www.aclweb.org/anthology/P19-1534/) | text | [https://github.com/facebookresearch/EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) |
| MS-COCO (2014) | [paper](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48) | image | [https://cocodataset.org/#download](https://cocodataset.org/#download) |
| Flickr 30k | [paper](https://openaccess.thecvf.com/content_iccv_2015/html/Plummer_Flickr30k_Entities_Collecting_ICCV_2015_paper.html) | image | [https://www.kaggle.com/hsankesara/flickr-image-dataset](https://www.kaggle.com/hsankesara/flickr-image-dataset) |


## Code Details 

Before running our code, you have to create Anaconda environment using given enviroment.yaml file.

```bash
conda env create --file environment.yaml
```

we provide two source code sets, similarity-prediction and dialogue-prediction.

With similarity-prediction source code, you can calculate the similarities between source dialogue dataset and image dataset using pretrained VSRN weight.
With dialogue-prediction source code, you can run the current and next dialogue prediction task using our multi-modal dialogue dataset as in the paper.

#### 1. Similarity Prediction
To directly run our similarity-prediction code, you have to download all source dialogue, image datasets, and weight of pre-trained VSRN. Especially for image dataset, we use pre-processed image features in which bottom-up attention is applied. You can download the all image features in [here](TBD), and the weight of pre-trained VSRN in [here](https://drive.google.com/drive/folders/1zUgma0SD4Dp3b3n55pv7QAsTBFRF_6m8?usp=sharing).  
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
