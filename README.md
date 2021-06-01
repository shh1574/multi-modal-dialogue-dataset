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

TBD

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
