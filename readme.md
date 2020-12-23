## 1. Introduction

Random Forest is an important ensemble learning method for classification andregression. Random forest algorithm is composed of several separate decision trees,these trees are trained with different combinations of features in the dataset. 

My code is evaluated on the dataset provided by [Tianchi Big Data Competition](https://tianchi.aliyun.com/competition/entrance/531830/information), the goal is to predict loan default based on applicant's information. 


## 2. Quick Start
First, download `csv` files from [Tianchi Big Data Competition](https://tianchi.aliyun.com/competition/entrance/531830/information), and put them to `data` folder.

Then Process the `csv` data with utility functions in `utils.py`. You can also download the data from [here](https://drive.google.com/file/d/15BC3JWSMpokUSPNCcP5sMA41zqwXmwnh/view?usp=sharing). Then the directory should look like this.

```shell
CS235_Random_Forest/
├── data
│   ├── sample_submit.csv
│   ├── testA.csv
│   └── train.csv
├── output
│   ├── train_partial.pkl
│   └── train.pkl
├── decision_tree.py
├── random_forest.py
├── readme.md
├── requirements.txt
└── utils.py
```

### 2.1 Install packages

```shell
pip install -r requirements.txt
```

### 2.2 Decision tree

python decision_tree.py

### 2.3 Random forest

python random_forest.py

