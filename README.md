# DeepREL

This is the artifact of the research paper, "Fuzzing Deep-Learning Libraries via Automated Relational API Inference", at ESEC/FSE 2022.

## About

DeepREL is a fully automated end-to-end relational API inference and fuzzing technique for DL libraries, which 1) automatically infers potential API relations based on API syntactic/semantic information, 2) synthesizes concrete test programs for invoking relational APIs, 3) validates the inferred relational APIs via representative test inputs, and finally 4) performs fuzzing on the verified relational APIs to find potential inconsistencies.

This is the DeepREL's implementation for testing PyTorch and TensorFlow.

## Reproduce Bugs

DeepREL has detected 162 bugs in total for PyTorch and TensorFlow. Notably, RelFuzz was able to detect 13.5% of the high-priority bugs for the entire PyTorch issue-tracking system since our first PyTorch bug report. 

We provide a list of confirmed bug reports on [PyTorch](https://github.com/ise-uiuc/DeepREL/blob/main/PyTorch%20Issues.csv) and [TensorFlow](https://github.com/ise-uiuc/DeepREL/blob/main/TF%20issues.csv).
We also provide a Colab notebook to reproduce all 23 high-priority bugs for PyTorch  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ise-uiuc/DeepREL/blob/main/PyTorch_High_Priority_Issue_Reproduction.ipynb).

## Getting Started

### 1. Requirements

1. Our testing framework leverages [MongoDB](https://www.mongodb.com/) so you should [install and run MongoDB](https://docs.mongodb.com/manual/installation/) first.
	- Run the command `ulimit -n 200000` to adjust the limit that the system resources a process may use. You can see this [document](https://docs.mongodb.com/manual/reference/ulimit/) for more details.
2. You should check our dependent python libraries in `requirements.txt` and run `pip install -r requirements.txt` to install them
3. Python version >= 3.9.0 (It must support f-string.)

### 2. Setting Up with Dataset

Run the following commands to load the database.

```shell
mongorestore -h 127.0.0.1:27017 --db tf dump/tf/
mongorestore -h 127.0.0.1:27017 --db torch dump/torch/
```


### 3. Run

After finishing above steps, run the following command to start RelFuzz to test PyTorch

```shell
cd pytorch/src && python DeepREL.py
```

Run the following command to start RelFuzz to test TensorFlow

```shell
cd tensorflow/src && python DeepREL.py
```

### 4. Configuration

There are some hyper-parameters in DeepREL and they could be easily configured by modifying `{pytorch, tensorflow}/src/config/demo.conf`:

1. MongoDB database configuration.

```conf
[mongodb]
# your-mongodb-server
host = 127.0.0.1
# mongodb port
port = 27017 
# name of pytorch database
torch_database = torch
# name of tensorflow database
tf_database = tf
```

2. DeepREL configuration
```conf
[DeepREL]
test_number = 1000
top_k = 10
iteration = 10
```

