Implant Global and Local Hierarchy Information to Sequence based Code Representation Models

Accepted by ICPC 2023: https://conf.researchr.org/details/icpc-2023/icpc-2023-research/1/Implant-Global-and-Local-Hierarchy-Information-to-Sequence-based-Code-Representation-

Link to Preprint: https://arxiv.org/pdf/2303.07826.pdf

## [Update Planning]
We are currently **expanding upon the framework presented in this paper**, and in the near future, we will expedite the addition of new experimental code, which will include:

- Hierarchy BPE

- Code Completion Task on Python150k and JS150k

- Pretraining Framework

**Technical details will be fully disclosed upon the new paper's approval for publication.**




## [Note]

- We list source code for four tasks, including code classificaiton, clone detection, method name prediction and variable scope detection.
- We edit config files which contain personal information. So be careful! We will release our full version of source code as soon as possible. *There might be some small adjustments to the file structure on Github that could cause certain bugs. I will organize it when I have some free time later. If you find any issues, feel free to create an issue or send me an email.*
- The preprocessing script for each task is in each dir.
- RAW DATASET Link are from open-source repo:
  - Code Classification: https://github.com/IBM/Project_CodeNet
  - Clone Detection: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104
  - Method Name Prediction: https://github.com/github/CodeSearchNet
  - You can download the raw dataset and preprocess with the script we provided.


### Clone Detection

In the dictionary `Clone-detection-POJ-104\`

`python preprocess_path.py` to process the datasets

`python run.py` to train our model

You can also change the config file in `config.py` for testing, just set `args.test = True`



### Code Classification

In the dictionary `classification\`

`python code_classfication_preprocess.py` and `python processed_tokens_with_path.py` to process the datasets, be careful to change the dataset path in each python file

`python generate_vocab.py` to create the vocab file.

`python run_xxx.py` to train or test different models, we give a example sh in `run.sh`

### Method Name Prediction

In the dictionary `methodname\`

`python preprocess_seq_path.py` to process the datasets

`python __main__.py` to train the model

You can also change the config file in `config.py` for testing, just set `args.test = True`

### Variable Scope Detection

In the dictionary `variable scope detection\`

Use the same datasets with the classification task and make tiny changes

`python probe_for_HiT.py` to train and test the model

You can also change the config file in `probe_config.py` for testing, just set `args.test = True`
