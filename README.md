<h1 align="center">Foward College</h1>
<h1 align="center">Applied Data Science - Capstone Project</h1>

## Project Title: Personally Identifiable Information (PII) Data Detection

## Problem Statement
The goal of this project is to identify PII in text data. PII detection is important for several reasons:
- <b>Privacy Protection</b>: PII includes sensitive details about individuals, such as names, addresses, email addresses, and phone numbers. Detecting and safeguarding PII ensures that personal data remains private and secure.
- <b>Legal Compliance</b>: Many data protection laws (e.g., GDPR, CCPA) mandate strict handling of PII. Organizations must comply with these regulations to avoid legal penalties.
- <b>Risk Mitigation</b>: Mishandling PII can lead to identity theft, fraud, or unauthorized access. Detecting PII helps mitigate these risks.
- <b>Machine Learning and NLP</b>: Detecting PII is important when training machine learning models, especially for tasks involving natural language processing (NLP). It ensures that models don’t accidentally learn and reveal personal data.

## Challenges
- <b>Variability in PII Formats</b>: PII can appear in various formats (e.g., full names, usernames, initials), making detection challenging.
- <b>Context Dependency</b>: The presence of PII often depends on the context (e.g., an address within a resume vs. an email body has different implications).
- <b>Data Imbalance</b>: PII entities are relatively rare in large datasets. Imbalanced classes affect model training.

## Approach
For this project, i use a dataset from a Kaggle competition ‘[The Learning Agency Lab - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)’. The competition was open from 18<sup>th</sup> January 2024 to 24<sup>th</sup> April 2024 with the goal to develop a model that detects personally identifiable information (PII) in student writing. I aim to train a model with the objective of achieving a high score on the competition leaderboard 🏆.

### __[The Learning Agency Lab - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)__ 
#### Overview
The goal of this competition is to develop a model that detects personally identifiable information (PII) in student writing.

#### Data
##### Data Description
1. The competition dataset comprises approximately 22,000 essays written by students.
2. A majority of the essays are reserved for the test set (70%)

##### PII Entities
- NAME_STUDENT - The full or partial name of a student that is not necessarily the author of the essay. This excludes instructors, authors, and other person names.
- EMAIL - A student’s email address.
- USERNAME - A student's username on any platform.
- ID_NUM - A number or sequence of characters that could be used to identify a student, such as a student ID or a social security number.
- PHONE_NUM - A phone number associated with a student.
- URL_PERSONAL - A URL that might be used to identify a student.
- STREET_ADDRESS - A full or partial street address that is associated with the student, such as their home address.

##### File and Field Information
- The data is presented in JSON format. 
- The documents were tokenized using the SpaCy English tokenizer.
- Token labels are presented in BIO (Beginning, Inner, Outer) format. (e.g.: Waseem Mabunda and Emily -> B-NAME_STUDENT, I-NAME_STUDENT, O, B-NAME_STUDENT)
- {test|train}.json - the test and training data; the test data given on this page is for illustrative purposes only, and will be replaced during Code rerun with a hidden test set.
  1. (int): the index of the essay
  2. document (int): an integer ID of the essay
  3. full_text (string): a UTF-8 representation of the essay
  4. tokens (string): a string representation of each token (list)
  5. trailing_whitespace (bool): a boolean value indicating whether each token is followed by whitespace (list)
  6. labels (string) [training data only]: a token label in BIO format (list)
- Token labels are presented in BIO (Beginning, Inner, Outer) format. 
  - PII type is prefixed with “B-” when it is the beginning of an entity. 
  - PII type is prefixed with “I-” PII type if the token is a continuation of an entity.
  - Tokens that are not PII are labeled “O”.
    
## Project File Structure
- [pii-data-detection-eda.ipynb](https://github.com/Bsting/pii-data-detection/blob/main/leaderboard-eda.ipynb) file, EDA notebook on the PII training dataset.
- [leaderboard-eda.ipynb](https://github.com/Bsting/pii-data-detection/blob/main/leaderboard-eda.ipynb) file, EDA notebook on competition leaderboard data released on 25<sup>th</sup> April 2024.
- [train_inference_notebook](https://github.com/Bsting/pii-data-detection/tree/main/train_inference_notebook) folder, contains notebooks used to train the model and generate submission file for the competition.
- [data](https://github.com/Bsting/pii-data-detection/tree/main/data), data folder. Only contains data file for the leaderboard. Due to upload file size limit, PII training dataset can be downloaded from following:
  - [pii-detection-removal-from-educational-data](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data)
  - [pii-dd-mistral-generated](https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated)
  - [fix-punctuation-tokenization-external-dataset](https://www.kaggle.com/code/valentinwerner/fix-punctuation-tokenization-external-dataset/output)
  - [reate-ai-generated-essays-using-llm](https://www.kaggle.com/datasets/minhsienweng/ai-generated-text-dataset)
  - [pii-mistral-2k-fit-competition-v2](https://www.kaggle.com/datasets/mandrilator/pii-mistral-2k-fit-competition-v2)

<br/>***Due to upload file size limit, no model file is uploaded to this repo***

## Workflow
Workflow use in this project as below
<br>![image](https://github.com/Bsting/pii-data-detection/assets/7638997/60d655d0-5ee5-4479-b3fa-867e4cf0e849)

### 1. Data Collection
Collect data from [The Learning Agency Lab - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data) and perform EDA on the data.

### 2. Data Processing
Process and prepare the data to ensure its suitability for model training.

### 3. Algorithm Selection
Select hyperparameters and a specific variant of a model family for training.

### 4. Model Training
Fit the model with the training data.

### 5. Model Evaluation
Evaluate the model performance by submitting the inference results to the competition. There are 2 scores: the public score and the private score.
- The public score is calculated by using approximately 20% of the test data, and this score is calculated after the inference results successfully submitted.
- The private score is calculated by using approximately 80% of the test data, and this score is calculated after the competition end. 

## Experiment
Observations from the EDA done on the data collected from [The Learning Agency Lab - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data):
- training dataset has imbalanced classes
- 86.12% of the documents with non-PII entity i.e. 'O' label (non-PII) only.
- 99.95% of the labels is non-PII entity ('O' label).
- label 'B-NAME_STUDENT' and 'I-NAME_STUDENT' for entity 'NAME_STUDENT' are have higher frequency compared to others, except 'O' label.
- minimum token length is 69, maximum token length is 3298.
- 31.64% of the documents have token length <= 512.
- for documents with at leaset one PII entity, 16.93% of them have token length <= 512.
- 76.74% of the PII entity tokens at position <= 512 in a document.
- if a pre-trained transformer model receives an input sequence with a maximum length of 512 tokens, any tokens beyond position 512 will not be used during training.

What we can do:
- collect external data to increase label for entities 'ID_NUM', 'EMAIL', 'URL_PERSONAL', 'PHONE_NUM' and 'STREET_ADDRESS'.
- down sample 'O' label.
- use stride in tokenazation process, stride helps break up large document into smaller ones with overlapping tokens.
- split the text to multiple sub-texts.
- upper whisker end of the position of PII token is 1184, we can fine tune a model which accept longer input sequence to include as much as possible PII tokens without striding the tokens.

For more EDA detail, refer [pii-data-detection-eda.ipynb](https://github.com/Bsting/pii-data-detection/blob/main/leaderboard-eda.ipynb).

Experiments done based on the EDA observations
- V1: Fine tune DistilBERT with train data
- V2: Fine tune DistilBERT with train data + stride
- V3: Fine tune DistilBERT with train data + stride + down sample 'O' 
- V4: Fine tune DistilBERT with train data + stride + down sample 'O' + external data
- V5: Fine tune BERT Base with train data + stride + down sample 'O' + external data
- V6: Fine tune DistilBERT with train data + stride + down sample 'O' + external data + evaluation metric F-Beta5
- V7: Fine tune DistilBERT with train data + stride + down sample 'O' + external data + evaluation metric F-Beta5 + cross validation
- V8: Fine tune BERT Base with train data + stride + down sample 'O' + external data + evaluation metric F-Beta5 + cross validation
- V9: Fine-Tune DeBERTa-V3-Small with train data + down sample 'O' + external data + evaluation metric F-Beta5 + cross validation + no Stride with max length 1024
- V10: Fine-Tune DeBERTa-V3-Small with train data + down sample 'O' + external data + evaluation metric F-Beta5 + cross validation + no Stride with max length 2048
- V11: Fine-Tune DeBERTa Base with train data + down sample 'O' + external data + evaluation metric F-Beta5 + cross validation + no Stride with max length 1024

### V2: Fine tune DistilBERT with train data + stride
In order to find the optimal stride value that yields the best evaluation results, the model was trained using various stride values. Evaluation resutls as below
Stride | Public Score
-|-
8 | 0.85732
<b>32</b> | <b>0.86220</b>
64 | 0.87082
128 | 0.86045

<br>Stride value 64 give best public score, and this value was used in V3, V4, V5, V6, V7 and V8.

### V3: Fine tune DistilBERT with train data + stride + down sample 'O' 
To determine the best down-sampling ratio value for ‘O’ label that yields the best evaluation results, the model was trained using different down-sampling ratio. Evaluation resutls as below
Ratio | Public Score
-|-
0.20 | 0.87436
0.30 | 0.87546
<b>0.40</b> | <b>0.86778</b>
0.50 | 0.86380

<br>Ratio 0.30 give best public score, and this value was used in V4, V5, V6, V7 and V8.

### Evaluation Resutls
Experiment | Model | Public Score | Private Score
-|-|-|-
V1 | DistilBERT | 0.85292 | 0.87363
V2 | DistilBERT | 0.87082 | 0.87942
V3 | DistilBERT | 0.87546 | 0.89158
V4 | DistilBERT | 0.88257 | 0.88788
V5 | BERT Base | 0.90284 | 0.90314
V6 | DistilBERT | 0.89153 | 0.89261
V7 | DistilBERT | 0.89589 | 0.89253
V8 | BERT Base | 0.90500 | 0.90788
V9 | DeBERTa-V3-Small | 0.94756 | 0.94219
<b>V10</b> | <b>DeBERTa-V3-Small</b> | <b>0.95791</b> | <b>0.94885</b>
V11 | DeBERTa Base | 0.93357 | 0.92906

V10 achieved the highest evaluation results on both the public and private leaderboards, placing me at rank 1178 out of 2049 on the public leaderboard and 1081 out of 2049 on the private leaderboard. More information of the leaderboard can refer [The Learning Agency Lab - PII Data Detection Leaderboard](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/leaderboard) or [leaderboard-eda](https://github.com/Bsting/pii-data-detection/blob/main/leaderboard-eda.ipynb).

![image](https://github.com/Bsting/pii-data-detection/assets/7638997/f062aeaf-7a27-432c-830c-a5227dfe4cf2)

Notebooks for the experiments refer [train_inference_notebook](https://github.com/Bsting/pii-data-detection/tree/main/train_inference_notebook).

## Key Takeaway
- EDA is crucial in the modeling process.
- Typically, larger models tend to outperform smaller ones.
- Consider experimenting with a small model, as larger model demand greater computational resources and longer training time.
- Models that do well during training might not perform as expected on new, unseen data. Keeping an eye on their performance from time to time is crucial.

## Future Work
- Experiment different pretrained models with different hyperparameters.
  - [Pretrained_Models](https://huggingface.co/transformers/v2.9.1/pretrained_models.html)
  - [microsoft/DeBERTa](https://github.com/microsoft/DeBERTa)
- Experiment ensemble modeling.
- Participatie in more Kaggle competition and learn from others.
