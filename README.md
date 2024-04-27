<h1 align="center">Foward College</h1>
<h1 align="center">Applied Data Science - Capstone Project</h1>

# Project Title: Personally Identifiable Information (PII) Data Detection

## Problem Statement
The goal of this project is to identify PII in text data. PII detection is important for several reasons:
- <b>Privacy Protection</b>: PII includes sensitive details about individuals, such as names, addresses, email addresses, and phone numbers. Detecting and safeguarding PII ensures that personal data remains private and secure.
- <b>Legal Compliance</b>: Many data protection laws (e.g., GDPR, CCPA) mandate strict handling of PII. Organizations must comply with these regulations to avoid legal penalties.
- <b>Risk Mitigation</b>: Mishandling PII can lead to identity theft, fraud, or unauthorized access. Detecting PII helps mitigate these risks.
- <b>Machine Learning and NLP</b>: Detecting PII is important when training machine learning models, especially for tasks involving natural language processing (NLP). It ensures that models don’t accidentally learn and reveal personal data.

## Challenges
- <b>Variability in PII Formats</b>: PII can appear in various formats (e.g., full names, usernames, initials), making detection challenging.
- <b>Context Dependency</b>: The presence of PII often depends on the context (e.g., an address within a resume vs. an email body has different implicationsl).
- <b>Data Imbalance</b>: PII entities are relatively rare in large datasets. Imbalanced classes affect model training.

## Approach
For this project, we use a dataset from a Kaggle competition ‘__[The Learning Agency Lab - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)__’. The competition was open from 18<sup>th</sup> January 2024 to 24<sup>th</sup> April 2024 with the goal of training a model to achieve a high score on the leaderboard.

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

## Project File Structure
- pii-data-detection-eda.ipynb file, EDA notebook on the PII training dataset.
- leaderboard-eda.ipynb file, EDA notebook on competition leaderboard data released on 25<sup>th</sup> April 2024.
- train_inference_notebook folder, contains notebooks used to train the model and generate submission file for the competition.
- data, data folder. Only contains data file for the leaderboard. Due to upload file size limit, PII training dataset can be downloaded from following:
  - __[pii-detection-removal-from-educational-data](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data)__
  - __[pii-dd-mistral-generated](https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated)__
  - __[fix-punctuation-tokenization-external-dataset](https://www.kaggle.com/code/valentinwerner/fix-punctuation-tokenization-external-dataset/output)__
  - __[reate-ai-generated-essays-using-llm](https://www.kaggle.com/datasets/minhsienweng/ai-generated-text-dataset)__
  - __[pii-mistral-2k-fit-competition-v2](https://www.kaggle.com/datasets/mandrilator/pii-mistral-2k-fit-competition-v2)__

<br/>***Due to upload file size limit, no model file is uploaded to this repo***
