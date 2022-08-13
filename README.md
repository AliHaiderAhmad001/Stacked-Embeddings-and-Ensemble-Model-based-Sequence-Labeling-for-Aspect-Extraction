# **Stacked-Embeddings-and-Ensemble-Model-based-Sequence-Labeling-for-Aspect-Extraction**
One key task of fine-grained sentiment analysis of product reviews is to extract product aspects or features that users have expressed opinions on. Aspect extraction is an important task in sentiment analysis (Hu and Liu, 2004) and has many applications (Liu, 2012). It aims to extract opinion targets (or aspects) from opinion text. In product reviews, aspects are product attributes or features. For example, from “Its speed is incredible” in a laptop review, it aims to extract “speed”.

## Task
Given a sentence, the task is to extract aspects. Here is an example:
```
"I like the battery life of this phone"

Converting this sentence to IOB would look like this -

I O
like O
the O
battery B-A
life I-A
of O
this O
phone O
```
## Result
| Dataset | F1-Score |
| -------- | -------- |
| Restaurants SemEval-14 | 89.3 |
| Restaurants SemEval-16 | 79.8 |
| Laptops SemEval-14 | 83.6 |

## Environment
All code are tested under python 3.9.8 + tensorflow 2.9.1
### Dependencies
* python 3.9.8
* tensorflow 2.9.1
* transformers 4.21.1
* seqeval 1.2.2
* flairNLP 0.11

## Work steps:
1. Download Datasets:
  Datasets are in xml format
  * SemEval 2014 Laptop and Restaurant (http://alt.qcri.org/semeval2014/task4/).
  * SemEval 2016 Restaurant (http://alt.qcri.org/semeval2016/task5/).
2. Extract data from XML files and store in IOB files using "adapter.py"
3. Generate embeddings using "prep_final_data.py".
4. Training SE-EM using "train.py".
