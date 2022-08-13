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
| Laptops SemEval-14 | 83.8 |
