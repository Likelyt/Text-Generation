# long_review_gen
Topic transition model

# Requirements
- python 3.6 
- nltk
- pytorch 0.4


# Data Format - yelp_data
- One line per pair [S_1, S_2]
- See examples in yelp_data/
- Original data in the https://www.yelp.com/dataset/challenge
- Need to download this file: glove.6B.300d.txt 
- create folder: Glove, model_result, testing_result, plot_result


# Running the code 
- load from already generated topic

`python main_mlp.py`

- Run the code, first generate the topic 

`python main.py`
