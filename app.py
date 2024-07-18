# inspired by https://www.datacamp.com/tutorial/text-analytics-beginners-nltk

# import libraries
import pandas as pd
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# download nltk corpus (first time only)
import nltk



# contents of 'small_comments.txt'
# comment,positive
# "I have constant headaches from Docker Desktop with regular regressions each update. How is the stability of Podman when it comes to day-to-day work and Docker compatibility? For example I won’t be able to switch away from the Docker engine on our AWS EC2:s, but I might be able to try Podman locally if it can output fully compatible images.",1
# "How good is it for Kubernetes cluster administration? I'm used to using kubectl and I don't know if I am losing something with a Gui like this or lens.",1
# "You can apply YAML, edit, as well as delete any artifacts (pods, services, ingresses, etc.).  It helps create a more visual approach similar to k9s / Lens in a more integrated area (you can see your local containers too through Podman Desktop)",1




# create preprocess_text function
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text


def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment


# used dialog from below to download 'popular' data to /home/rick/nltk_data
# nltk.download()

# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


df = pd.read_csv("/home/rick/Tools/C/Python/Sentiment_Analysis/PodmanD_HackerNews/Sentiment_Analysis_App/small_comments.txt")
print("Did it")
print(df)




# apply the function df, making new column 'reviewText'
df['reviewText'] = df['comment'].apply(preprocess_text)

df['sentiment'] = df['reviewText'].apply(get_sentiment)

print(df)



