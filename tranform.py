import string
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import re
import subprocess
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def process_text(text, use_tweet_tokenizer=False, remove_punctuation=True, remove_stopwords=True, lemmatize=True, stem=True, use_vw=False):
    try:
        if use_tweet_tokenizer:
            tokenizer = TweetTokenizer()
        else:
            tokenizer = word_tokenize
        tokens = tokenizer(text.lower())
            if remove_punctuation:
            table = str.maketrans('', '', string.punctuation)
            tokens = [token.translate(table) for token in tokens]
        
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize the tokens
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        if stem:
            stemmer = SnowballStemmer('english')
            tokens = [stemmer.stem(token) for token in tokens]
            processed_text = ' '.join(tokens)
        
        if use_vw:
            processed_text = '1 | ' + re.sub(r'\s+', ' ', processed_text)
            vw_output = subprocess.check_output(['vw', '-i', 'model.vw', '-t', '-'], input=processed_text, text=True)
            vw_prediction = float(vw_output.split()[0])
            return vw_prediction
        
        else:
            fdist = nltk.FreqDist(tokens)

            # Plot the top 20 most common tokens
            plt.figure(figsize=(12,6))
            fdist.plot(20)
            plt.title('Top 20 Most Common Tokens')
            plt.xlabel('Token')
            plt.ylabel('Frequency')
            plt.show()

            # Plot a histogram of token frequencies
            plt.figure(figsize=(12,6))
            sns.histplot(list(fdist.values()), bins=50)
            plt.title('Distribution of Token Frequencies')
            plt.xlabel('Frequency')
            plt.ylabel('Count')
            plt.show()

            return processed_text
    
    except Exception as e:
        print(f"Error processing text: {e}")
        return None