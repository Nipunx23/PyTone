from textblob import TextBlob

def sentiment_analysis(tweet):
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    tweet['Subjectivity'] = tweet['tweet'].apply(getSubjectivity)
    tweet['Polarity'] = tweet['tweet'].apply(getPolarity)

    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'

    tweet['Analysis'] = tweet['Polarity'].apply(getAnalysis)
    return tweet

import pandas as pd

data = {'tweet': ["This is a positive tweet."]}
tweet = pd.DataFrame(data)

result = sentiment_analysis(tweet)

print(result)
