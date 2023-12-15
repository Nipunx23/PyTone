import re 
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import itertools
def read_data(file):
    data = []
    with open(file, 'r')as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data

file = 'C:/Users/asus/OneDrive/Desktop/VS/PYTHON/SPEECH_EMOTION_RECOG_using_text/data.txt'
data = read_data(file)
print("Number of instances: {}".format(len(data)))

#to create tokens and generating the features of an input sentence


#An n-gram is a contiguous sequence of n items from a given sample of text or speech. 
#The function creates n-grams of the specified length and returns them as a list.
def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

#create_feature function is designed to generate text features from input text. 
#It tokenizes the input text, converts it to lowercase, and then separates alphanumeric characters and punctuation.
#It generates n-grams from the alphanumeric tokens and single words from the punctuation tokens, counts the occurrences of these features, and returns them as a Counter object.

def create_feature(text, nrange=(1, 1)):
    text_features = [] 
    text = text.lower() 
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)    
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)

#print(ngram("hello, I am VAIBHAV",19))
#print(create_feature("hello, I am VAIBHAV",(1,1)))

##Python function to store the labels, our labels will be based on emotions such as Joy, Fear, Anger, and so on:
def convert_label(item, name): 
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)): 
        if items[idx] == 1: 
            label += name[idx] + " "
    
    return label.strip()

emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

X_all = []
y_all = []
for label, text in data:
    y_all.append(convert_label(label, emotions))
    X_all.append(create_feature(text, nrange=(1, 4)))

#split the data in training and data sets

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = 123)

def train_test(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    return train_acc, test_acc

from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer(sparse = True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


#choosing 4 models and decide which one is best
svc = SVC()
lsvc = LinearSVC(random_state=123, dual='auto', max_iter=10000)
rforest = RandomForestClassifier(random_state=123)
dtree = DecisionTreeClassifier()


clifs = [ lsvc]

# train and test them 
print("| {:25} | {} | {} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
print("| {} | {} | {} |".format("-"*25, "-"*17, "-"*13))
for clf in clifs: 
    clf_name = clf.__class__.__name__
    train_acc, test_acc = train_test(clf, X_train, X_test, y_train, y_test)
    print("| {:25} | {:17.7f} | {:13.7f} |".format(clf_name, train_acc, test_acc))

#


l = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
l.sort()
label_freq = {}
for label, _ in data: 
    label_freq[label] = label_freq.get(label, 0) + 1

# print the labels and their counts in sorted order 
for l in sorted(label_freq, key=label_freq.get, reverse=True):
    print("{:10}({})  {}".format(convert_label(l, emotions), l, label_freq[l]))
emoji_dict = {"joy": "😂", "fear": "😱", "anger": "😠", "sadness": "😢", "disgust": "😒", "shame": "😳", "guilt": "😔"}

# Get the list of emotions
emotions = list(emoji_dict.keys())

# Generate all permutations of emotions
emotion_permutations = list(itertools.permutations(emotions, 2))

# Update the emoji_dict with permutations and emojis
for emotion_pair in emotion_permutations:
    emoji_dict[f"{emotion_pair[0]} {emotion_pair[1]}"] = emoji_dict[emotion_pair[0]] + emoji_dict[emotion_pair[1]]


t1 = "This looks so impressive"
t2 = "I have a fear of dogs"
t3 = "My dog died yesterday"
t4 = "I don't love you anymore..!"
t5="FUCK OFF!!!"
t6="FUCK OFF YOU BASTARD"
t7="When I felt that my love was returned"
texts = [t1, t2, t3, t4, t5, t6, t7]
# for text in texts: 
#     features = create_feature(text, nrange=(1, 4))
#     features = vectorizer.transform(features)
#     prediction = clf.predict(features)[0]
#     print(text,emoji_dict[prediction])

for text in texts:
    features = create_feature(text, nrange=(1, 4))
    features = vectorizer.transform(features)
    prediction = clf.predict(features)
    
    # Check if the prediction is empty (no emotion)
    if prediction:
        print(text,"-->",emoji_dict[prediction[0]])
    else:
        print(text, "--> Neutral or No emotion predicted")
