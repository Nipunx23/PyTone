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

file = 'C:/Users/asus/OneDrive/Desktop/VS/PYTHON/MINOR/SPEECH_EMOTION_RECOG_using_text/data.txt'
data = read_data(file)
print("Number of instances: {}".format(len(data)))


def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

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
lsvc = LinearSVC(random_state=123, max_iter=10000)
rforest = RandomForestClassifier(random_state=123)
dtree = DecisionTreeClassifier()

clifs = [svc,lsvc,rforest,dtree]
#clifs = [lsvc]

# train and test them 
print("| {:25} | {} | {} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
print("| {} | {} | {} |".format("-"*25, "-"*17, "-"*13))

max_train_acc = 0.0
max_test_acc = 0.0
best_clf = None

for clf in clifs: 
    clf_name = clf.__class__.__name__
    train_acc, test_acc = train_test(clf, X_train, X_test, y_train, y_test)
    print("| {:25} | {:17.7f} | {:13.7f} |".format(clf_name, train_acc, test_acc))
    
    # Check if this classifier has a higher test accuracy and/or training accuracy
    if test_acc > max_test_acc and train_acc > max_train_acc:
        max_train_acc = train_acc
        max_test_acc = test_acc
        best_clf = clf

print("\nHence,the optimum model is : ",best_clf,"\n")

l = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
l.sort()
label_freq = {}
for label, _ in data: 
    label_freq[label] = label_freq.get(label, 0) + 1

# print the labels and their counts in sorted order 
for l in sorted(label_freq, key=label_freq.get, reverse=True):
    print("{:10}({})  {}".format(convert_label(l, emotions), l, label_freq[l]))
emoji_dict = {"joy": "😊", "fear": "😱", "anger": "😠", "sadness": "😢", "disgust": "😒", "shame": "😳", "guilt": "😔"}

# Get the list of emotions
emotions = list(emoji_dict.keys())

# Generate all permutations of emotions
emotion_permutations = list(itertools.permutations(emotions, 2))

# Update the emoji_dict with permutations and emojis
for emotion_pair in emotion_permutations:
    emoji_dict[f"{emotion_pair[0]} {emotion_pair[1]}"] = emoji_dict[emotion_pair[0]] + emoji_dict[emotion_pair[1]]

# calling out take_input program
import os
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
program_path = os.path.join(current_dir, 'C:/Users/asus/OneDrive/Desktop/VS/PYTHON/MINOR/SPEECH_EMOTION_RECOG_using_text/input_gen_audio_text.py')
subprocess.run(['python', program_path])

# t1 = "This looks so impressive"
# t2 = "I have a fear of dogs"
# t3 = "My dog died yesterday"
# t4 = "I don't love you anymore..!"
# t5="See you later!!"
# t6="I'm not feeling good today"
# t7="When I felt that my love was returned"
# t8="Hello,I am vaibhav"
# t9="Nice to meet you"
# t10="you're not doing it correctly"
# t11 = "I received a lovely gift today"
# t12 = "I can't believe it's already Friday"
# t13 = "The weather is perfect for a picnic"
# t14 = "I'm feeling very content right now"
# t15 = "I hate getting stuck in traffic"
# t16 = "The movie was so boring"
# t17 = "I'm so excited about the upcoming vacation"
# t18 = "I'm really nervous about the presentation"
# t19 = "I won a surprise prize at the event"
# t20 = "I spilled coffee on my laptop, what a disaster"
f=open("C:/Users/asus/OneDrive/Desktop/VS/PYTHON/MINOR/input.txt")
t21=f.read()
print("\nAnalyzing for input text : ",t21,"\n")


# texts = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,t21]
texts = [t21]


emoji_dict_p=  {"joy": "😊", "fear": "😱", "anger": "😠", "sadness": "😢", "disgust": "😒", "shame": "😳", "guilt": "😔"}
print("\n",emoji_dict_p,"\n")
for text in texts:
    features = create_feature(text, nrange=(1, 4))
    features = vectorizer.transform(features)
    prediction = best_clf.predict(features)
    
    # Check if the prediction is empty (no emotion)
    if prediction:
        print(text,"-->",emoji_dict[prediction[0]])
    else:
        print(text, "--> Neutral or No emotion predicted")
