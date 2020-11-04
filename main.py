import json
import nltk
import numpy
import random
import tensorflow
import tflearn
import pickle
import speech_recognition as sr
from gtts import gTTS
import os
import webbrowser
import playsound

r= sr.Recognizer()
r1= sr.Recognizer()
from nltk.stem.lancaster import LancasterStemmer

nltk.download('punkt')
stemmer = LancasterStemmer()
with open('intents.json') as file:
    data = json.load(file)



try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []


    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("please say something")
            audio=r.listen(source)
         
        inp = r.recognize_google(audio)
        if inp.lower() == "quit":
            break
            
        if 'search' in inp:
            output=gTTS(text='what do you want to search', lang='en', slow=False)
            output.save("output.mp3")
            playsound.playsound("output.mp3")
            print("say")
            with sr.Microphone() as source:
                r1.adjust_for_ambient_noise(source)
                search=r1.listen(source)
                inp1 = r1.recognize_google(search)
                webbrowser.open('http://google.com/?#q='+ inp1)
                if inp.lower() == "quit":
                    break
                
            
        else:
            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            language='en'
            if (results[results_index] > 0.7):
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                output=gTTS(text=random.choice(responses), lang=language, slow=False)
                output.save("output.mp3")
                playsound.playsound("output.mp3")
            
            else:
                print("I'm not sure about that. Try again.")

chat()

