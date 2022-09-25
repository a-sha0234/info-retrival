#imports
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import csv
from elasticsearch import Elasticsearch
import json
from sklearn.metrics.pairwise import cosine_similarity



def uploadData():#function uploads data to elasticsearch

    try:
        es = Elasticsearch([{'host': 'localhost', 'port': 9200}])#connect to elasticsearch
        f = open('test.json')

        data = json.load(f)#loads data
        for i in data:
            res = es.index(index='ch123', body=i)


    except:
        print("error uploading data")


uploadData()






def retirveMovieReviews():#retirves reviews from elasticsearch and puts in csvfile
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    result = es.search(#query to match all
        index="ch123",
        body={
            "query": {
                "match_all": {}
            }
        },
    size=1000)
    with open('data.csv', 'w', encoding="utf-8") as f: #generates file with the movies reviews
        header_present = False
        for doc in result['hits']['hits']:
            my_dict = doc['_source']
            if not header_present:
                w = csv.DictWriter(f, my_dict.keys())
                w.writeheader()
                header_present = True

            w.writerow(my_dict)#writes to a new csv file


retirveMovieReviews()


def stem_sentences(sentence): #function to stem columns in csv file
    porter_stemmer = PorterStemmer()
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)




def tfid(field, query):#works out tfidf and handles query from user



    filename = 'data.csv'
    df = pd.read_csv(filename)
    #convert numbers to string
    df['Release Year'] = df['Release Year'].astype(str)

    #apply stemming function to csv in dataframe
    df['Title'] = df['Title'].apply(stem_sentences)
    df['Director'] = df['Director'].apply(stem_sentences)
    df['Origin/Ethnicity'] = df['Origin/Ethnicity'].apply(stem_sentences)
    df['Genre'] = df['Genre'].apply(stem_sentences)
    df['Plot'] = df['Plot'].apply(stem_sentences)
    df['Wiki Page'] = df['Wiki Page'].apply(stem_sentences)
                #perform tf-idf


    vectorizer = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word',stop_words='english',
                              ngram_range=(1, 1),
                             norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,lowercase=True, tokenizer=None  )#preprocess

    X = vectorizer.fit_transform(df[field])


    query_vec = vectorizer.transform([query])
    results = cosine_similarity(X, query_vec).reshape((-1,))

    for i in results.argsort()[-10:][::-1]:#list top 10 results
        print(df.iloc[i, 0], "--", df.iloc[i, 1])

    mylist = []
    for i in results:
        mylist.append(i)

    mylist.sort(reverse=True)
    print(mylist)













def mainf():#function to get query from user
    print("Please enter a number from below: ")
    print("1 = Release Year, 2 = Title, 3 = Origin/Ethnicity, 4 = Director, 5 = Cast, 6 = Genre, 7 =  Wiki Page, 8 = Plot  ")
    userInput = input("Enter a key:")
    if userInput == "1":
        userinput1 = input("Please enter a release year name : ")
        tfid("Release Year", userinput1)#put query information into function


    elif userInput == "2":
        userinput1 = input("Please enter a title name : ")
        tfid("Title", userinput1)

    elif userInput == "3":
        userinput1 = input("Please enter an Origin/Ethnicity  name : ")
        tfid("Origin/Ethnicity", userinput1)

    elif userInput == "4":
        userinput1 = input("Please enter a Directors name : ")
        tfid("Director", userinput1)

    elif userInput == "5":
        userinput1 = input("Please enter a cast's name : ")
        tfid("Cast", userinput1)

    elif userInput == "6":
        userinput1 = input("Please enter a Genre : ")
        tfid("Genre", userinput1)


    elif userInput == "7":
        userinput1 = input("Please enter a Wiki Page : ")
        tfid("Wiki Page", userinput1)

    elif userInput == "8":
        userinput1 = input("Please enter a keyword in the plot : ")
        tfid("Plot", userinput1)

    else:
        print("Please enter correct key")




mainf()

























































