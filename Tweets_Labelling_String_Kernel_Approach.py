from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
from wordcloud import WordCloud
import re
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import os
import math
from collections import Counter
from rake_nltk import Rake
import csv


class StdOutListener(StreamListener):

    def on_data(self, data):
        try:
            file = open("Tweets.txt", "w")
            file.write(data)
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
            return True

    def on_error(self, status):
        print(status)
        return True

  #CREDENTIALS FROM TWITTER'S DEVELOPER'S ACCOUNT
access_token = "870981368712318976-y3m7s6HcrIZObKgNwAiYoH3cn0jcg1o"
access_token_secret = "Swhx1wCgFxFHhLid2gra0KEJiYjIIZCQcvYddHaXqQxmi"
consumer_key = "deU6Ayy1kfHtyW3q3JxbBG8gw"
consumer_secret = "O2083tdzCyTz0bFSE1oFqsitO8BnsrNFKh4WpdRqnTiJg5Og30"


m=os.path.exists("Tweets.txt") #If the file is already scraped, no need to fetch data again. Besides, Twitter stream data fetch live tweets.
if(m== 1):
    print("requirement satisfied")

else:

    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    stream.filter(track=['Pulwama Attack','Pulwama','Pulwama Pakistan', 'Pulwama India', 'Pulwama Kashmir'])
#Tweets with these keywords were filtered.

##################################################FETCHING NEWS ITEMS########################################
#The approach is to fetch news items from BBC (neutral News items), Geo News, Dunya news

#########################################BBC#############################################

NeutralNews = list()
Neutral_News=os.path.exists("BBC_Headlines.txt") #If the file is already scraped, no need to fetch data again. Besides, Twitter stream data fetch live tweets.
if(Neutral_News== 1):
    print("requirement satisfied")
    with open("BBC_Headlines.txt") as f:
        NeutralNews = [line.rstrip('\n') for line in open("BBC_Headlines.txt")]
        print(NeutralNews)
else:
    print("BBC News")
    NeutralNews=[]
    source_code = requests.get('https://www.bbc.com/news/topics/cgmkz7g3xn0t/pulwama-attack')
    html_code = source_code.text
    soup = BeautifulSoup(html_code, 'lxml')

    headings = soup.find_all('h3')
    for i in headings:
        NeutralNews.append(i.text)
        print(i.text)
        file = open("BBC_Headlines.txt", "w")
        for i in range(1,40):
            file.write(NeutralNews[i])
            file.write("\n")



###########################################INDIAN NEWS################################
Indiannews=[]
Indian_News=os.path.exists("Indian_headlines.txt") #If the file is already scraped, no need to fetch data again. Besides, Twitter stream data fetch live tweets.
if(Indian_News== 1):
    print("requirement satisfied")
    with open("Indian_headlines.txt") as f:
        Indiannews = [line.rstrip('\n') for line in open("Indian_headlines.txt")]
        print(Indiannews)
else:
    file = open("Indian_headlines.txt", "w")
    s_c = requests.get('https://www.thehindu.com/topic/pulwama-attack-2019/')
    html_c = s_c.text
    soup_ = BeautifulSoup(html_c, 'lxml')

    headlines = soup_.find_all('h3')
    Indiannews=[]
    for i in headlines:
        Indiannews.append(i.text)
        print(i.text)
        print(len(Indiannews))
        for i in range(1,40):
            file.write(Indiannews[i])
            file.write("\n")



######################################################PAKISTANI NEWS ABOUT PULWAMA####################
Pak_news=[]
Pakistani_News=os.path.exists("Pakistan_News.txt") #If the file is already scraped, no need to fetch data again. Besides, Twitter stream data fetch live tweets.
if(Pakistani_News== 1):
    print("requirement satisfied")
    with open("Pakistan_News.txt") as f:
        Pak_news = [line.rstrip('\n') for line in open("Pakistan_News.txt")]
        print(Pak_news)
else:
    f=open('Pakistan_News.txt', 'w+')
    s_c1 = requests.get('https://www.google.com/search?ei=GtqCXNPrC-TmgwedoJWwAg&q=the+news+tribune+about+pulwama&oq=the+news+tribune+about+pulwama&gs_l=psy-ab.3..33i21.761.5152..5341...2.0..1.632.5442.2-3j4j1j5......0....1..gws-wiz.......0i71j0j0i22i30j33i22i29i30j0i22i10i30j33i160.UkOKG81djdk')
    html_c = s_c1.text
    soup_ = BeautifulSoup(html_c, 'lxml')
    headlines = soup_.find_all('h3')
    for i in headlines:
        Pak_news.append(i.text)
        print(Pak_news)
    s_con = requests.get('https://www.google.com/search?q=geo+about+pulwama&tbm=nws&source=univ&tbo=u&sa=X&ved=2ahUKEwjF2N_Pu_PgAhVJSxoKHda8Dn8Qt8YBKAF6BAgAEA4&biw=1093&bih=461')
    html_con = s_con.text
    s_ = BeautifulSoup(html_con, 'lxml')
    headlines_ = s_.find_all('h3')
    for i in headlines_:
        Pak_news.append(i.text)
        sc_con = requests.get('https://www.google.com/search?biw=1093&bih=461&tbm=nws&ei=ztuCXJTQA4-ua97QrPgE&q=ary+news+about+pulwama+attack&oq=ary+news+about+pulwama+attack&gs_l=psy-ab.3..33i10k1.8017.9224.0.9579.7.7.0.0.0.0.288.288.2-1.1.0....0...1c.1.64.psy-ab..6.1.287....0.fdRDgb1TCQs')
        html_cont = sc_con.text
        sp_ = BeautifulSoup(html_cont, 'lxml')
        headlines_Ary = sp_.find_all('h3')
        for i in headlines_Ary:
            Pak_news.append(i.text)
            sc = requests.get('https://www.google.com/search?biw=1093&bih=461&tbm=nws&ei=5NyCXJPAFKeDjLsP8pSqgA0&q=Samaa+news+pulwama+attack&oq=Samaa+news+pulwama+attack&gs_l=psy-ab.3...22030.36954.0.37379.10.9.0.0.0.0.387.1034.3-3.3.0....0...1c.1.64.psy-ab..7.0.0....0.iciy_VtyPxI')
            htm = sc.text
            sp = BeautifulSoup(htm, 'lxml')
            headlines_Samaa = sp.find_all('h3')
            for i in headlines_Samaa:
                Pak_news.append(i.text)
                print("Pak news",Pak_news)
            for i in range(0,40):
                f.write(str(Pak_news[i]))
                f.write("\n")

##########################################LOADING TWEETS#############################################
tweets_data_path = 'Tweets.txt'

t = []
tweets = []
tweets_data = []
tweet_text=[]
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweet_text.append(tweet['text'])
        tweets_data.append(tweet)
        t.append(tweet['text'])
        tags = []
        for ent in tweet['entities']['hashtags']:
                tags.append(ent['text'])
                tweet = {
                    "id": tweet['id'],
                    "text": tweet['text'],
                    'tags': tags
                }

        tweets.append(tweet)
    except:
        continue

print("length")
print(len(tweets))
###########################################CLEANING TWEETS############################################

def clean_tweet(tweet): #for discarding links & special characters.
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+|ur'[\W_])", " ", tweet).split())

def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i)<128)



tweets = pd.DataFrame()


############################################################# TWEET EXPLORATION (VISUALIZATION) ########################################3
stop_words=["is","rt" ,"@","am","are","we", "https", "should","have", "co", "the", "this","our", "but", "from","of", "when", "in", "on", "it", "not", "to", "who", "has","with","as","by","an","that","for","how","you","where", "what","when","why", "RT","after","and","did","at","amp","he","there","said","nhttps","was","out","be","their","will","now","even","if"]
wordcloud = WordCloud(width=800, height=800,stopwords=stop_words).generate(str(set(tweet_text)))

# Let's plot the WordCloud image of the tweets to find out the most frequent words in tweets.
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")

plt.margins(x=0, y=0)
plt.show()


######################################################TIME TO LABEL THE TWEETS########################################

def get_cosine(v1, v2): #Cosine similarity
    intersection = set(v1.keys()) & set(v2.keys())
    numerator = sum([v1[x] * v2[x] for x in intersection])

    sum1 = sum([v1[x] ** 2 for x in v1.keys()])
    sum2 = sum([v2[x] ** 2 for x in v2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

WORD = re.compile(r'\w+')

def text_to_vector(text): #considering term frequency as vector
    words = WORD.findall(text)
    return Counter(words)

FAIL = -1

def aho_corasick(string, keywords): #Took idea from the internet for matching keywords with strings.
    transitions = {}
    outputs = {}
    fails = {}

    new_state = 0

    for keyword in keywords:
        state = 0

        for j, char in enumerate(keyword):
            res = transitions.get((state, char), FAIL)
            if res == FAIL:
                break
            state = res

        for char in keyword[j:]:
            new_state += 1
            transitions[(state, char)] = new_state
            state = new_state

        outputs[state] = [keyword]

    queue = []
    for (from_state, char), to_state in transitions.items():
        if from_state == 0 and to_state != 0:
            queue.append(to_state)
            fails[to_state] = 0

    while queue:
        r = queue.pop(0)
        for (from_state, char), to_state in transitions.items():
            if from_state == r:
                queue.append(to_state)
                state = fails[from_state]

                while True:
                    res = transitions.get((state, char), state and FAIL)
                    if res != FAIL:
                        break
                    state = fails[state]

                failure = transitions.get((state, char), state and FAIL)
                fails[to_state] = failure
                outputs.setdefault(to_state, []).extend(
                    outputs.get(failure, []))

    state = 0
    results = []
    for i, char in enumerate(string):
        while True:
            res = transitions.get((state, char), state and FAIL)
            if res != FAIL:
                state = res
                break
            state = fails[state]

        for match in outputs.get(state, ()):
            pos = i - len(match) + 1
            results.append((pos, match))

    return results


def isListEmpty(inList): #for checking if the results attained from the above algorithm are empty or not.
    if isinstance(inList, list):
        return all( map(isListEmpty, inList) )
    return False

#####################################################MAIN LABELLING ##########################################

print(len(NeutralNews))



print(len(tweet_text))

with open("dataset2.csv",'a',newline='', encoding='utf-8') as f:
    thewriter=csv.writer(f)
    flag = 0
    t = []
    relevant_news = []
    irrelevant_news = []
    i = 0
    x = 0
    for i in range(0,6235):
            if (i == 6235):
                break
            else:
                counter = -1
                flag = 0
                print("Tweet: ", tweet_text[i])
            for z in range(1,9):
                for y in range(0,len(Indiannews)):
                    for x in range(0,len(Pak_news)):
                        counter += 1
                        r = Rake()
                        r.extract_keywords_from_text(tweet_text[i])
                        keyword = r.get_ranked_phrases()  # keywords extraction from each tweet
                        t.append(tweet_text[i])
                        res = aho_corasick(Pak_news[x], keyword)
                        if(len(keyword)!=0):
                            l = len(res) / len(keyword)
                            res2 = aho_corasick(Indiannews[y], keyword)
                            thresholdIndianNews = len(res2) / len(keyword)
                            res3 = aho_corasick(NeutralNews[z], keyword)
                            thresholdNeutralNews = len(res3) / len(keyword)
                            alpha1 = l * 100  # a % of how much keywords are matched with the headlines
                            alpha2 = thresholdIndianNews * 100
                            alpha3 = thresholdNeutralNews * 100

                            if alpha3 > 50 and alpha1 > 25 and alpha2 > 25:
                                vector1 = text_to_vector(tweet_text[i])
                                vector2 = text_to_vector(Pak_news[x])
                                cosine = get_cosine(vector1, vector2)  # cosine similarity between news and tweet
                                print("Not a Rumor")
                                flag = 1
                            if flag == 1:
                                print("Tweet: ", tweet_text[i])
                                print("Not a Rumor")
                                tweet_text[i] = remove_non_ascii(str(tweet_text[i]))
                                tweet_text[i] = clean_tweet(str(tweet_text[i]))
                                thewriter.writerow([str(tweet_text[i]), "Not a Rumor"])
                                i += 1
                                break
                            else:
                                if counter == len(Pak_news) - 1:
                                    i += 1
                                    print("Rumor")
                                    thewriter.writerow([str(tweet_text[i]), "Rumor"])




