# coding: utf-8
# # Term and Number of results to check
term = "invoice factoring"
num  = 100


# # General stuff
import string, numpy, pandas, math, re
from bs4 import BeautifulSoup
import urllib.request as req

#Punctuation Dict
translator = str.maketrans({key: None for key in string.punctuation}) 

#Zero Vector and Unit Vector in R^n
n = len(set(str.split(term.lower(), " ")))
unit = numpy.array([ 1 / math.sqrt(n)] * n)
zero = numpy.array([0] * n)

#Term Handelers
regex = r"(s|ing|ate|ize|ify|able)$"
reterm = re.sub(regex, "", term)
terms = list(set(str.split(reterm.lower(), " ")))

#urllib Headers
headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36"} 


# # Getting URL's from Google
urls = []

#Scraping links from Google. 
#Does not include AdLinks or Rich Snippets
google = "https://www.google.com/search?q=" + term.replace(" ", "%20") + "&num=" + str(num)
request = req.Request(google, None, headers) 
html = req.urlopen(request).read()
soup = BeautifulSoup(html, "lxml")
for script in soup.find_all("div", class_="srg"): # if you want Rich Snippets remove this line
    for link in script.find_all("h3", class_="r"):
        urls.append( link.find('a').get('href') )


# # TF-IDF without suffix
tf = pandas.DataFrame()  

for url in urls:
    try:        
        #Getting HTML and cleaning it
        request = req.Request(url,None,headers) 
        html    = req.urlopen(request).read()
        soup    = BeautifulSoup(html, "lxml")
        
        for script in soup(["script", "style", "link", "meta", "head"]):
            script.extract()
        
        soup   = soup.get_text(separator = " ")
        lines  = (line.strip() for line in soup.splitlines())   
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text   = (line.translate(translator) for line in chunks)
        
        #Raw term frequency counts
        counts = dict.fromkeys(terms, 0) 
        w      = 0
        for line in text:
            new_line = (line.split())
            for word in new_line:
                word = re.sub(regex,"",word.lower())
                w += 1
                if word in terms:
                    counts[word] += 1  
        
        #Term frequesncy as a percentage
        for key,value in counts.items():
            if value != 0:
                value = value / w

        #Add TFi to TF table
        tfi      = pandas.Series(counts)
        tfi.name = url
        tf       = tf.append(tfi)
        
    except Exception as inst:
        print("\n" + url + "\n" + str(type(inst)) + "\n" + str(inst.args) + "\n" + str(inst))
        urls.remove(url)
        continue

#Calculate IDF scores and TFIDF values
m     = len(tf)
idf   = numpy.log( m / tf[tf!=0].count(axis = 0) )
tfidf = pandas.DataFrame(tf.values * idf.values, columns = tf.columns, index = tf.index).values

#Vectorize each websites TFIDF scores 
#Calculate inner product between it and a unit(query) vector [values in cos(theta) terms]
#Generate score table
scores = pandas.DataFrame()
for i in range(len(tfidf)):
    vector = tfidf[i]
    
    if not numpy.array_equal(zero, vector):
        dot = numpy.dot(vector,unit) / numpy.linalg.norm(vector)  
    else:
        dot = 0 
    
    dist      = pandas.Series(dot, index = ["Distance"])
    dist.name = tf.index[i]
    scores    = scores.append(dist)


# # Rankings
#Ranking based on Google Indexing
j=1
for url in urls:
    scores.loc[url, "G Rank"] = j
    j+=1
    
#Ranking based on dist values
scores.sort_values("Distance", ascending = False, inplace = True)
j=1
for index, row in scores.iterrows():
    scores.loc[index, "Distance Rank"] = j
    j+=1

scores.to_csv('output.csv')
print('complete')

