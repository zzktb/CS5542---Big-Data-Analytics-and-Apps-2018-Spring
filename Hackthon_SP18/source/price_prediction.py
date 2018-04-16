

reload(sys)
sys.setdefaultencoding('utf8')

def createDict(price_file):
    price = pd.read_csv(price_file)
    d = {}
    i = 0
    for typesubcategory in price.TypeSubcategory:
        d[typesubcategory.lower()] = price.Prices[i]
        i = i + 1
    return d

def search(d, searchFor):
    df = pd.DataFrame({'type': d.keys(), 'price': d.values()})
    return df[df['type'].str.contains(searchFor)]

def getActualValue(searchFor):
    try:
        d = createDict(price_file='price_dataset.csv')
        a = search(d,searchFor.lower())
        return np.array([a.values[0][0]])
    except:
        return 1

def train_price_model(data_file):
    train = pd.read_csv(data_file)
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 10))
    X_train = vectorizer.fit_transform(np.array(train.Type))
    model = MultinomialNB().fit(X_train, np.array(train.Prices))
    return model, vectorizer

def get_score_of_price(text):
    actual = getActualValue(text)
    predicted = float(get_price(text)[0])
    #accuracy_score = metrics.accuracy_score(predicted,actual)
    accuracy_score = predicted/actual
    return accuracy_score

def get_price(text):
    data_file = 'price_dataset.csv'
    model, vectorizer = train_price_model(data_file)
    test = vectorizer.transform([text])
    return model.predict(test)

if __name__ == '__main__':
    type = "Rava" #Pizza, Stawberry, Burger, Fries, Biriyani, Dosa, Egg, etc...
    d = createDict(price_file='price_dataset.csv')
    print type,"has %s prices." % get_price(type.lower())
    print "Accuracy Score: %f" % get_score_of_price(type.lower())
    print "Other healthy options: \n %s" % (search(d, type.lower()))
    #print getActualValue(type)


