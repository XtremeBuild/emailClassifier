import Main
import string
from Main import df, stopwords_set, stemmer

email_to_classify = df.text.values[10]

email_text = email_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split()
email_text = [stemmer.stem(word) for word in email_text if not word in stopwords_set]
email_text = ' '.join(email_text)

email_corpus = [email_text]

X_email = Main.vectorizer.transform(email_corpus)

print(Main.clf.predict(X_email))
