import pandas as pd
import string
import pprint
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

def my_BOW(document):
	preprocessed_documents = []

	for stri in document:
		preprocessed_documents.append(stri.lower().translate(string.maketrans('', ''), string.punctuation).split(' ')) #map empty to empty and in delete chars set everything to None

	frequency_list = []
	for i in preprocessed_documents:
		frequency_count = Counter(i) 
		frequency_list.append(frequency_count)
	pprint.pprint(frequency_list)

def get_freq_matrix(document):
	count_vector = CountVectorizer()
	#print(count_vector)
	count_vector.fit(document)
	#print(count_vector.get_feature_names())

	doc_array = count_vector.transform(document).toarray()
	print doc_array
	return pd.DataFrame(doc_array, columns=count_vector.get_feature_names())

def naive_bayes_diabetes():
	p_d_pos = None
	p_d = 0.01
	p_no_d = 0.99
	p_pos_d = 0.9 #sensitivity- probability of selecting right
	p_neg_no_d = 0.9 #specificity- probability of rejecting right

	p_d_pos = p_d * p_pos_d / (p_pos_d * p_d + (1 - p_neg_no_d) * p_no_d)
	print "Diabetes given positive: ",
	print p_d_pos

def naive_bayes_jill_stein():
	p_j_fi = None
	p_f_j = 0.1
	p_i_j = 0.1
	p_e_j = 0.8

	p_f_g = 0.7
	p_i_g = 0.2
	p_e_g = 0.1

	(p_j, p_g ) = (0.5, 0.5)
	p_j_fi = p_j * (p_f_j * p_i_j) / (p_j * p_f_j * p_i_j + p_g * p_f_g * p_i_g)
	print "Jill stein given these words", 
	print p_j_fi


def main():
	df = pd.read_table('smsspamcollection/SMSSpamCollection', 
						sep='\t',
						header=None,
						names = ['label', 'sms_message']
						)
	df['label'] = df.label.map({'ham': 0, 'spam': 1})

	# documents = ['Hello, how are you!',
 #             'Win money, win from home.',
 #             'Call me now.',
 #             'Hello, Call hello you tomorrow?']

    #my_BOW(documents)
	#frequency_matrix = get_freq_matrix(documents)
	#print frequency_matrix
	X_train, X_test, Y_train, Y_test = train_test_split(df['sms_message'], df['label'], random_state=1)
	#print(X_test)
	#print(Y_test)

	count_vector = CountVectorizer()
	training_data = count_vector.fit_transform(X_train)
	testing_data = count_vector.transform(X_test) # Note, no fitting here so new words here won't be considered for hashing
	# pprint.pprint(testing_data)
	# print("Training data")
	# pprint.pprint(training_data)
	# naive_bayes_diabetes()
	# naive_bayes_jill_stein()
	naive_bayes = MultinomialNB()
	naive_bayes.fit(training_data, Y_train)
	predictions = naive_bayes.predict(testing_data)
	print("Predictions scoring")
	#Precision all positives
	#f1 score is average of precision and recall
	print("Accuracy score: {}".format(accuracy_score(Y_test, predictions)))
	print("Precision score: {}".format(precision_score(Y_test, predictions)))
	print("Recall score: {}".format(recall_score(Y_test, predictions)))
	print("F1 score: {}".format(f1_score(Y_test, predictions)))












	


if __name__ == "__main__":
	main()