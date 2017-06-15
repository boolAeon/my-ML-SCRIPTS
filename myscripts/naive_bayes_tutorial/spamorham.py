import pandas as pd
import string
import pprint
from collections import Counter

def main():
	df = pd.read_table('smsspamcollection/SMSSpamCollection', 
						sep='\t',
						header=None,
						names = ['label', 'sms_message']
						)
	df['label'] = df.label.map({'ham': 0, 'spam': 1})

	documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

	preprocessed_documents = []

	for stri in documents:
		preprocessed_documents.append(stri.lower().translate(string.maketrans('', ''), string.punctuation).split(' ')) #map empty to empty and in delete chars set everything to None
	print(preprocessed_documents)

	frequency_list = []
	for i in preprocessed_documents:
		frequency_count = Counter(i) 
		frequency_list.append(frequency_count)
	pprint.pprint(frequency_list)

	


if __name__ == "__main__":
	main()