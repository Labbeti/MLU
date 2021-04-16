
import torch

from mlu.utils.tf_idf import TFIDF
from mlu.utils.sentence import sentence_to_tensor
from mlu.utils.vocabulary import add_to_vocabulary, build_conversions_tables
from unittest import TestCase, main


class TestTFIDF(TestCase):
	def test_tf_idf(self):
		# example from https://fr.wikipedia.org/wiki/TF-IDF
		tf_idf = TFIDF()

		word = 'qui'
		documents = [
			'Son nom est célébré par le bocage qui frémit, et par le ruisseau qui murmure, les vents l’emportent '
			'jusqu’à l’arc céleste, l’arc de grâce et de consolation que sa main tendit dans les nuages.'
				.lower()
				.replace('.', "").replace(',', "").replace('’', ' ').replace('!', "").replace(';', "").replace('  ', ' ')
				.split(' '),

			'À peine distinguait-on deux buts à l’extrémité de la carrière : des chênes ombrageaient l’un, autour de '
			'l’autre des palmiers se dessinaient dans l’éclat du soir.'
				.lower()
				.replace('.', "").replace(',', "").replace('’', ' ').replace('!', "").replace(';', "").replace('  ', ' ')
				.split(' '),

			'Ah ! le beau temps de mes travaux poétiques ! les beaux jours que j’ai passés près de toi ! Les premiers, '
			'inépuisables de joie, de paix et de liberté ; les derniers, empreints d’une mélancolie qui eut bien aussi '
			'ses charmes.'
				.lower()
				.replace('.', "").replace(',', "").replace('’', ' ').replace('!', "").replace(';', "").replace('  ', ' ')
				.split(' '),
		]

		vocabulary = add_to_vocabulary(word)
		vocabulary = add_to_vocabulary(documents, vocabulary)
		word_to_cls_table, cls_to_word_table = build_conversions_tables(vocabulary)

		word = sentence_to_tensor([word], word_to_cls_table)
		documents = [sentence_to_tensor(document, word_to_cls_table) for document in documents]

		scores = tf_idf(word, documents)

		idf = torch.scalar_tensor(3 / 2)
		expected = torch.as_tensor([
			2 / 38 * idf.log(),
			0.0,
			1 / 40 * idf.log(),
		])

		self.assertTrue(scores.eq(expected).all())


if __name__ == '__main__':
	main()
