
from mlu.metrics.text.bleu import Bleu
from mlu.utils.sentence import sentence_to_tensor
from mlu.utils.vocabulary import add_to_vocabulary, build_conversions_tables
from unittest import TestCase, main


class TestBleu1(TestCase):
	def test_1(self):
		# example from the original paper https://www.aclweb.org/anthology/P02-1040.pdf
		candidate = ['the'] * 7
		references = [
			'The cat is on the mat'.lower().split(' '),
			'There is a cat on the mat'.lower().split(' '),
		]

		vocabulary = add_to_vocabulary(candidate)
		vocabulary = add_to_vocabulary(references, vocabulary)
		word_to_cls_table, cls_to_word_table = build_conversions_tables(vocabulary)

		candidate = sentence_to_tensor(candidate, word_to_cls_table)
		references = [sentence_to_tensor(reference, word_to_cls_table) for reference in references]

		bleu_metric = Bleu(1)
		expected = 2.0 / 7.0
		score = bleu_metric([candidate], [references]).item()

		self.assertAlmostEqual(score, expected)

	def test_2(self):
		candidate = 'a b c d e f'.lower().split(' ')
		references = [candidate, 'a g b e d']

		vocabulary = add_to_vocabulary(candidate)
		vocabulary = add_to_vocabulary(references, vocabulary)
		word_to_cls_table, cls_to_word_table = build_conversions_tables(vocabulary)

		candidate = sentence_to_tensor(candidate, word_to_cls_table)
		references = [sentence_to_tensor(reference, word_to_cls_table) for reference in references]

		bleu_metric = Bleu(1)
		expected = 1.0
		score = bleu_metric([candidate], [references]).item()

		self.assertAlmostEqual(score, expected)

	def test_3(self):
		candidates = [
			'It is a guide to action which ensures that the military always obeys the commands of the party'.lower().split(' '),
			# 'It is to insure the troops forever hearing the activity guidebook that party direct'.lower().split(' '),
		]

		references = [
			'It is a guide to action that ensures that the military will forever heed Party commands'.lower().split(' '),
			'It is the guiding principle which guarantees the military forces always being under the command of the Party'.lower().split(' '),
			'It is the practical guide for the army always to heed the directions of the party'.lower().split(' '),
		]
		expected_lst = [
			17 / 18,
			# 8 / 14, # no, its precision but without bp
		]

		bleu_metric = Bleu(1)

		for candidate, expected in zip(candidates, expected_lst):
			vocabulary = add_to_vocabulary(candidate)
			vocabulary = add_to_vocabulary(references, vocabulary)
			word_to_cls_table, cls_to_word_table = build_conversions_tables(vocabulary)

			candidate_tensor = sentence_to_tensor(candidate, word_to_cls_table)
			references_tensor = [sentence_to_tensor(reference, word_to_cls_table) for reference in references]

			score = bleu_metric([candidate_tensor], [references_tensor]).item()
			self.assertAlmostEqual(score, expected)


class TestBleuk(TestCase):
	def test_bleu_4(self):
		# example from the README of https://github.com/neural-dialogue-metrics/Bleu
		candidates = [
			'It is to insure the troops forever hearing the activity guidebook that party direct'.lower().split(' '),
			'It is a guide to action which ensures that the military always obeys the commands of the party'.lower().split(' '),
		]

		references = [
			'It is a guide to action that ensures that the military will forever heed Party commands'.lower().split(' '),
			'It is the guiding principle which guarantees the military forces always being under the command of the Party'.lower().split(' '),
			'It is the practical guide for the army always to heed the directions of the party'.lower().split(' '),
		]
		expected_lst = [0.1327211341271203, 0.5401725898595141]

		for candidate, expected in zip(candidates, expected_lst):
			vocabulary = add_to_vocabulary(candidate)
			vocabulary = add_to_vocabulary(references, vocabulary)
			word_to_cls_table, cls_to_word_table = build_conversions_tables(vocabulary)

			candidate_tensor = sentence_to_tensor(candidate, word_to_cls_table)
			references_tensor = [sentence_to_tensor(reference, word_to_cls_table) for reference in references]

			bleu_metric = Bleu(4, True)
			score = bleu_metric([candidate_tensor], [references_tensor]).item()

			self.assertAlmostEqual(score, expected)


if __name__ == '__main__':
	main(failfast=False)
