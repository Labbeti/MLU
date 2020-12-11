
from mlu.metrics.meteor import METEOR, get_nb_chunks
from mlu.utils.sentence import add_to_vocabulary, sentence_to_tensor, list_sentence_to_list_tensor, build_conversions_tables, list_tensor_to_tensor
from unittest import TestCase, main


class TestChunks(TestCase):
	def test_chunks(self):
		candidates = [
			"on the mat sat the cat",
			"the cat sat on the mat",
			"the cat was sat on the mat",
		]
		references = [
			"the cat sat on the mat",
		]
		candidates = [can.lower().split(" ") for can in candidates]
		references = [ref.lower().split(" ") for ref in references]

		vocabulary = add_to_vocabulary(candidates)
		vocabulary = add_to_vocabulary(references, vocabulary)
		word_to_cls_table, cls_to_word_table = build_conversions_tables(vocabulary)

		candidates = list_sentence_to_list_tensor(candidates, word_to_cls_table)
		references = list_sentence_to_list_tensor(references, word_to_cls_table)

		expected_list = [6, 1, 2]
		for candidate, expected in zip(candidates, expected_list):
			nb_chunks = get_nb_chunks(candidate, references)[0].item()
			self.assertEqual(nb_chunks, expected)


class TestMETEOR(TestCase):
	def test_meteor_1(self):
		candidate = "non matching hypothesis"
		references = [
			'this is a cat',
		]

		meteor = METEOR()
		score = meteor(candidate, references).item()
		expected = 0.0

		self.assertEqual(score, expected)

	def test_meteor_2(self):
		# Example from https://www.kite.com/python/docs/nltk.meteor
		candidate = "It is a guide to action which ensures that the military always obeys the commands of the party"
		references = [
			'It is a guide to action that ensures that the military will forever heed Party commands',
			'It is the guiding principle which guarantees the military forces always being under the command of the Party',
			'It is the practical guide for the army always to heed the directions of the party',
		]

		meteor = METEOR()
		score = meteor(candidate, references).item()
		expected = 0.7398

		self.assertAlmostEqual(score, expected, delta=10e-4)

	def test_meteor_3(self):
		# example from
		# https://stackoverflow.com/questions/63778133/how-can-i-implement-meteor-score-when-evaluating-a-model-when-using-the-meteor-s
		references = ["this is an apple", "that is an apple"]
		candidates = ["an apple on this tree", "a red color fruit"]
		expected_list = [0.6233062330623306, 0.0]

		meteor = METEOR()

		for candidate, expected in zip(candidates, expected_list):
			score = meteor(candidate, references).item()
			self.assertAlmostEqual(score, expected)


if __name__ == "__main__":
	main(failfast=True)
