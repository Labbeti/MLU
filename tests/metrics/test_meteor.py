
from unittest import TestCase, main
from mlu.metrics.text.meteor import Meteor


class TestMeteor(TestCase):
	def test_meteor_1(self):
		candidate = 'non matching hypothesis'
		references = [
			'this is a cat',
		]

		meteor = Meteor()
		score = meteor(candidate, references).item()
		expected = 0.0

		self.assertEqual(score, expected)

	def test_meteor_2(self):
		# Example from https://www.kite.com/python/docs/nltk.meteor
		candidate = 'It is a guide to action which ensures that the military always obeys the commands of the party'
		references = [
			'It is a guide to action that ensures that the military will forever heed Party commands',
			'It is the guiding principle which guarantees the military forces always being under the command of the Party',
			'It is the practical guide for the army always to heed the directions of the party',
		]

		meteor = Meteor()
		score = meteor(candidate, references).item()
		expected = 0.7398

		self.assertAlmostEqual(score, expected, delta=10e-4)

	def test_meteor_3(self):
		# example from
		# https://stackoverflow.com/questions/63778133/how-can-i-implement-meteor-score-when-evaluating-a-model-when-using-the-meteor-s
		references = ['this is an apple', 'that is an apple']
		candidates = ['an apple on this tree', 'a red color fruit']
		expected_list = [0.6233062330623306, 0.0]

		meteor = Meteor()

		for candidate, expected in zip(candidates, expected_list):
			score = meteor(candidate, references).item()
			self.assertAlmostEqual(score, expected)


if __name__ == '__main__':
	main(failfast=True)
