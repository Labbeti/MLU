
import torch

from mlu.utils.sentence import sentence_to_tensor
from mlu.utils.vocabulary import add_to_vocabulary
from unittest import TestCase, main


class TestSentence(TestCase):
	def test_text_to_tensor(self):
		sentence = ['a', 'b', 'a', 'c', 'd', 'b']
		expected = torch.as_tensor([0, 1, 0, 2, 3, 1])

		out = sentence_to_tensor(sentence)
		self.assertTrue(out.eq(expected).all())

	def test_vocabulary(self):
		sentence = ['a', 'b', 'a', 'c', 'd', 'b']

		voc = add_to_vocabulary(sentence)
		self.assertEqual({'a', 'b', 'c', 'd'}, voc)
		voc = add_to_vocabulary('e', voc)
		self.assertEqual({'a', 'b', 'c', 'd', 'e'}, voc)


if __name__ == '__main__':
	main()
