
import os
import os.path as osp
import subprocess
import tqdm

from typing import Callable, List, Optional


def get_sub_fpaths(root: str, pred: Optional[Callable[[str], bool]]) -> List[str]:
	if osp.isfile(root):
		if pred is None or pred(root):
			return [root]
		else:
			return []
	elif osp.isdir(root):
		names = os.listdir(root)
		fpaths = []
		for name in names:
			fpaths += get_sub_fpaths(osp.join(root, name), pred)
		return fpaths
	else:
		raise RuntimeError(f'Path "{root}" is not a file or directory.')


def main():
	excluded = [
		'test_all.py',
		'test_split_multilabel.py',
		'test_ads.py',
		'test_meteor.py',
	]
	fpaths = get_sub_fpaths('.', lambda p: osp.basename(p) not in excluded)

	errors = []
	for fpath in tqdm.tqdm(fpaths):
		print(f'Run file "{fpath}"...')
		try:
			code = subprocess.check_call(['python', fpath])
			print(f'Code: {code}')
			if code != 0:
				errors.append(fpath)
		except subprocess.CalledProcessError as ex:
			print(f'Exception for file "{fpath}" : ')
			print(ex)
			errors.append(fpath)

	n_errors = len(errors)
	n_total = len(fpaths)
	print('All tests finished.')
	print(f'Errors: {n_errors} / {n_total}')
	print(f'Errors files: ', str(errors))


if __name__ == '__main__':
	main()
