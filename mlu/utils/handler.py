
import os
import os.path as osp

from logging import FileHandler


class FileHandlerCustom(FileHandler):
	def __init__(self, filename: str, mode='a', encoding=None, delay=False):
		dpath_parent = osp.dirname(filename)
		if dpath_parent != "" and not osp.isdir(dpath_parent):
			os.mkdir(dpath_parent)
		super().__init__(filename, mode, encoding, delay)
