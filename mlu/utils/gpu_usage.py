
import os
import subprocess

from xml.dom import minidom
from typing import Any, Dict, Tuple


class GPUUsage:
	def __init__(self):
		super().__init__()
		self._info_pids = {}
		self._used_memory = (0, "MiB")
		self._total_memory = (0, "MiB")
		self._command = ["nvidia-smi", "-x", "-q"]

	def update(self):
		output = subprocess.check_output(self._command)
		output = output.decode()
		tree = minidom.parseString(output)

		# print(tree.toxml())
		elt_nvidia_smi_log = tree.childNodes[1]
		elt_gpu_lst = elt_nvidia_smi_log.getElementsByTagName("gpu")
		assert len(elt_gpu_lst) == 1, "Multiple GPU is not supported"
		elt_gpu = elt_gpu_lst[0]

		def get_text(elt) -> str:
			node_str = elt.toxml()
			start = node_str.find(">") + 1
			end = -start - 1
			return node_str[start:end]

		# gpu/fb_memory_usage
		elt_fb_memory_usage = elt_gpu.getElementsByTagName("fb_memory_usage")[0]
		elt_used = elt_fb_memory_usage.getElementsByTagName("used")[0]
		elt_total = elt_fb_memory_usage.getElementsByTagName("total")[0]

		self._used_memory = get_text(elt_used).split(" ")
		self._total_memory = get_text(elt_total).split(" ")

		info_pids = {}

		elt_processes_lst = elt_gpu.getElementsByTagName("processes")
		for elt_processes in elt_processes_lst:
			elt_process_info_lst = elt_processes.getElementsByTagName("process_info")
			for elt_process_info in elt_process_info_lst:

				elt_process_name_lst = elt_process_info.getElementsByTagName("process_name")
				elt_used_memory_lst = elt_process_info.getElementsByTagName("used_memory")
				elt_pid_lst = elt_process_info.getElementsByTagName("pid")

				assert len(elt_process_name_lst) == len(elt_used_memory_lst) == len(elt_pid_lst) == 1
				elt_process_name = elt_process_name_lst[0]
				elt_used_memory = elt_used_memory_lst[0]
				elt_pid = elt_pid_lst[0]

				# Get process pid
				pid = get_text(elt_pid)
				pid = int(pid)

				# Get process name
				process_name = get_text(elt_process_name)

				# Get process used memory
				# Example : '<used_memory>16 MiB</used_memory>'
				used_memory = get_text(elt_used_memory).split(" ")

				info_pids[pid] = {"process_name": process_name, "used_memory": used_memory}

		self._info_pids = info_pids

	def get_pid_info(self, pid: int = os.getpid()) -> Dict[str, Any]:
		if pid in self._info_pids.keys():
			return self._info_pids[pid]
		else:
			return {}

	def get_pid_used_memory(self, pid: int = os.getpid()) -> int:
		""" Returns the PID GPU memory used in MiB. """
		if pid not in self._info_pids.keys():
			return 0

		info = self._info_pids[pid]
		memory = info["used_memory"]
		return self._to_mib(memory)

	def get_used_memory(self) -> int:
		""" Returns the GPU memory used in MiB. """
		return self._to_mib(self._used_memory)

	def get_total_memory(self) -> int:
		""" Returns the total GPU memory in MiB. """
		return self._to_mib(self._total_memory)

	def _to_mib(self, memory: Tuple[int, str]) -> int:
		value, unit = memory
		if unit == "KiB":
			value = value // 2 ** 10
		elif unit == "MiB":
			pass
		elif unit == "GiB":
			value = value * 2 ** 10
		else:
			raise ValueError(f"Unsupported unit '{unit}'.")
		return value


def test():
	import torch
	from torch.nn import BCELoss

	usage = GPUUsage()
	usage.update()
	print(usage.get_pid_info())
	print(usage.get_pid_used_memory())
	print(usage.get_used_memory())
	print(usage.get_total_memory())

	bce = BCELoss()
	with torch.no_grad():
		for _ in range(10000):
			z = torch.zeros(64, 500, device=torch.device("cuda"))
			u = torch.ones(64, 500, device=z.device)
			loss = bce(z, u)
			usage.update()
			print(usage.get_used_memory())


if __name__ == "__main__":
	test()
