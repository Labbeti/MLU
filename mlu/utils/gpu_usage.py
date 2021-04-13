
import os
import subprocess

from xml.dom import minidom


class GPUUsage:
	def __init__(self):
		super().__init__()
		self.command = ["nvidia-smi", "-x", "-q"]
		self.info = {}

	def update(self):
		output = subprocess.check_output(self.command)
		output = output.decode()
		tree = minidom.parseString(output)

		elt_nvidia_smi_log = tree.childNodes[1]
		elt_gpu_lst = elt_nvidia_smi_log.getElementsByTagName("gpu")
		info = {}

		def get_text(node_str: str) -> str:
			start = node_str.find(">") + 1
			end = -start - 1
			return node_str[start:end]

		for elt_gpu in elt_gpu_lst:
			# print(elt_gpu.tagName)
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
					text = elt_pid.toxml()
					pid = get_text(text)
					pid = int(pid)

					# Get process name
					text = elt_process_name.toxml()
					process_name = get_text(text)

					# Get process used memory
					# Example : '<used_memory>16 MiB</used_memory>'
					text = elt_used_memory.toxml()
					text = get_text(text)
					value, unit = text.split(" ")
					value = int(value)

					info[pid] = {"process_name": process_name, "used_memory_value": value, "used_memory_unit": unit}

		self.info = info

	def get_used_memory(self, pid: int = os.getpid()) -> int:
		""" Returns the GPU memory used in KiB. """
		if pid not in self.info.keys():
			return 0

		info = self.info[pid]
		value = info["used_memory_value"]
		unit = info["used_memory_unit"]

		if unit == "KiB":
			pass
		elif unit == "MiB":
			value *= 2 ** 10
		elif unit == "GiB":
			value *= 2 ** 20
		else:
			raise ValueError(f"Unsupported unit '{unit}' for pid '{pid}'.")

		return value


def test():
	import torch

	z = torch.zeros(256, 64, 500, dtype=torch.bool, device=torch.device("cuda"))
	u = torch.ones(64, 500, device=z.device)
	x = z * u

	usage = GPUUsage()
	usage.update()
	print(usage.info[os.getpid()])
	print(usage.get_used_memory())
	input("> ")


if __name__ == "__main__":
	test()
