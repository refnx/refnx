import re
import os.path

class ProcessNexus(object):

	def __init__(self):
		pass
		
	def process(self):
		pass
		
	def process_event_stream(self):
		pass
		
	def catalogue(self):
		pass

    def nexus_file(self, fname):
        path, file = os.path.split(fname)

        regex = re.compile("([A-Za-z]{3})([0-9]{7}).nx.hdf")
        r = regex.search(file)
        if r is not None:
            instrument, number = r.groups()
            return instrument, int(number)
