import reduce
import reflectdataset
import string
import argparse
from time import gmtime, strftime


def sanitize_string_input(file_list_string):
    """
    given a string like '1 2 3 4 1000 -1 sijsiojsoij' return an integer list where the numbers are greater than 0 and less than 9999999
    it strips the string.ascii_letters and any string.punctuation, and converts all the numbers to ints.
    """
    return [int(x) for x in file_list_string.translate(None, string.punctuation).translate(None, string.ascii_letters).split() if 0 < int(x) < 9999999]
    
def reduce_stitch_files(reflect_list, direct_list, **kwds):
	"""
	kwds passed onto processnexusfile
	"""
	scalefactor = kwds.get('scalefactor', 1.)
		   
	#now reduce all the files.
	zipped = zip(reflect_list, direct_list)

	combineddataset = reflectdataset.ReflectDataset()

	for index, val in enumerate(zipped):
		reduced = reduce.Reduce(val[0], val[1], **kwds)
		combineddataset.adddataset(reduced)

	return combineddataset


if __name__ == "__main__":
	print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
	f=open('test.xml', 'w')
	a = reduce_stitch_files([708, 709, 710], [711,711,711])
	a.rebin(rebinpercent = 8)
	a.writereflectivityXML(f)
	f.close()
	print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())        
	