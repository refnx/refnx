#!/usr/local/bin/python
import os
import sys
import cgi, cgitb
import StringIO
from pyplatypus import reduce

cgitb.enable()

#specify where the files are to originate from
#FILEPATH = '\\\\Filer\\experiments\\platypus\\commissioning\\'
FILEPATH = './'

initial_page = """<html>
            <head>
                <title>platypus data reduction</title>
            </head>
            <body>
                <h1>Platypus Data Reduction</h1>
                <p>To reduce your files please enter the required file numbers, separated by a space</p>
                <form action='reduceData.cgi' method = "POST">
                    reflected angles <input type='text' name='reflected_angles' /><br/>
                    direct angles <input type='text' name='direct_angles' /><br/>
                    background subn<input type="checkbox" name="background_subn" value="on" /> <br/>
                    rebin percentage (0 < x < 10)<input type='text' name='rebinpercent' value = "4.0"/><br/>
                    <input type='submit', value = "Submit" />
                </form>
            </body>
        </html>
"""

formcontent = "Content-type: text/html\n"
datacontent = '\r\n'.join(["Content-type: %s;",
                            "Content-Disposition: attachment; filename=\"%s\"",
                            "Content-Title: %s",
                            "Content-Length: %i",
                            "",])

def main():
    form = cgi.FieldStorage()
    keys = form.keys()
    if 'reflected_angles' in keys and 'direct_angles' in keys:
        reflect_list = reduce.sanitize_string_input(form['reflected_angles'].value)
        direct_list = reduce.sanitize_string_input(form['direct_angles'].value)
        
        if not 'background_subn' in keys:
            background = False
        else:
            background = True
        if 'rebinpercent' in keys:
            if 0. < float(form['rebinpercent'].value) < 10.:
                rebinpercent = float(form['rebinpercent'].value)
        else:
            rebinpercent = 4.
        
        reduce_dataset = reduce.reduce_stitch_files(reflect_list,
                                                      direct_list,
                                                        rebinpercent = rebinpercent,
                                                         background = background,
                                                          basedir = FILEPATH)
        reduce_dataset.rebin(rebinpercent=rebinpercent)
        
        reduce_file = StringIO.StringIO()
        reduce_dataset.write_reflectivity_XML(reduce_file)
                    
        if reduce_file:
            #serve the file 
            length = len(reduce_file.getvalue())
            print datacontent % ('application/octet-stream', 'data.xml', 'data.xml', length)
            sys.stdout.write(reduce_file.getvalue()) # *not* print, doh !
            sys.stdout.flush()
        else:
            print formcontent
            print "SOMETHING WENT WRONG, CHECK INPUT AND TRY AGAIN"
            print initial_page
    else:
        print formcontent
        print initial_page
    
main()

