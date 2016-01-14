# coding: utf-8

"""
Batch reduction of reflectometry data based on a spreadsheet
"""

import collections
import numpy as np
import os.path
import pandas as pd
import re
import sys
import warnings
import IPython.display

import refnx.reduce


ReductionEntryTuple = collections.namedtuple('ReductionEntry',
    [
      'row',
      'ds',
      'name',
      'fname',
      'entry',
    ])

                                        
class ReductionEntry(ReductionEntryTuple):
                         
    def rescale(self, scale_factor, write=True):
        self.ds.scale(scale_factor)
        if write:
            with open(self.fname, 'w') as w:
                self.ds.save_xml(w)
    
    
class ReductionCache(list):
    """
    Cache for the reduced data to enable look-up by name, run number or row.

    Entries in the cache are ReductionEntry objects.

    Examples
    --------

    >>> reduced = batch_reduce('reduction.xls', pth, rebin_percent, cache)
    >>> cache.summary()

    Find the filename of a run in the cache by sample name

    >>> cache.name('W1234').fname

    Find a run in the cache by run number and plot it

    >>> data = cache.run(24623)
    >>> plt.plot(data[0], data[1])

    Search for data by run name (starting substring or regular expression)

    >>> cache.name_startswith('W')
    >>> plot_data_sets(cache.name_search('^W')
    """

    def __init__(self):
        """
        Create a new reduction cache
        """
        self.name_cache = {}
        self.run_cache = {}
        self.row_cache = {}

    def add(self, row, ds, name, fname, entry, update=True):
        """
        Add (or update) a data set in the reduction cache

        Parameters
        ----------
        row : int
            row number in the batch reduction spreadsheet
        ds : ReflectDataset
            reduced data from the reduce_stitch (or similar) function
        name : str
            name of the same, as specified in the spreadsheet
        fname : str
            filename that was used to save the data during reduction
        entry : pandas.Series
            the row of the spreadsheet (as a row from the pandas table)
            from which the reduction was controlled
        update : boolean (optional)
            if the spreadsheet row (as identified by the `row` argument)
            has already been added to the cache, then replace the existing
            entry in the cache.
        """
        data = ReductionEntry(row, ds, name, fname, entry)

        # if the data is already in the cache, update it
        if update and row in self.row_cache:
            idx = self.row_cache[row]
            self[idx] = data
        else:
            idx = len(self)
            self.append(data)
            self.name_cache[name] = idx

        self.row_cache[row] = idx

        # also cache the runs that made up the reduction, which may be
        # several since they can be stitched together
        runs = run_list(entry)
        for run in runs:
            self.run_cache[run] = idx
        return data

    def run(self, run_number):
        """ select a single cached data set by run number

        Parameters
        ----------
        run_number : int
            run number to find
        """
        return self[self.run_cache[run_number]]

    def runs(self, run_numbers):
        """ select several cached data sets by run number

        Parameters
        ----------
        run_numbers : iterable
            run numbers to find
        """
        return [self.run[r] for r in runs]

    def row(self, row_number):
        """ select a single cached data set by spreadsheet row number

        Parameters
        ----------
        row_number : int
            row numbers to find
        """
        return self[self.row_cache[row_number]]

    def rows(self, row_numbers):
        """ select several cached data sets by spreadsheet row number

        Parameters
        ----------
        row_numbers : iterable
            row numbers to find
        """
        return [entry for entry in self if entry.row in row_numbers]

    def name(self, name):
        """ select a single cached data set by sample name

        Parameters
        ----------
        name : str
            sample name to find
        """
        return self[self.name_cache[name]]

    def name_startswith(self, name):
        """ select cached data sets by start of sample name

        Parameters
        ----------
        name : str
            fragment that must be at the start of the sample name
        """
        matches = [entry for entry in self if entry.name.startswith(name)]
        return matches

    def name_search(self, search):
        """ select cached data sets by a regular expression on sample name

        The search pattern is a `regular expression`_ that is matched with

        Parameters
        ----------
        search : str or SRE_Pattern
            string or compiled regular expression (from `re.compile(pattern)`)
            that will be checked against the sample name.

        Examples
        --------
        Select all data where the name starts with `Sample 1`:
        >>> cache.name_search("^Sample 1")

        Select all data where the name contains `pH 4.0`:
        >>> cache.name_search(r"pH 4\.0")

        .. _`regular expression`:
           https://docs.python.org/3/howto/regex.html
        """
        if isinstance(search, str):
            name_re = re.compile(search)
        else:
            name_re = search
        matches = [entry for entry in self if name_re.search(entry.name)]
        return matches

    def summary(self):
        """ pretty print a list of all data sets currently in the cache

        The pandas pretty printer is used with the IPython HTML display.
        """
        df = pd.DataFrame(columns=self[0].entry.axes)
        for i, entry in enumerate(self):
            df.loc[i] = entry.entry
        IPython.display.display(
          IPython.display.HTML("<b>Summary of reduced data</b>"))
        IPython.display.display(df)

        
class BatchReducer():
    """
    Batch reduction of reflectometry data based on spreadsheet metadata.

    Example
    -------

        >>> from batchreduction import BatchReducer
        >>> pth = r'V:\data\current'
        >>> b = BatchReducer('reduction.xls', pth, 2.0)
        >>> b.reduce()
        >>> b.summary()

    The spreadsheet must have columns:

        reduce name scale refl1 refl2 refl3 dir1 dir2 dir3

    Only rows where the value of the `reduce` column is 1 will be processed.
    """
    
    def __init__(self, filename, pth, rebin_percent):
        """
       Create a batch reducer using metadata from a spreadsheet

        Parameters
        ----------
        filename : str
            The filename of the spreadsheet to be used. Must be readable by
            `pandas.read_excel` (`.xls` and `.xlsx` files).
        pth : str
            Filesystem path for the raw data files
        rebin_percent : float
            percentage rebinning to be applied in the reduction (e.g. 2.0)
        """
        self.cache = ReductionCache()
        self.filename = filename
        self.pth = pth
        self.rebin_percent = rebin_percent

    @staticmethod
    def _reduce_row(entry, pth, rebin_percent):
        """ Process a single row using reduce_stitch

        Parameters
        ----------
        entry : pandas.Series
            Spreadsheet row for this data set
        pth : str
            Filesystem path for the raw data files
        rebin_percent : float
            percentage rebinning to be applied in the reduction (e.g. 2.0)
        """
        # Identify the runs to be used for reduction
        runs = run_list(entry, 'refl')
        directs = run_list(entry, 'directs')

        fmt = "Reducing %s [%s]/[%s]"
        print(fmt % (entry['name'],
                     ", ".join('%d' % r for r in runs),
                     ", ".join('%d' % r for r in directs)))
        sys.stdout.flush()   # keep progress updated

        if not runs:
            warnings.warn("Row %d (%s) has no reflection runs" %
                          (entry['source'], entry['name']))
            return None, None
        if not directs:
            warnings.warn("Row %d (%s) has no direct beam runs" %
                          (entry['source'], entry['name']))
            return None, None

        if len(runs) != len(directs):
            warnings.warn(
              "Row %d (%s) has differing numbers of direct & refln runs" %
              (entry['source'], entry['name']))
            return None, None

        ds, fname = refnx.reduce.reduce_stitch(runs, directs,
                                               data_folder=pth,
                                               rebin_percent=rebin_percent)
        return ds, fname


    def reduce(self, show=True):
        """
        Batch reduce data based on metadata from a spreadsheet

        Parameters
        ----------
        show : bool (optional, default=True)
            display a summary table of the rows that were reduced
        """
        cols='A:I'
        all_runs = pd.read_excel(self.filename, parse_cols=cols)

        # Add the row number in the spreadsheet as an extra column
        # row numbers for the runs will start at 2 not 0
        all_runs.insert(0, 'source', all_runs.index + 2)

        # add in some extra columns to indicate successful reduction
        all_runs['reduced'] = np.zeros(len(all_runs))
        all_runs['filename'] = np.zeros(len(all_runs))
        mask = all_runs.reduce == 1
        rows = all_runs[mask].index

        # iterate through the rows that were marked for reduction
        for idx in rows:
            # ensure that the name is a string (will be NaN if blank in sheet)
            name = str(all_runs.loc[idx, 'name'])

            try:
                ds, fname = self._reduce_row(all_runs.loc[idx],
                                             self.pth, self.rebin_percent)
            except OSError as e:
                # data file not found (normally)
                reduction_ok = str(e)
                warnings.warn("Run %s: %s" % (name, str(e)))
                ds = None
                fname = None
            else:
                reduction_ok = (ds is not None)
                if reduction_ok:
                    # store this away to make plotting easier later
                    ds.name = name
                   
            # record outcomes of reduction in the table
            all_runs.loc[idx, 'filename'] = fname
            all_runs.loc[idx, 'reduced'] = reduction_ok

            cached = self.cache.add(all_runs.loc[idx, 'source'],
                                    ds, name, fname, all_runs.loc[idx])
            if reduction_ok:
                scale = all_runs.loc[idx, 'scale']
                if not np.isnan(scale) and scale != 1:
                    print("Applying scale factor %f" % scale)
                    sys.stdout.flush()   # keep progress updated
                    cached.rescale(scale)

        if show:
            IPython.display.display(all_runs[mask])

        return self.cache


def run_list(entry, mode='refl'):
    """
    Generates a list of run numbers from a reduction spreadsheet entry

    Parameters
    ----------
    entry : pandas.Series
        A row from the reduction spreadsheet expressed
    mode : 'refl' or 'directs'
        Fetch either the run numbers from the reflectometry experiment
        or from the direct beams.
    """
    if mode not in ('refl', 'directs'):
        # FIXME: crap API
        raise ValueError("Unknown mode %s" % mode)

    if mode == 'refl':
        listed = [entry['refl1'], entry['refl2'], entry['refl3']]
    else:
        listed = [entry['dir1'], entry['dir2'], entry['dir3']]

    valid = []
    for item in listed:
        if isinstance(item, str) and ',' in item:
            runs = [int(i) for i in item.split(',')]
        else:
            runs = [item]
        for run in runs:
            try:
                if not np.isnan(run):
                    valid.append(run)
            except TypeError:
                raise ValueError(
                  "Value '%s' could not be interpreted as a run number" % run)

    #valid = [int(r) for r in l if not np.isnan(r)]
    return [int(v) for v in valid]

