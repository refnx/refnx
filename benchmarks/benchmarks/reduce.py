# Copyright (c) 2011-2026 A. R. J. Nelson, Australian Nuclear Science and Technology Organisation
# SPDX-License-Identifier: BSD-3-Clause

import zipfile
import shutil
import urllib.request
import os
import sys

from pathlib import Path

import numpy as np
from refnx.reduce.event import events
from .common import Benchmark


class Event(Benchmark):
    def setup(self):
        pth = Path(
            os.getcwd(),
            "refnx-testdata-master",
            "data",
            "reduce",
            "DAQ_2012-01-20T21-50-32",
            "DATASET_0",
        )
        try:
            self.fi = open(pth / "EOS.bin", "rb")
        except FileNotFoundError:
            raise FileNotFoundError(os.listdir(os.getcwd()))

    def teardown(self):
        self.fi.close()

    def setup_cache(self):
        url = "https://github.com/refnx/refnx-testdata/archive/master.zip"

        try:
            # grab the test data
            with (
                urllib.request.urlopen(url, timeout=5) as response,
                open("master.zip", "wb") as f,
            ):
                shutil.copyfileobj(response, f)

            # master.zip is in tmpdir
            with zipfile.ZipFile("master.zip") as zf:
                zf.extractall(path=os.getcwd())

        except (urllib.error.URLError, TimeoutError) as e:
            raise e

    def time_events(self):
        # single load from the same EOS.bin file.
        _ = events(self.fi)

    def time_events_multi(self):
        # multiple loads from the same EOS.bin file
        f = [[1]]
        end_last_event = [127]
        while len(f[0]):
            f, end_last_event = events(
                self.fi, end_last_event=end_last_event[-1], max_frames=2400
            )
