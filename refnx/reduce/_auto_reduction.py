# Copyright (c) 2011-2026 A. R. J. Nelson, Australian Nuclear Science and Technology Organisation
# SPDX-License-Identifier: BSD-3-Clause

# Helper classes for auto-reduction

from watchdog.events import PatternMatchingEventHandler


class NXEH(PatternMatchingEventHandler):
    def __init__(self, queue):
        PatternMatchingEventHandler.__init__(self, patterns=["*.nx.hdf"])
        self.queue = queue

    def process(self, event):
        """
        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """
        self.queue.put(event)

    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)
