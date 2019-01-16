.. _gui_chapter:

===
gui
===

For simple slab models *refnx* offers either a simple *Jupyter* graphical user
interface (GUI), or a more sophisticated *PyQt* interface. To use the latter
from a console one needs to have the *pyqt5, periodictable* packages installed.
The PyQt interface is still in alpha development. To start the gui from the
interpreter:

    ::

     >>> from refnx.reflect import gui
     >>> gui()
