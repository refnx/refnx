# -*- mode: python -*-
from os.path import join as pjoin
import os
import sys

block_cipher = None


pathex = pjoin('.', '')
uiloc = (pjoin('..', '..', 'refnx', 'reduce', '_app', 'ui', '*.ui'),
         pjoin('refnx', 'reduce', '_app', 'ui'))

a = Analysis(['slim.py'],
             pathex=[os.getcwd()],
             binaries=[],
             datas=[uiloc],
             hiddenimports=['refnx', 'refnx.dataset', 'refnx.reduce', 'refnx.reflect', 'refnx.util',
                            'refnx._lib', 'refnx.util', 'refnx.reduce._app', 'pandas',
                            'h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

# OSX
if sys.platform == 'darwin':
    exe = EXE(pyz,
              a.scripts,
              a.binaries,
              a.zipfiles,
              a.datas,
              name='slim',
              debug=False,
              strip=False,
              upx=True,
              console=False)
    app = BUNDLE(exe,
                 name='slim.app',
                 bundle_identifier=None)

# windows

elif sys.platform in ['win32', 'cygwin']:
    axe = EXE(pyz,
              a.scripts,
              a.binaries,
              a.zipfiles,
              a.datas,
              name='slim',
              debug=False,
              strip=False,
              upx=True,
              console=False)
