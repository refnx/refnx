# -*- mode: python -*-
import os.path
import sys


UI = os.path.join('ui', '*.ui')
ICONS = os.path.join('icons', '*.png')

block_cipher = None

a = Analysis(['slim.py'],
             binaries=[],
             datas=[(UI, 'ui'), (ICONS, 'icons')],
             hiddenimports=['h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy'],
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
              console=False , icon='icons/scattering.icns')
    app = BUNDLE(exe,
                 name='slim.app',
                 icon='icons/scattering.icns',
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
              console=False , icon='icons\\scattering.ico')
