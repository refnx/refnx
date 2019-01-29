# -*- mode: python ; coding: utf-8 -*-
from os.path import join as pjoin
import os
import sys

block_cipher = None


pathex = pjoin('.', '')
uiloc = (pjoin('..', '..', 'refnx', 'reflect', '_app', 'ui', '*.ui'),
         pjoin('refnx', 'reflect', '_app', 'ui'))
icons = (pjoin('..', '..', 'refnx', 'reflect', '_app', 'icons', '*.png'),
         pjoin('refnx', 'reflect', '_app', 'icons'))
licences = (pjoin('..', '..', 'refnx', 'reflect', '_app', 'ui', 'licences', '*'),
            pjoin('refnx', 'reflect', '_app', 'ui', 'licences'))

a = Analysis(['motofit.py'],
             pathex=[os.getcwd()],
             binaries=[],
             datas=[uiloc,
                    icons,
                    licences],
             hiddenimports=['periodictable', 'refnx',
                            'refnx.analysis', 'refnx.dataset', 'refnx.reflect',
                            'refnx.reflect._app'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

# OSX
if sys.platform == 'darwin':
    exe = EXE(pyz,
              a.scripts,
              a.binaries,
              a.zipfiles,
              a.datas,
              name='motofit',
              debug=False,
              strip=False,
              upx=True,
              console=False , icon='../../refnx/reflect/_app/icons/Motofit.icns')
    app = BUNDLE(exe,
                 name='motofit.app',
                 icon='../../refnx/reflect/_app/icons/Motofit.icns',
                 bundle_identifier=None,
                 info_plist={
                    'NSHighResolutionCapable': 'True'},
                 )

# windows

elif sys.platform in ['win32', 'cygwin']:
    exe = EXE(pyz,
      a.scripts,
      a.binaries,
      a.zipfiles,
      a.datas,
      name='motofit',
      bootloader_ignore_signals=False,
      debug=False,
      strip=False,
      upx=False,
      console=False , icon='..\\..\\refnx\\reflect\\_app\\icons\\scattering.ico')

"""
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='motofit',
          debug=False,
          bootloader_ignore_signals=False,
          strip=True,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=True,
               upx=True,
               name='motofit')
"""

