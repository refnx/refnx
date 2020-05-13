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
lipid_data = (pjoin('..', '..', 'refnx', 'reflect', '_app', 'lipids.json'),
              pjoin('refnx', 'reflect', '_app'))

block_cipher = None


a = Analysis(['motofit.py'],
             pathex=['F:\\programming\\refnx\\misc\\app'],
             binaries=[],
             datas=[uiloc,
                    icons,
                    licences,
                    lipid_data],
             hiddenimports=['periodictable', 'refnx',
                            'refnx.analysis', 'refnx.dataset', 'refnx.reflect',
                            'refnx.reflect._app', 'distutils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='motofit',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='motofit')
