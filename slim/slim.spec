# -*- mode: python -*-

block_cipher = None


a = Analysis(['slim.py'],
             binaries=[],
             datas=[('ui/*.ui', 'ui'), ('icons/*.png', 'icons')],
             hiddenimports=['h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='slim',
          debug=False,
          strip=False,
          upx=True,
          console=False , icon='icons/Motofit.icns')
app = BUNDLE(exe,
             name='slim.app',
             icon='icons/Motofit.icns',
             bundle_identifier=None)
