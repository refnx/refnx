import os.path

from refnx.reflect._code_fragment import code_fragment


class TestCodeFragment(object):
    def setup_method(self):
        pass

    def test_code_fragment(self):
        e361 = ReflectDataset(os.path.join(self.pth, 'e361r.txt'))

        code = code_fragment(None)
        print(code)
        eval(code)
