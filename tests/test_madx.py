from unittest import TestCase
from pathlib import Path
import pyIOTA.madx as madx

CWD = Path(__file__).parent


class TestMADX(TestCase):
    def test_madx_parse_iota(self):
        fpath = CWD / 'files' / 'IOTA_1IO_v86_twiss.tfs'
        seq, _, _, _, _ = madx.parse_lattice(fpath)
        totallen = sum(el.l for el in seq)
        self.assertEqual(totallen, 39.96606465400015)



