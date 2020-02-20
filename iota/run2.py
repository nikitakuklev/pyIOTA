class BPMS:

    @staticmethod
    def to_acnet(names: list):
        return ['N:' + n for n in names]

    @staticmethod
    def from_acnet(names: list):
        return [n[2:] if n[0:2] == 'N:' else n for n in names]

    @staticmethod
    def add_axis(names: list, axis: str):
        assert axis in ['H', 'V', 'S']
        return [n + axis for n in names]

    ROOT_NAMES = ['IBB2R', 'IBC1R', 'IBC2R', 'IBD1R', 'IBD2R', 'IBE1R', 'IBE2R', 'IBE2L', 'IBE1L', 'IBD2L', 'IBD1L',
                  'IBC2L', 'IBC1L', 'IBB2L', 'IBB1L', 'IBA3L', 'IBA2L', 'IBA1C', 'IBA2R', 'IBA3R', 'IBB1R']

    ACNET_NAMES = to_acnet(ROOT_NAMES)

    H = add_axis(ROOT_NAMES, 'H')
    HA = to_acnet(H)

    V = add_axis(ROOT_NAMES, 'V')
    VA = to_acnet(H)

    S = add_axis(ROOT_NAMES, 'S')
    SA = to_acnet(H)
