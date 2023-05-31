"""
Contains several pre-programmed lattices from elegant and other places
"""
from ocelot import Twiss


def PAR():
    pass


from ocelot.cpbd.elements import *


def LTP():
    """ Based on injRingMatch1 example """
    lattice = """
    "LTP:PH": HMONITOR, L=0.0
    "LTP:PH1": "LTP:PH"
    "LTP:PH2": "LTP:PH"
    "LTP:PH3": "LTP:PH"
    "LTP:PH4": "LTP:PH"
    "LTP:PV": VMONITOR, L=0.0
    "LTP:PV1": "LTP:PV"
    "LTP:PV2": "LTP:PV"
    "LTP:PV3": "LTP:PV"
    "LTP:PV4": "LTP:PV"

    "LTP:H": HKICKER, L=0.06
    "LTP:H1": "LTP:H"
    "LTP:H2": "LTP:H"
    "LTP:H3": "LTP:H"
    "LTP:H4": "LTP:H"
    "LTP:V": VKICKER, L=0.06
    "LTP:V1": "LTP:V"
    "LTP:V2": "LTP:V"
    "LTP:V3": "LTP:V"
    "LTP:V4": "LTP:V"
    "LTP:HV": KICKER, L=0.06

    LTPLMQ: DRIFT,L=0.18739
    LTPLMQ1: DRIFT, L=0.26603
    LTPLQC: DRIFT,L=0.12017
    LTPLQ5C: DRIFT, L=0.19867
    LTPLQC1: DRIFT, L=0.10525
    LTPL12: DRIFT, L=1.926950000000000
    LTPL11: DRIFT, L=2.173730000000000
    LTPL10: DRIFT, L=2.413060000000000
    LTPL9: DRIFT, L=1.946510000000000
    LTPL8A: DRIFT, L=1.462329
    LTPL8B: DRIFT, L=0.20099
    LTPL7: DRIFT, L=0.59678
    LTPL6: DRIFT, L=0.70942
    LTPL5: DRIFT, L=0.62634
    LTPL4: DRIFT, L=0.62634
    LTPL3: DRIFT, L=0.62634
    LTPL3A: DRIFT, L=0.42535
    LTPL3B: DRIFT, L=0.20099
    LTPL2A: DRIFT, L=0.3317
    LTPL2B: DRIFT, L=0.496950000000000
    LTPL1: DRIFT, L=2.904756
    LTPL1A: DRIFT, L=1.345630000000000
    LTPL1B: DRIFT, L=0.7262
    LTPL1C: DRIFT,L=0.15240
    LTPL1D: DRIFT, L=0.677476

    "LTP:Q10": QUADRUPOLE, L=0.3061, K1=1.32123855914
    "LTP:Q9": QUADRUPOLE, L=0.3061,  K1=-1.429201568445
    "LTP:Q8": QUADRUPOLE, L=0.3061,  K1=1.65302041674
    "LTP:Q7": QUADRUPOLE, L=0.3061,  K1=-1.782715766623
    "LTP:Q6": QUADRUPOLE, L=0.3061,  K1=2.927638995223
    "LTP:Q5": QUADRUPOLE, L=0.1491,  K1=-7.015653483830028
    "LTP:Q4": QUADRUPOLE, L=0.3061,  K1=4.908184713491
    "LTP:Q3": QUADRUPOLE, L=0.3061,  K1=-3.820413232879
    "LTP:Q2": QUADRUPOLE, L=0.3061,  K1=3.632979368008
    "LTP:Q1": QUADRUPOLE, L=0.3061,  K1=-0.934293465253
    "LTP:B1": SBEND, L=0.4, ANGLE=-0.2, E1=-0.1, E2=-0.1, HGAP=0.02, FINT=0.5
    "LTP:SP": SBEND, L=0.4, ANGLE=0.2, E1=0.2, E2=0.0, HGAP=0.01, FINT=0.5
    "LTP:FL1": MARK
    "LTP:FL2": MARK
    "LTP:FL3": MARK
    "LTP:CM1": MARK
    "LTP:END": MARK
    """

    """
    LTP1: LINE=("LTP:HV",LTPL12,"LTP:Q10",LTPLQC,"LTP:V4",LTPL11,&
        "LTP:Q9",LTPLQC,"LTP:H4",LTPL10,&
        "LTP:Q8",LTPL9,"LTP:PV4",LTPLMQ,"LTP:Q7",LTPLQC,"LTP:V3",&
        LTPL8A,"LTP:FL3",LTPL8B,"LTP:PH4",LTPLMQ,"LTP:Q6",LTPLQC,&
        "LTP:H3",LTPL7,"LTP:B1")
    LTP2: LINE=(LTPL6,"LTP:PV3",LTPLMQ1,"LTP:Q5",LTPLQ5C,"LTP:V2",&
        LTPL5,"LTP:PH3",LTPLMQ,"LTP:Q4",LTPLQC,"LTP:H2",LTPL4,&
        "LTP:PV2",LTPLMQ,"LTP:Q3",LTPLQC,"LTP:V1",LTPL3A,"LTP:FL2",&
        LTPL3B,"LTP:PH2",LTPLMQ,"LTP:Q2",LTPLQC1,"LTP:H1",&
        LTPL2A,"LTP:CM1",LTPL2B,"LTP:Q1",LTPL1A,"LTP:FL1",LTPL1B,&
        "LTP:PH1",LTPL1C,"LTP:PV1",LTPL1D,"LTP:SP","LTP:END")
    LTP: LINE=(LTP1,LTP2)
    """

    """
    PAR match at L1A
    s            betax          alphax          psix           etax          etaxp        xAperture        betay     
     alphay          psiy           etay          etayp        yAperture      pCentral0    ElementName  ElementOccurence 
    ElementType    ChamberShape         dI1            dI2            dI3            dI4            dI5   
    2.000000e-01   2.021119e+00  -9.994353e-02   9.961274e-02   5.415971e-03  -3.295975e-16   2.000000e-02   9.807002e+00 
    -2.040208e-02   2.039925e-02   0.000000e+00   0.000000e+00   1.000000e+01   8.800000e+02          L1A                 1 
    DRIF                 ?   0.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00   0.000000e+00 
    """
    elements = {}

    elements["LTP:PH"] = Monitor()
    elements["LTP:PH1"] = Monitor()
    elements["LTP:PH2"] = Monitor()
    elements["LTP:PH3"] = Monitor()
    elements["LTP:PH4"] = Monitor()
    elements["LTP:PV"] = Monitor()
    elements["LTP:PV1"] = Monitor()
    elements["LTP:PV2"] = Monitor()
    elements["LTP:PV3"] = Monitor()
    elements["LTP:PV4"] = Monitor()

    elements["LTP:H"] = Drift(l=0.06)
    elements["LTP:H1"] = Drift(l=0.06)
    elements["LTP:H2"] = Drift(l=0.06)
    elements["LTP:H3"] = Drift(l=0.06)
    elements["LTP:H4"] = Drift(l=0.06)
    elements["LTP:V"] = Drift(l=0.06)
    elements["LTP:V1"] = Drift(l=0.06)
    elements["LTP:V2"] = Drift(l=0.06)
    elements["LTP:V3"] = Drift(l=0.06)
    elements["LTP:V4"] = Drift(l=0.06)
    elements["LTP:HV"] = Drift(l=0.06)

    elements["LTPLMQ"] = Drift(l=0.18739)
    elements["LTPLMQ1"] = Drift(l=0.26603)
    elements["LTPLQC"] = Drift(l=0.12017)
    elements["LTPLQ5C"] = Drift(l=0.19867)
    elements["LTPLQC1"] = Drift(l=0.10525)
    elements["LTPL12"] = Drift(l=1.926950)
    elements["LTPL11"] = Drift(l=2.173730)
    elements["LTPL10"] = Drift(l=2.413060)
    elements["LTPL9"] = Drift(l=1.9465100)
    elements["LTPL8A"] = Drift(l=1.462329)
    elements["LTPL8B"] = Drift(l=0.20099)
    elements["LTPL7"] = Drift(l=0.59678)
    elements["LTPL6"] = Drift(l=0.70942)
    elements["LTPL5"] = Drift(l=0.62634)
    elements["LTPL4"] = Drift(l=0.62634)
    elements["LTPL3"] = Drift(l=0.62634)
    elements["LTPL3A"] = Drift(l=0.42535)
    elements["LTPL3B"] = Drift(l=0.20099)
    elements["LTPL2A"] = Drift(l=0.3317)
    elements["LTPL2B"] = Drift(l=0.496950)
    elements["LTPL1"] = Drift(l=2.904756)
    elements["LTPL1A"] = Drift(l=1.345630)
    elements["LTPL1B"] = Drift(l=0.7262)
    elements["LTPL1C"] = Drift(l=0.15240)
    elements["LTPL1D"] = Drift(l=0.677476)

    elements["LTP:Q10"] = Quadrupole(l=0.3061, k1=1.32123855914)
    elements["LTP:Q9"] = Quadrupole(l=0.3061, k1=-1.429201568445)
    elements["LTP:Q8"] = Quadrupole(l=0.3061, k1=1.65302041674)
    elements["LTP:Q7"] = Quadrupole(l=0.3061, k1=-1.782715766623)
    elements["LTP:Q6"] = Quadrupole(l=0.3061, k1=2.927638995223)
    elements["LTP:Q5"] = Quadrupole(l=0.1491, k1=-7.015653483830028)
    elements["LTP:Q4"] = Quadrupole(l=0.3061, k1=4.908184713491)
    elements["LTP:Q3"] = Quadrupole(l=0.3061, k1=-3.820413232879)
    elements["LTP:Q2"] = Quadrupole(l=0.3061, k1=3.632979368008)
    elements["LTP:Q1"] = Quadrupole(l=0.3061, k1=-0.934293465253)
    elements["LTP:B1"] = SBend(l=0.4, angle=-0.2, e1=-0.1, e2=-0.1, gap=0.02 * 2, fint=0.5)
    elements["LTP:SP"] = SBend(l=0.4, angle=0.2, e1=0.2, e2=0.0, gap=0.01 * 2, fint=0.5)

    elements["LTP:FL1"] = Marker()
    elements["LTP:FL2"] = Marker()
    elements["LTP:FL3"] = Marker()
    elements["LTP:CM1"] = Marker()
    elements["LTP:END"] = Marker()

    for k, el in elements.items():
        spl = k.split(':')
        el.id = spl[-1]

    ltp1 = ["LTP:HV", "LTPL12", "LTP:Q10", "LTPLQC", "LTP:V4", "LTPL11",
            "LTP:Q9", "LTPLQC", "LTP:H4", "LTPL10",
            "LTP:Q8", "LTPL9", "LTP:PV4", "LTPLMQ", "LTP:Q7", "LTPLQC", "LTP:V3",
            "LTPL8A", "LTP:FL3", "LTPL8B", "LTP:PH4", "LTPLMQ", "LTP:Q6", "LTPLQC",
            "LTP:H3", "LTPL7", "LTP:B1"]
    ltp2 = ["LTPL6", "LTP:PV3", "LTPLMQ1", "LTP:Q5", "LTPLQ5C", "LTP:V2",
            "LTPL5", "LTP:PH3", "LTPLMQ", "LTP:Q4", "LTPLQC", "LTP:H2", "LTPL4",
            "LTP:PV2", "LTPLMQ", "LTP:Q3", "LTPLQC", "LTP:V1", "LTPL3A", "LTP:FL2",
            "LTPL3B", "LTP:PH2", "LTPLMQ", "LTP:Q2", "LTPLQC1", "LTP:H1",
            "LTPL2A", "LTP:CM1", "LTPL2B", "LTP:Q1", "LTPL1A", "LTP:FL1", "LTPL1B",
            "LTP:PH1", "LTPL1C", "LTP:PV1", "LTPL1D", "LTP:SP", "LTP:END"]
    ltp1elems = [elements[eid] for eid in ltp1]
    ltp2elems = [elements[eid] for eid in ltp2]
    ltp = ltp1elems + ltp2elems

    return ltp

class LTP2:
    def __init__(self):
        """ Based on injRingMatch1 example """
        elements = {}

        elements["LTP:PH"] = Monitor()
        elements["LTP:PH1"] = Monitor()
        elements["LTP:PH2"] = Monitor()
        elements["LTP:PH3"] = Monitor()
        elements["LTP:PH4"] = Monitor()
        elements["LTP:PV"] = Monitor()
        elements["LTP:PV1"] = Monitor()
        elements["LTP:PV2"] = Monitor()
        elements["LTP:PV3"] = Monitor()
        elements["LTP:PV4"] = Monitor()

        elements["LTP:H"] = Hcor(l=0.06)
        elements["LTP:H1"] = Hcor(l=0.06)
        elements["LTP:H2"] = Hcor(l=0.06)
        elements["LTP:H3"] = Hcor(l=0.06)
        elements["LTP:H4"] = Hcor(l=0.06)
        elements["LTP:V"] = Vcor(l=0.06)
        elements["LTP:V1"] = Vcor(l=0.06)
        elements["LTP:V2"] = Vcor(l=0.06)
        elements["LTP:V3"] = Vcor(l=0.06)
        elements["LTP:V4"] = Vcor(l=0.06)
        elements["LTP:HV"] = Vcor(l=0.06)

        elements["LTPLMQ"] = Drift(l=0.18739)
        elements["LTPLMQ1"] = Drift(l=0.26603)
        elements["LTPLQC"] = Drift(l=0.12017)
        elements["LTPLQ5C"] = Drift(l=0.19867)
        elements["LTPLQC1"] = Drift(l=0.10525)
        elements["LTPL12"] = Drift(l=1.926950)
        elements["LTPL11"] = Drift(l=2.173730)
        elements["LTPL10"] = Drift(l=2.413060)
        elements["LTPL9"] = Drift(l=1.9465100)
        elements["LTPL8A"] = Drift(l=1.462329)
        elements["LTPL8B"] = Drift(l=0.20099)
        elements["LTPL7"] = Drift(l=0.59678)
        elements["LTPL6"] = Drift(l=0.70942)
        elements["LTPL5"] = Drift(l=0.62634)
        elements["LTPL4"] = Drift(l=0.62634)
        elements["LTPL3"] = Drift(l=0.62634)
        elements["LTPL3A"] = Drift(l=0.42535)
        elements["LTPL3B"] = Drift(l=0.20099)
        elements["LTPL2A"] = Drift(l=0.3317)
        elements["LTPL2B"] = Drift(l=0.496950)
        elements["LTPL1"] = Drift(l=2.904756)
        elements["LTPL1A"] = Drift(l=1.345630)
        elements["LTPL1B"] = Drift(l=0.7262)
        elements["LTPL1C"] = Drift(l=0.15240)
        elements["LTPL1D"] = Drift(l=0.677476)

        elements["LTP:Q10"] = Quadrupole(l=0.3061, k1=1.32123855914)
        elements["LTP:Q9"] = Quadrupole(l=0.3061, k1=-1.429201568445)
        elements["LTP:Q8"] = Quadrupole(l=0.3061, k1=1.65302041674)
        elements["LTP:Q7"] = Quadrupole(l=0.3061, k1=-1.782715766623)
        elements["LTP:Q6"] = Quadrupole(l=0.3061, k1=2.927638995223)
        elements["LTP:Q5"] = Quadrupole(l=0.1491, k1=-7.015653483830028)
        elements["LTP:Q4"] = Quadrupole(l=0.3061, k1=4.908184713491)
        elements["LTP:Q3"] = Quadrupole(l=0.3061, k1=-3.820413232879)
        elements["LTP:Q2"] = Quadrupole(l=0.3061, k1=3.632979368008)
        elements["LTP:Q1"] = Quadrupole(l=0.3061, k1=-0.934293465253)
        elements["LTP:B1"] = SBend(l=0.4, angle=-0.2, e1=-0.1, e2=-0.1, gap=0.02 * 2, fint=0.5)
        elements["LTP:SP"] = SBend(l=0.4, angle=0.2, e1=0.2, e2=0.0, gap=0.01 * 2, fint=0.5)

        elements["LTP:FL1"] = Marker()
        elements["LTP:FL2"] = Marker()
        elements["LTP:FL3"] = Marker()
        elements["LTP:CM1"] = Marker()
        elements["LTP:END"] = Marker()

        ltp1 = ["LTP:HV", "LTPL12", "LTP:Q10", "LTPLQC", "LTP:V4", "LTPL11",
                "LTP:Q9", "LTPLQC", "LTP:H4", "LTPL10",
                "LTP:Q8", "LTPL9", "LTP:PV4", "LTPLMQ", "LTP:Q7", "LTPLQC", "LTP:V3",
                "LTPL8A", "LTP:FL3", "LTPL8B", "LTP:PH4", "LTPLMQ", "LTP:Q6", "LTPLQC",
                "LTP:H3", "LTPL7", "LTP:B1"]
        ltp2 = ["LTPL6", "LTP:PV3", "LTPLMQ1", "LTP:Q5", "LTPLQ5C", "LTP:V2",
                "LTPL5", "LTP:PH3", "LTPLMQ", "LTP:Q4", "LTPLQC", "LTP:H2", "LTPL4",
                "LTP:PV2", "LTPLMQ", "LTP:Q3", "LTPLQC", "LTP:V1", "LTPL3A", "LTP:FL2",
                "LTPL3B", "LTP:PH2", "LTPLMQ", "LTP:Q2", "LTPLQC1", "LTP:H1",
                "LTPL2A", "LTP:CM1", "LTPL2B", "LTP:Q1", "LTPL1A", "LTP:FL1", "LTPL1B",
                "LTP:PH1", "LTPL1C", "LTP:PV1", "LTPL1D", "LTP:SP", "LTP:END"]
        self.elements = elements
        self.ltp1 = ltp1
        self.ltp2 = ltp2
        self.sequence = ltp1 + ltp2
    # ltp1elems = [elements[eid] for eid in ltp1]
    # ltp2elems = [elements[eid] for eid in ltp2]
    # ltp = ltp1elems + ltp2elems



def LTP_props():
    tw0 = Twiss()
    tw0.beta_x = 5.73409
    tw0.beta_y = 5.54673
    tw0.alpha_x = -1.74977
    tw0.alpha_y = 1.67146

    tw1 = Twiss()
    tw1.beta_x = 2.021119
    tw1.beta_y = 9.807002
    tw1.alpha_x = -9.994353e-02
    tw1.alpha_y = -2.040208e-02

    # Delta in k1 for Q1, Q2, ...
    solution8 = [1.063487642784899e-02, -1.227321491303734e-02, -4.623347377433529e-03, 1.859625460168495e-01,
                2.981512293469102e-03,  1.535776614578888e-02, -6.167335184456668e-03, -2.006271804780935e-02][::-1]
    solution2 = [9.726812187365041e-02, -4.030189860345601e-02]

    return {'tw0': tw0, 'tw1': tw1, 'solution2': solution2, 'solution8': solution8}
