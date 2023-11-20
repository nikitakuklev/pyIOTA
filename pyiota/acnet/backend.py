from concurrent.futures import ThreadPoolExecutor
from enum import Enum
# from frontends import DoubleDevice
#
#
# class DBR_TYPE(Enum):
#     DOUBLE = 1
#     STATUS = 2
#
#     ARRAYDOUBLE = 0
#
#
# class AcnetContext:
#     def __init__(self):
#         self.executor = ThreadPoolExecutor(1)
#
#     def get_pvs(self, names, type=DBR_TYPE.DOUBLE):
#         for name in names:
#             dev = DoubleDevice(name)
