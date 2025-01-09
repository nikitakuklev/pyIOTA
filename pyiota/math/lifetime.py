import logging

import caproto.threading.client
import numpy as np

"""
Script to get lifetime based on loss threshold
"""

logger = logging.getLogger(__name__)

ctx = caproto.threading.client.Context()
data: dict = {}
pvs: dict = {}
subs: dict = {}

DCCT = 'S35DCCT:currentCC'
COUPLING = 'S:VID1:filteredCoupling'
PVLIST = [DCCT, COUPLING]


def compute_lifetime_exponential(times, current_data, offset=0.0, normalized=False,
                                 coupling=None, reference_current=25.0, normalized_power=2/3):
    # N = N0*exp(-lambda*t)
    # ln(N) = -lambda*t + ln(N0)
    # y = a*t+b
    # lifetime = 1/lambda = 1/-slope
    # normlifetime = lifetime * (current^2/3 / ref_current^2/3) * (1/sqrt(couplingratio))

    if len(current_data) < 2:
        raise ValueError('Error - not enough current datapoints')
    assert np.all(current_data > 0)
    if normalized:
        assert coupling is not None and coupling > 0
        assert reference_current > 0
    current_data += offset
    avg_current = current_data.mean()

    lncurrent = np.log(current_data)
    coef, cov = np.polyfit(times, lncurrent, 1, cov=True)  # cov
    stdev = np.sqrt(np.diag(cov))
    slope = coef[0]

    lifetime = -1 / slope / 3600
    # z2 = np.polyfit(times, current_data, 1)
    # logger.debug(f'Unnormalized lifetime is %.3f h (linear lifetime %.3f)', lifetime, -avg_current/z2[0]/3600)
    logger.debug('Unnormalized lifetime is %.3fh', lifetime)

    if normalized:
        normalized_lifetime = lifetime * (avg_current / reference_current) ** normalized_power
        normalized_lifetime = normalized_lifetime * np.sqrt(1.0) / np.sqrt(coupling)
        logger.debug(f'Normalized lifetime is {normalized_lifetime:.3f}h')
        print(normalized_lifetime)


class LifetimeMeasurer:
    def __init__(self, mode='relative_change', mode_kwargs=None):
        assert mode in ['relative_change', 'absolute_change']
        self.mode = mode
        self.mode_kwargs = mode_kwargs
        self.time_slices = []
        self.data_slices = []
        self.initial_current = None

    def update(self, time: np.ndarray, data: np.ndarray, aux_df=None):
        if self.initial_current is None:
            self.initial_current = data[0]
        if self.mode == 'relative_change':
            threshold = self.initial_current * (1 - self.mode_kwargs['threshold'])
        elif self.mode == 'absolute_change':
            threshold = self.initial_current - self.mode_kwargs['threshold']
        # if current_data[-1] > threshold:
        #     raise ValueError(f'Error - too small current change {current_data[0]} -> {current_data[-1]}')

        return False

# class EPICSLifetimeMeasurement:
#     def callback(sub, response):
#         name = sub.pv.name
#         data[name].append(response)
#
#     def setup_data_collection(pv_name):
#         """ Set up EPICS monitoring for pv"""
#         pv, = ctx.get_pvs(pv_name)
#
#         # read once to populate lists...this will duplicate first reading, but not a big deal
#         # response = pv.read(data_type='time')
#         data[pv_name] = []
#
#         subscription = pv.subscribe(data_type='time')
#         subscription.add_callback(callback)
#         subs[pv_name] = subscription
#         pvs[pv_name] = pv
#
#     def get_data_from_responses(responses):
#         times = []
#         values = []
#         for r in responses:
#             times.append(r.metadata.timestamp)
#             values.append(r.data[0])
#         return np.array(times), np.array(values)
#
#     def compute_lifetime(args):
#         logging.basicConfig()
#         logging.getLogger('caproto').propagate = False
#         if args.verbose:
#             logger.setLevel(logging.DEBUG)
#         else:
#             logging.disable()
#
#         assert args.maxTime > args.minTime
#         assert args.maxCurrentChange > args.minCurrentChange
#
#         t_start = time.time()
#         delta = lambda: time.time() - t_start
#
#         for pv in PVLIST:
#             setup_data_collection(pv)
#
#         for pv in pvs.values():
#             pv.wait_for_connection()
#
#         while True:
#             if delta() < args.minTime:
#                 time.sleep(0.1)
#                 continue
#
#             if delta() > args.maxTime:
#                 logger.debug(f'Stopping in {delta():.2f}s due to exceeding max time')
#                 break
#
#             times, current_data = get_data_from_responses(data[DCCT])
#             if len(current_data) == 1:
#                 time.sleep(0.1)
#                 continue
#
#             if args.relativeChangeMode:
#                 threshold = current_data[0] * (1 - args.maxCurrentChange)
#             else:
#                 threshold = current_data[0] - args.maxCurrentChange
#             if current_data[-1] < threshold:
#                 logger.debug(f'Stopping in {delta():.2f}s due to exceeding current loss {current_data[0] - threshold}')
#                 break
#
#         t_elapsed = delta()
#
#         for pv in pvs.values():
#             pv.unsubscribe_all()
#
#         if args.normalized:
#             times2, coupling_data = get_data_from_responses(data[COUPLING])
#             if len(coupling_data) < 2:
#                 raise ValueError('Error - not enough coupling datapoints')
#             logger.debug(f'Got {len(coupling_data)} coupling points in {t_elapsed:.2f}s')
#             avg_coupling = coupling_data.mean()
#             logger.debug(f'Average coupling is {avg_coupling:.3f}')
#             normalized_lifetime = lifetime * (avg_current / args.currentRef) ** args.normalizedPower
#             normalized_lifetime = normalized_lifetime * np.sqrt(1.0) / np.sqrt(avg_coupling)
#             logger.debug(f'Normalized lifetime is {normalized_lifetime:.3f}h')
#             print(normalized_lifetime)
#         else:
#             print(lifetime)
#
#     if __name__ == '__main__':
#         parser = argparse.ArgumentParser()
#         parser.add_argument('-minCurrentChange', type=float, default=0.0002, help='min fractional/abs current change')
#         parser.add_argument('-maxCurrentChange', type=float, default=0.0006, help='max fractional/abs current change')
#         parser.add_argument('-relativeChangeMode', action='store_true', default=True)
#         parser.add_argument('-maxTime', type=float, default=25.0)
#         parser.add_argument('-minTime', type=float, default=10.0)
#         parser.add_argument('-currentRef', type=float, default=25.0)
#         parser.add_argument('-normalized', action='store_true', default=True)
#         parser.add_argument('-normalizedPower', type=float, default=2 / 3)
#         parser.add_argument('-verbose', action='store_true', default=False)
#         args = parser.parse_args()
#
#         compute_lifetime(args)
