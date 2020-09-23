import cProfile
import logging
import pstats

from MC.doScatter import main


# log information
# mp.log_to_stderr()
# logger = mp.get_logger()
# logger.setLevel(logging.INFO)
#
# logging.basicConfig(level=logging.INFO,
#                 format='%(asctime)s:%(process)d:%(thread)d:%(message)s')
# logger = logging.getLogger()

# write out profile results
cProfile.run('main()', 'restats')

# read profile results
p = pstats.Stats('restats')

# interested in only 10 slowest calls
p.sort_stats(pstats.SortKey.TIME).print_stats(10)
