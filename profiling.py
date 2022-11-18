import cProfile
import io
import pstats


def get_str_profile(target_func):
    pr = cProfile.Profile()
    pr.enable()
    target_func()
    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    return s.getvalue()
