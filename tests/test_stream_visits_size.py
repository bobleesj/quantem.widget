"""A scan kept as visits never takes an image's place in the live view (11-19): the image in view stays while it
arrives, when it is whole, and after it is aborted; its grid joins the scans in view."""

from types import SimpleNamespace

from quantem.widget.show4dstem.streaming import StreamFollower


class Scan:
    def __init__(self, shape, complete=True, visits=False):
        self.shape, self.complete = shape, complete
        self.metadata = {"scan_positions": [0], "scan_positions_shape": list(visits)} if visits else {}


def _follower():
    viewer = SimpleNamespace(size_groups=[], size_group=0)
    return viewer, StreamFollower(viewer, SimpleNamespace(datasets=[]), [], labels=None, refresh_s=0.1, options={})


def test_the_image_in_view_stays_while_a_visits_scan_arrives_completes_or_is_aborted():
    viewer, follower = _follower()
    grid = Scan((512, 512, 192, 192))
    assert follower.of_shown_size([grid]) == [grid]
    arriving = Scan((1, 262144, 192, 192), complete=False, visits=(512, 512))
    assert follower.of_shown_size([grid, arriving]) == [grid]
    assert follower.arriving is arriving and viewer.size_groups == []      # counted in the title; no size tab for it
    arriving.complete = True                                # whole, its grid not made yet
    assert follower.of_shown_size([grid, arriving]) == [grid] and follower.arriving is None
    its_grid = Scan((512, 512, 192, 192))
    assert follower.of_shown_size([grid, arriving, its_grid]) == [grid, its_grid]
    second = Scan((1, 262144, 192, 192), complete=False, visits=(512, 512))
    assert follower.of_shown_size([grid, arriving, its_grid, second]) == [grid, its_grid]
    assert follower.of_shown_size([grid, arriving, its_grid]) == [grid, its_grid]      # aborted: the image is still there


def test_other_scans_bring_their_size_into_view_as_before():
    _, follower = _follower()
    grid = Scan((512, 512, 192, 192))
    follower.of_shown_size([grid])
    smaller = Scan((256, 256, 192, 192), complete=False)
    assert follower.of_shown_size([grid, smaller]) == [smaller]
    larger = Scan((1, 1048576, 192, 192), complete=False, visits=(1024, 1024))   # visits of another size: held too
    assert follower.of_shown_size([grid, smaller, larger]) == [smaller] and follower.arriving is larger
    _, first = _follower()
    only = Scan((1, 4096, 192, 192), complete=False, visits=(64, 64))
    assert first.of_shown_size([only]) == [only]            # nothing else to show
