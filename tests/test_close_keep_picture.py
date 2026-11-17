"""2026-11-17: Show4DSTEM.close(keep_picture=True) frees the scan but keeps the widget open, so its last images stay
in the notebook; nothing answers the front end afterwards, and the title says so."""

import numpy as np

from quantem.widget import Show4DSTEM


def test_keep_picture_releases_the_scan_and_keeps_the_widget():
    viewer = Show4DSTEM(np.random.default_rng(0).poisson(2.0, (4, 4, 8, 8)).astype(np.uint16))
    viewer.title = "scan 7"
    viewer.close(keep_picture=True)
    assert viewer.comm is not None                                  # the widget, and its images, stay
    assert viewer._data is None                                     # the scan is released
    assert viewer.title == "scan 7 · scan released: picture only" and viewer.show_title
    viewer.pos_row, viewer.pos_col = 2, 3                           # a drag afterwards: nothing answers, no error
    viewer.close(keep_picture=True)                                 # again: unchanged
    assert viewer.title == "scan 7 · scan released: picture only"
    viewer.close()                                                  # and it can still be closed for good
    assert viewer.comm is None


def test_plain_close_still_closes_the_widget():
    viewer = Show4DSTEM(np.zeros((2, 2, 4, 4), np.uint16))
    viewer.close()
    assert viewer.comm is None
