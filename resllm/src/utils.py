#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.utils

Utility functions for the resllm project.

"""


def cfs_to_taf(cfs):
    return cfs * 2.29568411 * 10**-5 * 86400 / 1000


def taf_to_cfs(taf):
    return taf * 1000 * 43560 / 86400


def water_day(d):
    """
    Converts a day of the year to a water day.
    Parameters:
        d (int): Day of the year (1-365/366).
    Returns:
        int: Water day (1-365/366).
    """
    return d - 274 if d >= 274 else d + 91
