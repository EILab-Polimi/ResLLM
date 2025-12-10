#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.utils

Utility functions for the resllm project.

"""

import numpy as np


def cfs_to_taf(cfs):
    return cfs * 2.29568411 * 10**-5 * 86400 / 1000


def taf_to_cfs(taf):
    return taf * 1000 / 86400 * 43560


def select_closest_choice(numbers):
    """
    Given a list of numbers, select the closest choice from a predefined set
    (10, 20, 30, 40, 50, 60, 70, 80, 90, 100) based on a weighted scoring system.
    The scoring system assigns weights to distances from the choices:
    - 5: 10 points
    - 10: 5 points
    - 30: 3 points
    - 60: 2 points
    - 90: 1 point
    The choice with the highest score is returned.
    Parameters:
        numbers (list): A list of numbers to compare against the choices.
    Returns:
        int: The choice with the highest score.
    Example:
    >>> select_closest_choice([15, 25, 35, 40])
        30
    >>> select_closest_choice([5, 15, 25, 30])
        20
    """
    choices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    weights = {5: 10, 10: 5, 30: 3, 60: 2, 90: 1}

    def calculate_weighted_score(choice):
        score = 0
        for num in numbers:
            distance = abs(num - choice)
            # print(f"Choice: {choice}, Number: {num}, Distance: {distance}")
            for threshold, weight in weights.items():
                if distance <= threshold:
                    # print(f"Threshold: {threshold}, Weight: {weight}")
                    score += weight
                    break
        return score

    # Calculate scores for each choice and select the one with the highest score
    best_choice = max(choices, key=calculate_weighted_score)
    return best_choice


def water_day(d):
    """
    Converts a day of the year to a water day.
    Parameters:
        d (int): Day of the year (1-365/366).
    Returns:
        int: Water day (1-365/366).
    """
    return d - 274 if d >= 274 else d + 91
