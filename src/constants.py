"""
Constants used throughout the RL voice leading project.

This module contains reward weights, musical intervals, and other configuration constants.
"""

# Voice Leading Reward Weights
# These weights determine the penalty for various voice leading rule violations
# Values were determined empirically during model training
ILLEGAL_LEAP_WEIGHT = 0.1407
VOICE_CROSSING_WEIGHT = 0.1281
PARALLEL_FIFTHS_OCTAVES_WEIGHT = 0.0919
DIRECT_FIFTHS_OCTAVES_WEIGHT = 0.1229
ILLEGAL_COMMON_TONE_WEIGHT = 0.1063
ILLEGAL_LEADING_TONE_RESOLUTION_WEIGHT = 0.1291
ILLEGAL_SEVENTH_APPROACH_WEIGHT = 0.1427
ILLEGAL_SEVENTH_RESOLUTION_WEIGHT = 0.1383

# Musical Intervals (in semitones)
PERFECT_FIFTH = 7
PERFECT_OCTAVE = 12
TRITONE = 6
MAJOR_SEVENTH = 11

# MIDI Note Ranges for each voice part
# Format: (min_note, max_note)
BASS_RANGE = (40, 64)    # E2 to E4
TENOR_RANGE = (48, 69)   # C3 to A4
ALTO_RANGE = (53, 76)    # F3 to E5
SOPRANO_RANGE = (60, 81) # C4 to A5
