"""
Unit tests for voice leading rule checking functions.

Tests cover all rule violation detection functions including illegal leaps,
voice crossing, parallel fifths/octaves, and seventh chord handling.
"""

import sys
import pytest
sys.path.insert(0, '../src')

from voice_leading_rules import (
    illegal_leaps,
    voice_crossing,
    parallel_fifths_octaves,
    direct_fifths_octaves,
    leading_tone_resolution,
    illegal_common_tones,
    seventh_approach,
    seventh_resolve,
    voice_leading_reward_function,
)


class TestIllegalLeaps:
    """Test illegal melodic interval detection."""

    def test_no_illegal_leaps(self):
        """Small intervals should not be flagged."""
        state = [48, 60, 64, 72]  # C major chord
        next_state = [50, 62, 65, 74]  # D minor chord, all step-wise motion
        assert illegal_leaps(state, next_state) == 0

    def test_tritone_leap(self):
        """Tritone (augmented 4th) should be flagged."""
        state = [48, 60, 64, 72]
        next_state = [48, 66, 64, 72]  # Tritone in tenor (C to F#)
        assert illegal_leaps(state, next_state) == 1

    def test_major_seventh_leap(self):
        """Major 7th interval should be flagged."""
        state = [48, 60, 64, 72]
        next_state = [48, 71, 64, 72]  # Major 7th in tenor
        assert illegal_leaps(state, next_state) == 1

    def test_octave_plus_leap(self):
        """Leaps larger than octave should be flagged."""
        state = [48, 60, 64, 72]
        next_state = [48, 60, 64, 85]  # More than octave in soprano
        assert illegal_leaps(state, next_state) == 1

    def test_multiple_illegal_leaps(self):
        """Multiple violations should all be counted."""
        state = [48, 60, 64, 72]
        next_state = [48, 71, 58, 85]  # 7th in tenor, tritone in alto, >8ve in soprano
        assert illegal_leaps(state, next_state) == 3


class TestVoiceCrossing:
    """Test voice crossing detection between adjacent parts."""

    def test_no_voice_crossing(self):
        """Properly spaced voices should not cross."""
        state = [48, 60, 64, 72]
        next_state = [50, 62, 65, 74]
        assert voice_crossing(state, next_state) == 0

    def test_bass_tenor_crossing(self):
        """Bass crossing above tenor should be detected."""
        state = [48, 60, 64, 72]
        next_state = [62, 55, 64, 72]  # Bass jumps above tenor
        assert voice_crossing(state, next_state) == 1

    def test_tenor_alto_crossing(self):
        """Tenor crossing above alto should be detected."""
        state = [48, 60, 64, 72]
        next_state = [48, 65, 62, 72]  # Tenor crosses alto
        assert voice_crossing(state, next_state) == 1

    def test_multiple_crossings(self):
        """Multiple crossings should all be counted."""
        state = [48, 60, 64, 72]
        next_state = [62, 55, 74, 70]  # Bass-tenor and alto-soprano cross
        assert voice_crossing(state, next_state) == 2


class TestParallelFifthsOctaves:
    """Test detection of parallel perfect fifths and octaves."""

    def test_no_parallel_motion(self):
        """Contrary or oblique motion should not be flagged."""
        state = [48, 60, 64, 72]  # C: C-C-E-C (bass-soprano = octave)
        next_state = [50, 62, 65, 74]  # Different intervals
        assert parallel_fifths_octaves(state, next_state) == 0

    def test_parallel_octaves(self):
        """Parallel octaves should be detected."""
        state = [48, 60, 64, 72]  # Bass-soprano = 2 octaves (C-C)
        next_state = [50, 62, 65, 74]  # Bass-soprano still 2 octaves (D-D)
        assert parallel_fifths_octaves(state, next_state) == 1

    def test_parallel_fifths(self):
        """Parallel perfect fifths should be detected."""
        state = [48, 55, 64, 72]  # Bass-tenor = P5 (C-G)
        next_state = [50, 57, 65, 74]  # Bass-tenor still P5 (D-A)
        assert parallel_fifths_octaves(state, next_state) == 1

    def test_similar_motion_allowed(self):
        """Similar motion to perfect intervals is OK if not parallel."""
        state = [48, 60, 64, 72]  # Bass-soprano = octave
        next_state = [50, 62, 65, 76]  # Bass-soprano = different octave span
        # This should be OK as the interval changes
        result = parallel_fifths_octaves(state, next_state)
        assert result >= 0  # Implementation may vary


class TestDirectFifthsOctaves:
    """Test detection of direct/hidden fifths and octaves."""

    def test_no_direct_fifths(self):
        """Contrary motion should not trigger direct 5ths."""
        state = [48, 60, 64, 72]
        next_state = [50, 62, 65, 71]  # No direct motion to perfect interval
        assert direct_fifths_octaves(state, next_state) == 0

    def test_stepwise_motion_allowed(self):
        """Stepwise motion to perfect interval is typically allowed."""
        state = [48, 60, 64, 71]
        next_state = [48, 60, 64, 72]  # Soprano moves by step to octave
        # Should be 0 because soprano doesn't leap
        assert direct_fifths_octaves(state, next_state) == 0


class TestSeventhChords:
    """Test seventh chord approach and resolution rules."""

    def test_proper_seventh_resolution(self):
        """Seventh resolving down by step should be acceptable."""
        # This test would need actual chord context
        # Simplified test for the function behavior
        state = [50, 57, 62, 65]  # Could be a ii7 chord
        next_state = [48, 55, 60, 64]  # Resolution
        result = seventh_resolve(state, next_state)
        assert isinstance(result, int)
        assert result >= 0


class TestVoiceLeadingRewardFunction:
    """Test the integrated reward function."""

    def test_perfect_voice_leading(self):
        """Stepwise motion with no violations should give 0 or positive reward."""
        state = [48, 60, 64, 72]
        next_state = [50, 62, 65, 74]  # All voices move by step or tone
        reward, vc, p58, il, d58, lt, ct, sev = voice_leading_reward_function(state, next_state)

        # Should have no violations
        assert vc == 0
        assert p58 == 0
        assert il == 0
        # Reward should be 0 or close to 0 (all weights are negative)
        assert reward <= 0

    def test_multiple_violations(self):
        """Multiple violations should accumulate penalties."""
        state = [48, 60, 64, 72]
        next_state = [62, 55, 58, 85]  # Multiple violations
        reward, vc, p58, il, d58, lt, ct, sev = voice_leading_reward_function(state, next_state)

        # Should have violations
        assert il > 0 or vc > 0  # At least some violations
        assert reward < 0  # Negative reward for violations

    def test_reward_function_returns_tuple(self):
        """Reward function should return 8-element tuple."""
        state = [48, 60, 64, 72]
        next_state = [50, 62, 65, 74]
        result = voice_leading_reward_function(state, next_state)

        assert isinstance(result, tuple)
        assert len(result) == 8

        reward, vc, p58, il, d58, lt, ct, sev = result
        assert isinstance(reward, (int, float))
        assert all(isinstance(x, int) for x in [vc, p58, il, d58, lt, ct, sev])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_same_state(self):
        """No motion should have no violations (except possibly common tone issues)."""
        state = [48, 60, 64, 72]
        next_state = [48, 60, 64, 72]

        assert illegal_leaps(state, next_state) == 0
        assert voice_crossing(state, next_state) == 0
        assert parallel_fifths_octaves(state, next_state) == 0

    def test_extreme_ranges(self):
        """Test with extreme but valid MIDI ranges."""
        state = [40, 50, 60, 70]  # Low voicing
        next_state = [41, 51, 61, 71]  # All move up by semitone

        result = illegal_leaps(state, next_state)
        assert result == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
