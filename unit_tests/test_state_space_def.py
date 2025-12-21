"""
Unit tests for state space and chord analysis functions.

Tests cover chord identification, inversion detection, and voicing validation.
"""

import sys
import pytest
sys.path.insert(0, '../src')

from state_space_def import (
    determine_chord_from_voicing,
    determine_inversion,
    is_complete,
    doubled_leading_tone,
    check_inv_triad_complete,
    note_names_to_numbers,
)


class TestDetermineChordFromVoicing:
    """Test chord identification from voicings."""

    def test_c_major_triad(self):
        """Should identify C major (I) chord."""
        voicing = [48, 60, 64, 72]  # C-C-E-C
        chord = determine_chord_from_voicing(voicing)
        assert chord == 1  # I chord in C major

    def test_g_major_triad(self):
        """Should identify G major (V) chord."""
        voicing = [55, 59, 62, 67]  # G-B-D-G
        chord = determine_chord_from_voicing(voicing)
        assert chord == 5  # V chord in C major

    def test_d_minor_triad(self):
        """Should identify D minor (ii) chord."""
        voicing = [50, 57, 62, 65]  # D-A-D-F
        chord = determine_chord_from_voicing(voicing)
        assert chord == 2  # ii chord in C major

    def test_seventh_chord(self):
        """Should identify seventh chords."""
        voicing = [55, 59, 62, 65]  # G-B-D-F (G7)
        chord = determine_chord_from_voicing(voicing)
        assert chord == 10  # V7 chord

    def test_inverted_chord(self):
        """Should identify chord regardless of inversion."""
        voicing = [52, 60, 64, 67]  # E-C-E-G (C major, 1st inversion)
        chord = determine_chord_from_voicing(voicing)
        assert chord == 1  # Still I chord


class TestDetermineInversion:
    """Test chord inversion identification."""

    def test_root_position(self):
        """Root in bass should be root position (0)."""
        voicing = [48, 60, 64, 72]  # C in bass
        chord = 1  # I chord
        inversion = determine_inversion(voicing, chord)
        assert inversion == 0

    def test_first_inversion(self):
        """Third in bass should be first inversion (1)."""
        voicing = [52, 60, 64, 67]  # E in bass (C major 1st inv)
        chord = 1  # I chord
        inversion = determine_inversion(voicing, chord)
        assert inversion == 1

    def test_second_inversion(self):
        """Fifth in bass should be second inversion (2)."""
        voicing = [55, 60, 64, 67]  # G in bass (C major 2nd inv)
        chord = 1  # I chord
        inversion = determine_inversion(voicing, chord)
        assert inversion == 2

    def test_seventh_chord_third_inversion(self):
        """Seventh in bass should be third inversion (3)."""
        voicing = [53, 59, 62, 67]  # F in bass (G7 3rd inv)
        chord = 10  # V7 chord
        inversion = determine_inversion(voicing, chord)
        assert inversion == 3


class TestIsComplete:
    """Test chord completeness checking."""

    def test_complete_triad(self):
        """Triad with all three notes should be complete."""
        voicing = [48, 60, 64, 67]  # C-C-E-G (all chord tones present)
        chord = 1  # I chord
        assert is_complete(voicing, chord) is True

    def test_incomplete_triad(self):
        """Triad missing a chord tone should be incomplete."""
        voicing = [48, 60, 64, 72]  # C-C-E-C (no fifth)
        chord = 1  # I chord
        result = is_complete(voicing, chord)
        # Doubled third, missing fifth
        assert result is False

    def test_complete_seventh(self):
        """Seventh chord with all four notes should be complete."""
        voicing = [55, 59, 62, 65]  # G-B-D-F (all chord tones)
        chord = 10  # V7
        assert is_complete(voicing, chord) is True

    def test_incomplete_seventh(self):
        """Seventh chord missing a note should be incomplete."""
        voicing = [55, 59, 62, 67]  # G-B-D-G (no seventh)
        chord = 10  # V7
        assert is_complete(voicing, chord) is False


class TestDoubledLeadingTone:
    """Test detection of doubled leading tone."""

    def test_no_doubled_leading_tone(self):
        """Single leading tone should not be flagged."""
        voicing = [48, 59, 64, 67]  # C-B-E-G (one B)
        result = doubled_leading_tone(voicing)
        assert result == 0

    def test_doubled_leading_tone(self):
        """Two leading tones should be flagged."""
        voicing = [59, 59, 64, 71]  # B-B-E-B (three Bs)
        result = doubled_leading_tone(voicing)
        assert result == 1

    def test_no_leading_tone(self):
        """No leading tone should not be flagged."""
        voicing = [48, 60, 64, 67]  # C-C-E-G (no B)
        result = doubled_leading_tone(voicing)
        assert result == 0


class TestCheckInvTriadComplete:
    """Test inverted triad completeness rule."""

    def test_complete_first_inversion(self):
        """Complete first inversion should pass."""
        voicing = [52, 60, 64, 67]  # E-C-E-G (first inversion, complete)
        result = check_inv_triad_complete(voicing)
        assert result == 0  # No violation

    def test_incomplete_first_inversion(self):
        """Incomplete first inversion should be flagged."""
        voicing = [52, 60, 64, 76]  # E-C-E-E (first inv, missing G)
        result = check_inv_triad_complete(voicing)
        # This should detect the incomplete triad
        assert result == 1


class TestNoteNamesToNumbers:
    """Test conversion from note names to MIDI numbers."""

    def test_single_notes(self):
        """Convert single note names."""
        notes = note_names_to_numbers(["C4", "E4", "G4"])
        assert notes == [60, 64, 67]

    def test_different_octaves(self):
        """Handle different octaves correctly."""
        notes = note_names_to_numbers(["C3", "C4", "C5"])
        assert notes == [48, 60, 72]

    def test_accidentals(self):
        """Handle sharps and flats."""
        notes = note_names_to_numbers(["C#4", "Bb3"])
        expected = [61, 58]  # C# is 61, Bb is 58
        assert notes == expected


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_low_voicing(self):
        """Test with low MIDI numbers."""
        voicing = [36, 48, 52, 55]  # Very low C major
        chord = determine_chord_from_voicing(voicing)
        assert chord == 1

    def test_very_high_voicing(self):
        """Test with high MIDI numbers."""
        voicing = [60, 72, 76, 84]  # High C major
        chord = determine_chord_from_voicing(voicing)
        assert chord == 1

    def test_close_voicing(self):
        """Test with very close spacing."""
        voicing = [60, 64, 67, 72]  # Close position C major
        chord = determine_chord_from_voicing(voicing)
        assert chord == 1

    def test_open_voicing(self):
        """Test with wide spacing."""
        voicing = [48, 64, 67, 84]  # Open position C major
        chord = determine_chord_from_voicing(voicing)
        assert chord == 1


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_chord_and_inversion_together(self):
        """Test chord identification and inversion detection."""
        voicing = [55, 59, 62, 65]  # G7 in root position

        chord = determine_chord_from_voicing(voicing)
        assert chord == 10  # V7

        inversion = determine_inversion(voicing, chord)
        assert inversion == 0  # Root position

        complete = is_complete(voicing, chord)
        assert complete is True

    def test_inverted_seventh_chord(self):
        """Test inverted seventh chord analysis."""
        voicing = [59, 62, 65, 67]  # B-D-F-G (G7 second inversion)

        chord = determine_chord_from_voicing(voicing)
        assert chord == 10  # V7

        inversion = determine_inversion(voicing, chord)
        assert inversion == 2  # Second inversion (fifth in bass)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
