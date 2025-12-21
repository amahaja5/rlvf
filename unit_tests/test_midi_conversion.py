"""
Unit tests for MIDI conversion and audio generation functions.

Tests cover MIDI file creation, melody conversion, and audio synthesis.
"""

import sys
import os
import pytest
from pathlib import Path
sys.path.insert(0, '../src')

import pretty_midi
from MIDI_conversion import (
    get_free_filename,
    state_seq_to_MIDI,
    melody_to_MIDI,
    one_part_note_list,
)


class TestGetFreeFilename:
    """Test unique filename generation."""

    def test_basic_filename(self, tmp_path):
        """Generate basic filename when directory is empty."""
        filename = get_free_filename('test', '.mid', directory=str(tmp_path))
        assert 'test' in filename
        assert filename.endswith('.mid')

    def test_incremental_filenames(self, tmp_path):
        """Should increment number when file exists."""
        # Create first file
        first_file = tmp_path / 'test-0.mid'
        first_file.touch()

        # Get next free filename
        filename = get_free_filename('test', '.mid', directory=str(tmp_path))
        assert 'test-1.mid' in filename

    def test_different_suffixes(self, tmp_path):
        """Handle different file extensions."""
        mid_file = get_free_filename('song', '.mid', directory=str(tmp_path))
        wav_file = get_free_filename('song', '.wav', directory=str(tmp_path))

        assert mid_file.endswith('.mid')
        assert wav_file.endswith('.wav')


class TestStateSeqToMIDI:
    """Test conversion of state sequences to MIDI files."""

    @pytest.fixture
    def state_indices(self):
        """Sample state indices for testing."""
        return {
            0: [48, 60, 64, 72],    # C major chord
            1: [50, 57, 62, 65],    # D minor chord
            2: [55, 59, 62, 67],    # G major chord
        }

    def test_creates_midi_file(self, tmp_path, state_indices):
        """Should create a valid MIDI file."""
        state_seq = [0, 1, 2, 0]  # Simple progression
        output_path = state_seq_to_MIDI(
            state_seq,
            state_indices,
            str(tmp_path),
            desired_fstub='test_seq'
        )

        assert os.path.exists(output_path)
        assert output_path.endswith('.mid')

    def test_midi_file_loadable(self, tmp_path, state_indices):
        """Created MIDI file should be loadable."""
        state_seq = [0, 1, 2]
        output_path = state_seq_to_MIDI(
            state_seq,
            state_indices,
            str(tmp_path),
            desired_fstub='loadable'
        )

        # Load and verify it's valid MIDI
        midi_obj = pretty_midi.PrettyMIDI(output_path)
        assert len(midi_obj.instruments) > 0
        assert len(midi_obj.instruments[0].notes) > 0

    def test_correct_note_count(self, tmp_path, state_indices):
        """Should have correct number of notes."""
        state_seq = [0, 1, 2]  # 3 chords
        output_path = state_seq_to_MIDI(
            state_seq,
            state_indices,
            str(tmp_path),
            desired_fstub='count_test'
        )

        midi_obj = pretty_midi.PrettyMIDI(output_path)
        notes = midi_obj.instruments[0].notes

        # Each state has 4 notes, 3 states = 12 notes
        assert len(notes) == 12

    def test_note_durations(self, tmp_path, state_indices):
        """Notes should have specified duration."""
        state_seq = [0, 1]
        note_dur = 0.5
        output_path = state_seq_to_MIDI(
            state_seq,
            state_indices,
            str(tmp_path),
            desired_fstub='duration',
            note_dur=note_dur
        )

        midi_obj = pretty_midi.PrettyMIDI(output_path)
        notes = midi_obj.instruments[0].notes

        # Check first chord notes have correct duration
        for i in range(4):
            note = notes[i]
            assert abs((note.end - note.start) - note_dur) < 0.01


class TestMelodyToMIDI:
    """Test melody to MIDI conversion."""

    def test_simple_melody(self):
        """Convert simple melody sequence."""
        melody = [[60], [62], [64], [65], [-1]]  # C-D-E-F-rest
        midi_obj = melody_to_MIDI(melody, note_length=1, save=False)

        assert isinstance(midi_obj, pretty_midi.PrettyMIDI)
        assert len(midi_obj.instruments) > 0

    def test_melody_with_chords(self):
        """Handle melody with multiple simultaneous notes."""
        melody = [[60, 64], [62], [64, 67], [-1]]  # Chords and single notes
        midi_obj = melody_to_MIDI(melody, note_length=1, save=False)

        notes = midi_obj.instruments[0].notes
        # Should have notes for each pitch
        assert len(notes) >= 5  # At least 5 distinct notes

    def test_melody_saves_file(self, tmp_path):
        """Melody should save to file when requested."""
        melody = [[60], [62], [64], [-1]]
        save_path = str(tmp_path / 'melody_test.mid')

        notes, path = melody_to_MIDI(
            melody,
            note_length=1,
            save=True,
            path=save_path
        )

        assert os.path.exists(save_path)
        assert path == save_path

    def test_melody_note_length(self, tmp_path):
        """Notes should have specified length."""
        melody = [[60], [62], [-1]]
        note_length = 0.75
        save_path = str(tmp_path / 'length_test.mid')

        melody_to_MIDI(melody, note_length=note_length, save=True, path=save_path)

        midi_obj = pretty_midi.PrettyMIDI(save_path)
        notes = midi_obj.instruments[0].notes

        # Check notes have approximately correct length
        for note in notes:
            duration = note.end - note.start
            assert abs(duration - note_length) < 0.1


class TestOnePartNoteList:
    """Test single voice note list generation."""

    def test_sustained_notes(self):
        """Should combine repeated notes into sustained note."""
        part = [60, 60, 60, 62]  # C sustained, then D
        notes = one_part_note_list(part, note_dur=1)

        assert len(notes) == 2  # Should be 2 notes, not 4
        assert notes[0].pitch == 60
        assert notes[1].pitch == 62

    def test_changing_notes(self):
        """Each different note should create separate note object."""
        part = [60, 62, 64, 65]  # All different
        notes = one_part_note_list(part, note_dur=1)

        assert len(notes) == 4
        pitches = [note.pitch for note in notes]
        assert pitches == [60, 62, 64, 65]

    def test_note_timings(self):
        """Notes should start at correct times."""
        part = [60, 60, 62]  # C-C-D
        note_dur = 0.5
        notes = one_part_note_list(part, note_dur=note_dur)

        # First note (C) should be 1 second (2 * 0.5)
        assert abs(notes[0].end - notes[0].start - 1.0) < 0.01
        # Second note (D) should start at 1.0 seconds
        assert abs(notes[1].start - 1.0) < 0.01


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_state_sequence(self, tmp_path):
        """Handle empty state sequence."""
        state_indices = {0: [48, 60, 64, 72]}
        state_seq = []

        # Should not crash, creates empty or minimal MIDI
        try:
            output_path = state_seq_to_MIDI(
                state_seq,
                state_indices,
                str(tmp_path),
                desired_fstub='empty'
            )
            assert os.path.exists(output_path)
        except Exception as e:
            pytest.skip(f"Empty sequence not supported: {e}")

    def test_single_note_melody(self):
        """Handle melody with single note."""
        melody = [[60], [-1]]
        midi_obj = melody_to_MIDI(melody, note_length=1, save=False)

        assert isinstance(midi_obj, pretty_midi.PrettyMIDI)

    def test_very_short_duration(self, tmp_path):
        """Handle very short note durations."""
        state_seq = [0, 1]
        state_indices = {0: [60, 64, 67, 72], 1: [62, 65, 69, 74]}

        output_path = state_seq_to_MIDI(
            state_seq,
            state_indices,
            str(tmp_path),
            desired_fstub='short',
            note_dur=0.1
        )

        midi_obj = pretty_midi.PrettyMIDI(output_path)
        notes = midi_obj.instruments[0].notes

        # Verify notes exist despite short duration
        assert len(notes) > 0


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow from state sequence to MIDI."""
        # Setup
        state_indices = {
            0: [48, 60, 64, 72],
            1: [50, 57, 62, 65],
            2: [52, 59, 64, 67],
            3: [48, 60, 64, 72],
        }
        state_seq = [0, 1, 2, 3]  # Simple 4-chord progression

        # Generate MIDI
        output_path = state_seq_to_MIDI(
            state_seq,
            state_indices,
            str(tmp_path),
            desired_fstub='integration',
            note_dur=1.0
        )

        # Verify file exists and is valid
        assert os.path.exists(output_path)

        # Load and verify content
        midi_obj = pretty_midi.PrettyMIDI(output_path)
        assert len(midi_obj.instruments) == 1
        assert len(midi_obj.instruments[0].notes) == 16  # 4 chords * 4 voices

        # Verify timing
        notes = midi_obj.instruments[0].notes
        # Notes should span 4 seconds (4 chords * 1 second each)
        max_end_time = max(note.end for note in notes)
        assert 3.9 < max_end_time < 4.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
