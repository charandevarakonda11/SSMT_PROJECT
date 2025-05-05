import random
import numpy as np
import torch
import torch.utils.data

import commons
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence

import random
import numpy as np
import torch
import torch.utils.data
import librosa

import commons
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence

# class TextMelLoader(torch.utils.data.Dataset):
#     """
#     1) loads audio,text pairs
#     2) normalizes text and converts them to sequences of one-hot vectors
#     3) computes mel-spectrograms from audio files.
#     """

#     def __init__(self, audiopaths_and_text, hparams):
#         self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
#         self.text_cleaners = hparams.text_cleaners
#         self.max_wav_value = hparams.max_wav_value
#         self.sampling_rate = hparams.sampling_rate
#         self.load_mel_from_disk = hparams.load_mel_from_disk
#         self.add_noise = hparams.add_noise
#         self.symbols = hparams.punc + hparams.chars
#         self.add_blank = getattr(hparams, "add_blank", False)  # improved version
#         self.stft = commons.TacotronSTFT(
#             hparams.filter_length,
#             hparams.hop_length,
#             hparams.win_length,
#             hparams.n_mel_channels,
#             hparams.sampling_rate,
#             hparams.mel_fmin,
#             hparams.mel_fmax,
#         )
#         random.seed(1234)
#         random.shuffle(self.audiopaths_and_text)

#     def get_mel_text_pair(self, audiopath_and_text):
#         # separate filename and text
#         audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
#         text = self.get_text(text)
#         mel = self.get_mel(audiopath)
#         return (text, mel)

#     def get_mel(self, filename):
#         if not self.load_mel_from_disk:
#             audio, sampling_rate = load_wav_to_torch(filename)
#             if sampling_rate != self.stft.sampling_rate:
#                 raise ValueError(
#                     "{} {} SR doesn't match target {} SR".format(
#                         sampling_rate, self.stft.sampling_rate
#                     )
#                 )
#             if self.add_noise:
#                 audio = audio + torch.rand_like(audio)
#             audio_norm = audio / self.max_wav_value
#             audio_norm = audio_norm.unsqueeze(0)
#             melspec = self.stft.mel_spectrogram(audio_norm)
#             melspec = torch.squeeze(melspec, 0)
#         else:
#             melspec = torch.from_numpy(np.load(filename))
#             assert (
#                 melspec.size(0) == self.stft.n_mel_channels
#             ), "Mel dimension mismatch: given {}, expected {}".format(
#                 melspec.size(0), self.stft.n_mel_channels
#             )

#         return melspec

#     def get_text(self, text):
#         text_norm = text_to_sequence(text, self.symbols, self.text_cleaners)
#         if self.add_blank:
#             text_norm = commons.intersperse(
#                 text_norm, len(self.symbols)
#             )  # add a blank token, whose id number is len(symbols)
#         text_norm = torch.IntTensor(text_norm)
#         return text_norm
    
#     def _load_pitch(self, audiopath):
#         # Implement your pitch loading/extraction logic here
#         # This is a placeholder - replace with your actual code
#         # Example: load from a .pitch file or extract using a library
#         pitch = np.load(audiopath.replace(".wav", ".pitch.npy"))
#         return pitch

#     def _load_energy(self, audiopath):
#         # Implement your energy loading/extraction logic here
#         # This is a placeholder - replace with your actual code
#         # Example: calculate RMS energy from the audio
#         audio, sr = load_wav_to_torch(audiopath)
#         energy = torch.sqrt(torch.mean(audio**2))
#         return energy.item()

#     def __getitem__(self, index):
#         audiopath, text = self.audiopaths_and_text[index]
#         text = self.get_text(text)
#         mel = self.get_mel(audiopath)
#         pitch = self._load_pitch(audiopath)
#         energy = self._load_energy(audiopath)
#         return text, mel, pitch, energy # Changed the return order to match train.py


#     # def __getitem__(self, index):
#     #     return self.get_mel_text_pair(self.audiopaths_and_text[index])

#     def __len__(self):
#         return len(self.audiopaths_and_text)


# import torch.nn.functional as F

# class TextMelCollate:
#     """Zero-pads model inputs and targets based on number of frames per step"""

#     def __init__(self, n_frames_per_step=1):
#         self.n_frames_per_step = n_frames_per_step

#     def __call__(self, batch):
#         """Collate's training batch from normalized text and mel-spectrogram
#         PARAMS
#         ------
#         batch: [text_normalized, mel_normalized, pitch, energy]
#         """
#         # Right zero-pad all one-hot text sequences to max input length
#         input_lengths, ids_sorted_decreasing = torch.sort(
#             torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
#         )
#         max_input_len = input_lengths[0]

#         text_padded = torch.LongTensor(len(batch), max_input_len)
#         text_padded.zero_()
#         for i in range(len(ids_sorted_decreasing)):
#             text = batch[ids_sorted_decreasing[i]][0]
#             text_padded[i, : text.size(0)] = text

#         # Right zero-pad mel-spec
#         num_mels = batch[0][1].size(0)
#         max_target_len = max([x[1].size(1) for x in batch])
#         if max_target_len % self.n_frames_per_step != 0:
#             max_target_len += (
#                 self.n_frames_per_step - max_target_len % self.n_frames_per_step
#             )
#             assert max_target_len % self.n_frames_per_step == 0

#         # include mel padded
#         mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
#         mel_padded.zero_()
#         output_lengths = torch.LongTensor(len(batch))
#         for i in range(len(ids_sorted_decreasing)):
#             mel = batch[ids_sorted_decreasing[i]][1]
#             mel_padded[i, :, : mel.size(1)] = mel
#             output_lengths[i] = mel.size(1)

#         # Collate pitch
#         pitch_padded = torch.FloatTensor(len(batch), max_target_len)
#         pitch_padded.zero_()
#         for i in range(len(ids_sorted_decreasing)):
#             pitch = batch[ids_sorted_decreasing[i]][2]
#             pitch_padded[i, : pitch.size(0)] = torch.tensor(pitch) # Ensure it's a tensor

#         # Collate energy
#         energy_padded = torch.FloatTensor(len(batch), max_target_len)
#         energy_padded.zero_()
#         for i in range(len(ids_sorted_decreasing)):
#             energy = batch[ids_sorted_decreasing[i]][3]
#             energy_padded[i, : pitch.size(0)] = torch.tensor([energy] * pitch.size(0)) # Repeat energy to match mel length

#         return text_padded, input_lengths, mel_padded, output_lengths, pitch_padded, energy_padded


class TextMelLoader(torch.utils.data.Dataset):
    """
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files and extracts pitch and energy.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.symbols = hparams.punc + hparams.chars
        self.add_blank = getattr(hparams, "add_blank", False)
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.n_mel_channels = hparams.n_mel_channels
        self.mel_fmin = hparams.mel_fmin
        self.mel_fmax = hparams.mel_fmax
        self.stft = commons.TacotronSTFT(
            self.filter_length,
            self.hop_length,
            self.win_length,
            self.n_mel_channels,
            self.sampling_rate,
            self.mel_fmin,
            self.mel_fmax,
        )
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError(
                    "{} {} SR doesn't match target {} SR".format(
                        sampling_rate, self.stft.sampling_rate
                    )
                )
            if self.add_noise:
                audio = audio + torch.rand_like(audio)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert (
                melspec.size(0) == self.stft.n_mel_channels
            ), "Mel dimension mismatch: given {}, expected {}".format(
                melspec.size(0), self.stft.n_mel_channels
            )
        return melspec

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.symbols, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(
                text_norm, len(self.symbols)
            )
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def _extract_pitch(self, audiopath):
        audio, sr = librosa.load(audiopath, sr=self.sampling_rate)
        yin_output = librosa.yin(
            audio,
            fmin=librosa.note_to_hz('C0'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )
        print(f"Shape of yin_output in _extract_pitch: {np.array(yin_output).shape}") # ADD THIS LINE
        print(f"Type of yin_output in _extract_pitch: {type(yin_output)}") # ADD THIS LINE

        if isinstance(yin_output, tuple):
            if len(yin_output) == 4:
                f0, voiced_flag, voiced_prob, _ = yin_output
            elif len(yin_output) == 5:  # Adjust if you see 5
                f0, voiced_flag, voiced_prob, acorr, _ = yin_output
            else:
                raise ValueError(f"Unexpected number of return values from librosa.yin: {len(yin_output)}")
        else:
            f0 = yin_output
            voiced_flag = np.ones_like(f0, dtype=bool) # Assume all are voiced if only f0 is returned
            voiced_prob = np.ones_like(f0, dtype=float)

        pitch_tensor = torch.from_numpy(f0).float()
        pitch_tensor[torch.isnan(pitch_tensor)] = 0.0
        print(f"Shape of pitch_tensor in _extract_pitch: {pitch_tensor.shape}") # ADD THIS LINE
        return pitch_tensor

    def _extract_energy(self, audiopath):
        audio, sr = librosa.load(audiopath, sr=self.sampling_rate)
        # Calculate RMS energy per frame
        energy = librosa.feature.rms(y=audio, frame_length=self.win_length, hop_length=self.hop_length)[0]
        energy_tensor = torch.from_numpy(energy).float()
        return energy_tensor

    # def __getitem__(self, index):
    #     audiopath, text = self.audiopaths_and_text[index]
    #     text = self.get_text(text)
    #     mel = self.get_mel(audiopath)
    #     pitch = self._extract_pitch(audiopath)
    #     energy = self._extract_energy(audiopath)
    #     return text, mel, pitch, energy
    # def __getitem__(self, index):
    #     audiopath, text = self.audiopaths_and_text[index]
    #     text = self.get_text(text)
    #     mel = self.get_mel(audiopath)
    #     pitch = self._extract_pitch(audiopath).unsqueeze(0)  # Add channel dimension here
    #     energy = self._extract_energy(audiopath).unsqueeze(0) # Add channel dimension here
    #     return text, mel, pitch, energy

    # def __getitem__(self, index):
    #     audiopath, text = self.audiopaths_and_text[index]
    #     text = self.get_text(text)
    #     mel = self.get_mel(audiopath)
    #     pitch = self._extract_pitch(audiopath)
    #     energy = self._extract_energy(audiopath)
    #     print(f"Shape of pitch in __getitem__: {pitch.shape}") # ADD THIS LINE
    #     print(f"Shape of energy in __getitem__: {energy.shape}") # ADD THIS LINE
    #     return text, mel, pitch.unsqueeze(0), energy.unsqueeze(0) # Keep the unsqueeze here

    def __getitem__(self, index):
        audiopath, text = self.audiopaths_and_text[index]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        pitch = self._extract_pitch(audiopath)
        energy = self._extract_energy(audiopath)

        # Ensure pitch has the same length as mel (time dimension)
        if pitch.shape[-1] > mel.shape[-1]:
            pitch = pitch[:mel.shape[-1]]
        elif pitch.shape[-1] < mel.shape[-1]:
            # Pad with zeros if pitch is shorter (you might need a more sophisticated padding strategy)
            padding = torch.zeros(mel.shape[-1] - pitch.shape[-1])
            pitch = torch.cat((pitch, padding), dim=-1)

        energy = energy[:mel.shape[-1]] # Ensure energy also matches mel length

        return text, mel, pitch.unsqueeze(0), energy.unsqueeze(0)


    def __len__(self):
        return len(self.audiopaths_and_text)

class TextMelCollate:
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, pitch, energy]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        # Pad pitch
        pitch_padded = torch.FloatTensor(len(batch), 1, max_target_len) # Expecting [B, 1, T_mel]
        pitch_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][2] # Shape [1, T_pitch]
            if pitch.shape[-1] > max_target_len:
                pitch = pitch[:, :max_target_len]
            pitch_padded[i, :, : pitch.size(-1)] = pitch

        # Pad energy
        energy_padded = torch.FloatTensor(len(batch), 1, max_target_len) # Expecting [B, 1, T_mel]
        energy_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            energy = batch[ids_sorted_decreasing[i]][3] # Shape [1, T_energy]
            if energy.shape[-1] > max_target_len:
                energy = energy[:, :max_target_len]
            energy_padded[i, :, : energy.size(-1)] = energy

        return text_padded, input_lengths, mel_padded, output_lengths, pitch_padded, energy_padded
# class TextMelCollate:
#     """Zero-pads model inputs and targets based on number of frames per step"""

#     def __init__(self, n_frames_per_step=1):
#         self.n_frames_per_step = n_frames_per_step

#     def __call__(self, batch):
#         """Collate's training batch from normalized text and mel-spectrogram
#         PARAMS
#         ------
#         batch: [text_normalized, mel_normalized, pitch, energy]
#         """
#         # Right zero-pad all one-hot text sequences to max input length
#         input_lengths, ids_sorted_decreasing = torch.sort(
#             torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
#         )
#         max_input_len = input_lengths[0]

#         text_padded = torch.LongTensor(len(batch), max_input_len)
#         text_padded.zero_()
#         for i in range(len(ids_sorted_decreasing)):
#             text = batch[ids_sorted_decreasing[i]][0]
#             text_padded[i, : text.size(0)] = text

#         # Right zero-pad mel-spec
#         num_mels = batch[0][1].size(0)
#         max_target_len = max([x[1].size(1) for x in batch])
#         if max_target_len % self.n_frames_per_step != 0:
#             max_target_len += (
#                 self.n_frames_per_step - max_target_len % self.n_frames_per_step
#             )
#             assert max_target_len % self.n_frames_per_step == 0

#         # include mel padded
#         mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
#         mel_padded.zero_()
#         output_lengths = torch.LongTensor(len(batch))
#         for i in range(len(ids_sorted_decreasing)):
#             mel = batch[ids_sorted_decreasing[i]][1]
#             mel_padded[i, :, : mel.size(1)] = mel
#             output_lengths[i] = mel.size(1)

#         # Pad pitch
#         pitch_padded = torch.FloatTensor(len(batch), max_target_len)
#         pitch_padded.zero_()
#         for i in range(len(ids_sorted_decreasing)):
#             pitch = batch[ids_sorted_decreasing[i]][2]
#             pitch_padded[i, : pitch.size(0)] = pitch

#         # Pad energy
#         energy_padded = torch.FloatTensor(len(batch), max_target_len)
#         energy_padded.zero_()
#         for i in range(len(ids_sorted_decreasing)):
#             energy = batch[ids_sorted_decreasing[i]][3]
#             energy_padded[i, : energy.size(0)] = energy

#         return text_padded, input_lengths, mel_padded, output_lengths, pitch_padded, energy_padded

"""Multi speaker version"""


class TextMelSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.add_noise = hparams.add_noise
        self.symbols = hparams.punc + hparams.chars
        self.add_blank = getattr(hparams, "add_blank", False)  # improved version
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        self.stft = commons.TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            hparams.n_mel_channels,
            hparams.sampling_rate,
            hparams.mel_fmin,
            hparams.mel_fmax,
        )

        self._filter_text_len()
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

    def _filter_text_len(self):
        audiopaths_sid_text_new = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
        self.audiopaths_sid_text = audiopaths_sid_text_new

    def get_mel_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = (
            audiopath_sid_text[0],
            audiopath_sid_text[1],
            audiopath_sid_text[2],
        )
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        sid = self.get_sid(sid)
        return (text, mel, sid)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError(
                    "{} {} SR doesn't match target {} SR".format(
                        sampling_rate, self.stft.sampling_rate
                    )
                )
            if self.add_noise:
                audio = audio + torch.rand_like(audio)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert (
                melspec.size(0) == self.stft.n_mel_channels
            ), "Mel dimension mismatch: given {}, expected {}".format(
                melspec.size(0), self.stft.n_mel_channels
            )

        return melspec

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.symbols, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(
                text_norm, len(self.symbols)
            )  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.IntTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_mel_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextMelSpeakerCollate:
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded & sid
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            output_lengths[i] = mel.size(1)
            sid[i] = batch[ids_sorted_decreasing[i]][2]

        return text_padded, input_lengths, mel_padded, output_lengths, sid
