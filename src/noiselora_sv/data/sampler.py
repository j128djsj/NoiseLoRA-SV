import numpy as np
from torch.utils.data import Sampler


class BalancedSpeakerSampler(Sampler):
    def __init__(self, dataset, batch_size, samples_per_speaker, max_segments_per_speaker=500):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.samples_per_speaker = int(samples_per_speaker)
        self.max_segments_per_speaker = int(max_segments_per_speaker)
        self.epoch = 0
        self.utt_per_spk = dataset.utt_per_spk
        self.labels = list(self.utt_per_spk.keys())
        self.speakers_per_batch = max(1, self.batch_size // self.samples_per_speaker)
        self.num_batches = max(0, len(dataset) // self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.epoch)
        speakers = rng.permutation(self.labels)
        ptr = 0
        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.speakers_per_batch):
                if ptr >= len(speakers):
                    speakers = rng.permutation(self.labels)
                    ptr = 0
                spk = speakers[ptr]
                ptr += 1
                choices = self.utt_per_spk[spk][:self.max_segments_per_speaker]
                replace = len(choices) < self.samples_per_speaker
                batch.extend(rng.choice(choices, self.samples_per_speaker, replace=replace).tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
