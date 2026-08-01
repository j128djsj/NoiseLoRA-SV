from torch.utils.data import DataLoader

from noiselora_sv.data import BalancedSpeakerSampler, VoxCelebTrainDataset, VoxCelebTrialsDataset


def build_train_loader(cfg):
    audio = cfg.get("audio", {})
    paths = cfg.get("paths", {})
    train_cfg = cfg.get("training", {})
    dataset = VoxCelebTrainDataset(
        train_root=paths.get("train_root", ""),
        train_list=paths.get("train_list", ""),
        sample_rate=audio.get("sample_rate", 16000),
        crop_seconds=audio.get("crop_seconds", 3.0),
        musan_train_root=paths.get("musan_train_root", ""),
        augment_prob=train_cfg.get("augment_prob", 1.0),
        snr_range=train_cfg.get("snr_range", (0.0, 20.0)),
        noise_types=train_cfg.get("noise_types", ["babble", "music", "noise"]),
        babble_sources=train_cfg.get("babble_sources", 3),
    )
    batch_size = int(train_cfg.get("batch_size", 300))
    samples_per_speaker = int(train_cfg.get("samples_per_speaker", 2))
    if len(dataset) == 0:
        raise RuntimeError("No training samples found. Set paths.train_root or paths.train_list in the config.")
    sampler = BalancedSpeakerSampler(dataset, batch_size, samples_per_speaker)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
    )


def build_eval_dataset(cfg, condition="clean", snr=None, noise_type=None):
    audio = cfg.get("audio", {})
    paths = cfg.get("paths", {})
    eval_cfg = cfg.get("evaluation", {})
    return VoxCelebTrialsDataset(
        test_root=paths.get("test_root", ""),
        trials_path=paths.get("trials", ""),
        sample_rate=audio.get("sample_rate", 16000),
        condition=condition,
        musan_test_root=paths.get("musan_test_root", ""),
        unseen_noise_root=paths.get("unseen_noise_root", ""),
        snr_db=snr,
        noise_type=noise_type,
        babble_sources=eval_cfg.get("babble_sources", 3),
        seed=eval_cfg.get("seed", 0),
    )


def build_eval_loader(cfg, condition="clean", snr=None, noise_type=None):
    eval_cfg = cfg.get("evaluation", {})
    if int(eval_cfg.get("batch_size", 1)) != 1:
        raise ValueError("evaluation.batch_size must be 1 for variable-length utterances")
    dataset = build_eval_dataset(cfg, condition=condition, snr=snr, noise_type=noise_type)
    if len(dataset) == 0:
        raise RuntimeError("No evaluation utterances found. Set paths.test_root and paths.trials in the config.")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 2)),
        pin_memory=True,
    )
    return loader, dataset
