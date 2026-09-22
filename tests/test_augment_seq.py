import multiprocessing as mp

import pytest


def _run_single_interaction_augmentation(result_queue):
    from utils.augment_seq import augment_kt_seqs

    seq_len = 100
    q_seq = [0] * (seq_len - 1) + [1]
    s_seq = [0] * (seq_len - 1) + [1]
    r_seq = [-1] * (seq_len - 1) + [1]
    result = augment_kt_seqs(
        q_seq,
        s_seq,
        r_seq,
        mask_prob=0.0,
        crop_prob=0.3,
        permute_prob=0.5,
        replace_prob=0.0,
        negative_prob=0.0,
        easier_skills={},
        harder_skills={},
        q_mask_id=2,
        s_mask_id=2,
        seq_len=seq_len,
        seed=12405,
    )
    result_queue.put(len(result[0]))


def test_augmentation_terminates_for_single_interaction():
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_run_single_interaction_augmentation,
        args=(result_queue,),
    )
    process.start()
    process.join(timeout=2)

    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("augmentation did not terminate for a one-interaction sequence")

    assert process.exitcode == 0
    assert result_queue.get(timeout=1) == 100
