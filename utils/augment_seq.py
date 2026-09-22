import random
import numpy as np
import math


def replace_only(
    q_seq,
    s_seq,
    r_seq,
    replace_prob,
    easier_skills,
    harder_skills,
    q_mask_id,
    s_mask_id,
    seq_len,
    seed=None,
):
    # Difficulty-aware skill replacement.
    rng = random.Random(seed)
    np.random.seed(seed)

    masked_q_seq = []
    masked_s_seq = []
    masked_r_seq = []
    replace_mask = []

    if replace_prob > 0:
        for i, elem in enumerate(zip(s_seq, r_seq)):
            s, r = elem
            prob = rng.random()
            if prob < replace_prob and s != 0 and s != s_mask_id:
                replace_mask.append(r)
                if (
                    r == 0 and s in harder_skills
                ):
                    masked_s_seq.append(harder_skills[s])
                elif (
                    r == 1 and s in easier_skills
                ):
                    masked_s_seq.append(easier_skills[s])
            else:
                masked_s_seq.append(s)
                replace_mask.append(-1)

    return masked_q_seq, masked_s_seq, masked_r_seq, replace_mask


def augment_kt_seqs(
    q_seq,
    s_seq,
    r_seq,
    mask_prob,
    crop_prob,
    permute_prob,
    replace_prob,
    negative_prob,
    easier_skills,
    harder_skills,
    q_mask_id,
    s_mask_id,
    seq_len,
    seed=None,
    skill_rel=None,
):
    # Create two augmented sequence views.
    rng = random.Random(seed)
    np.random.seed(seed)

    masked_q_seq = []
    masked_s_seq = []
    masked_r_seq = []
    negative_r_seq = []

    if mask_prob > 0:
        for q, s, r in zip(q_seq, s_seq, r_seq):
            prob = rng.random()
            if prob < mask_prob and s != 0:
                prob /= mask_prob
                if prob < 0.8:
                    masked_q_seq.append(q_mask_id)
                    masked_s_seq.append(s_mask_id)
                elif prob < 0.9:
                    masked_q_seq.append(
                        rng.randint(1, q_mask_id - 1)
                    )  # Replace with a random token, as in BERT masking.
                    masked_s_seq.append(
                        rng.randint(1, s_mask_id - 1)
                    )  # Both endpoints are inclusive.
                else:
                    masked_q_seq.append(q)
                    masked_s_seq.append(s)
            else:
                masked_q_seq.append(q)
                masked_s_seq.append(s)
            masked_r_seq.append(r)

            # Flip responses to construct hard negatives.
            neg_prob = rng.random()
            if neg_prob < negative_prob and r != -1:  # padding
                negative_r_seq.append(1 - r)
            else:
                negative_r_seq.append(r)
    else:
        masked_q_seq = q_seq[:]
        masked_s_seq = s_seq[:]
        masked_r_seq = r_seq[:]

        for r in r_seq:
            # Flip responses to construct hard negatives.
            neg_prob = rng.random()
            if neg_prob < negative_prob and r != -1:  # padding
                negative_r_seq.append(1 - r)
            else:
                negative_r_seq.append(r)

    # Replace skills using empirical difficulty neighbors.
    if replace_prob > 0:
        for i, elem in enumerate(zip(masked_s_seq, masked_r_seq)):
            s, r = elem
            prob = rng.random()
            if prob < replace_prob and s != 0 and s != s_mask_id:
                if (
                    r == 0 and s in harder_skills
                ):
                    masked_s_seq[i] = harder_skills[s]
                elif (
                    r == 1 and s in easier_skills
                ):
                    masked_s_seq[i] = easier_skills[s]

    true_seq_len = np.sum(np.asarray(q_seq) != 0)
    if permute_prob > 0:
        reorder_seq_len = math.floor(permute_prob * true_seq_len)
        if reorder_seq_len > 0:
            start_idx = (np.asarray(q_seq) != 0).argmax()
            start_pos = _sample_subsequence_start(
                rng, start_idx, reorder_seq_len, seq_len
            )

            # Permute one contiguous subsequence.
            perm = np.random.permutation(reorder_seq_len)
            masked_q_seq = (
                masked_q_seq[:start_pos]
                + np.asarray(masked_q_seq[start_pos : start_pos + reorder_seq_len])[
                    perm
                ].tolist()
                + masked_q_seq[start_pos + reorder_seq_len :]
            )
            masked_s_seq = (
                masked_s_seq[:start_pos]
                + np.asarray(masked_s_seq[start_pos : start_pos + reorder_seq_len])[
                    perm
                ].tolist()
                + masked_s_seq[start_pos + reorder_seq_len :]
            )
            masked_r_seq = (
                masked_r_seq[:start_pos]
                + np.asarray(masked_r_seq[start_pos : start_pos + reorder_seq_len])[
                    perm
                ].tolist()
                + masked_r_seq[start_pos + reorder_seq_len :]
            )

    # Crop one contiguous subsequence.
    if 0 < crop_prob < 1:
        crop_seq_len = math.floor(crop_prob * true_seq_len)
        if crop_seq_len == 0:
            crop_seq_len = 1
        start_idx = (np.asarray(q_seq) != 0).argmax()
        start_pos = _sample_subsequence_start(
            rng, start_idx, crop_seq_len, seq_len
        )

        masked_q_seq = masked_q_seq[start_pos : start_pos + crop_seq_len]
        masked_s_seq = masked_s_seq[start_pos : start_pos + crop_seq_len]
        masked_r_seq = masked_r_seq[start_pos : start_pos + crop_seq_len]

    pad_len = seq_len - len(masked_q_seq)

    attention_mask = [0] * pad_len + [1] * len(masked_s_seq)
    masked_q_seq = [0] * pad_len + masked_q_seq
    masked_s_seq = [0] * pad_len + masked_s_seq
    masked_r_seq = [-1] * pad_len + masked_r_seq

    return masked_q_seq, masked_s_seq, masked_r_seq, negative_r_seq, attention_mask


def _sample_subsequence_start(rng, start_idx, subsequence_len, seq_len):
    """Sample a valid inclusive start without retrying an impossible endpoint."""
    last_start = seq_len - subsequence_len
    if last_start < start_idx:
        return start_idx
    return rng.randint(start_idx, last_start)


def preprocess_qr(questions, responses, seq_len, pad_val=-1):
    """Split interactions into fixed-length sequences."""
    preprocessed_questions = []
    preprocessed_responses = []

    for q, r in zip(questions, responses):
        i = 0
        while i + seq_len < len(q):
            preprocessed_questions.append(q[i : i + seq_len])
            preprocessed_responses.append(r[i : i + seq_len])

            i += seq_len

        preprocessed_questions.append(
            np.concatenate([q[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_responses.append(
            np.concatenate([r[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )

    return preprocessed_questions, preprocessed_responses


def preprocess_qsr(questions, skills, responses, seq_len, pad_val=0):
    """Split question-skill-response data into fixed-length sequences."""
    preprocessed_questions = []
    preprocessed_skills = []
    preprocessed_responses = []
    attention_mask = []

    for q, s, r in zip(questions, skills, responses):
        i = 0
        while i + seq_len < len(q):
            preprocessed_questions.append(q[i : i + seq_len])
            preprocessed_skills.append(s[i : i + seq_len])
            preprocessed_responses.append(r[i : i + seq_len])
            attention_mask.append(np.ones(seq_len))
            i += seq_len

        preprocessed_questions.append(
            np.concatenate([q[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_skills.append(
            np.concatenate([s[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_responses.append(
            np.concatenate([r[i:], np.array([-1] * (i + seq_len - len(q)))])
        )
        attention_mask.append(
            np.concatenate(
                [np.ones_like(r[i:]), np.array([0] * (i + seq_len - len(q)))]
            )
        )

    return (
        preprocessed_questions,
        preprocessed_skills,
        preprocessed_responses,
        attention_mask,
    )
