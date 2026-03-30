# Copyright © 2023 Apple Inc.

import zlib
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def detect_language(
    model: "Whisper", mel: mx.array, tokenizer: Tokenizer = None
) -> Tuple[mx.array, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : mx.array, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages
        )
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = mel[None]

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = mx.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = np.full(logits.shape[-1], -np.inf, dtype=np.float32)
    mask[list(tokenizer.all_language_tokens)] = 0.0
    logits += mx.array(mask)
    language_tokens = mx.argmax(logits, axis=-1)
    language_token_probs = mx.softmax(logits, axis=-1)
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details
    fp16: bool = True  # use fp16 for most of the calculation


@dataclass(frozen=True)
class DecodingResult:
    audio_features: Optional[mx.array] = None
    language: str = ""
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    _compression_ratio: float = np.nan
    token_probs: Optional[np.ndarray] = None

    @property
    def compression_ratio(self) -> float:
        """Lazy computation — zlib.compress only runs when actually accessed."""
        if np.isnan(self._compression_ratio) and self.text:
            ratio = compression_ratio(self.text)
            object.__setattr__(self, '_compression_ratio', ratio)
        return self._compression_ratio


class Inference:
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = None

    def logits(self, tokens: mx.array, audio_features: mx.array) -> mx.array:
        """Perform a forward pass on the decoder and return per-token logits"""
        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        # return_cross_qk=False (default) skips cross-attention QK collection
        logits, self.kv_cache, _ = self.model.decoder(
            tokens, audio_features, kv_cache=self.kv_cache
        )
        return logits.astype(mx.float32)

    def rearrange_kv_cache(self, source_indices):
        """Update the key-value cache according to the updated beams"""
        # After batch compaction the new active rows may still be [0..N-1] while
        # the cached tensors remain larger, e.g. dropping only the last row from
        # a 9-item batch leaves source_indices == [0..7]. Always apply the row
        # selection whenever a cache exists so cache batch size stays aligned.
        if self.kv_cache is not None:
            idx = mx.array(source_indices, dtype=mx.int32)
            self.kv_cache = tree_map(lambda x: x[idx], self.kv_cache)

    def reset(self):
        self.kv_cache = None


class SequenceRanker:
    def rank(
        self, tokens: List[List[mx.array]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[List[int]]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> Tuple[mx.array, bool, mx.array]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : mx.array, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : mx.array, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : mx.array, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : mx.array, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        sum_logprobs: mx.array, shape = (n_batch)
            updated cumulative log probabilities for each sequence

        """
        raise NotImplementedError

    def finalize(
        self, tokens: mx.array, sum_logprobs: mx.array
    ) -> Tuple[Sequence[Sequence[mx.array]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : mx.array, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : mx.array, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[mx.array]], length = n_audio
            sequence of mx.arrays containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot
        self._step_probs = []
        self._batch_indices = None
        self.last_step_probs = None

    def reset(self):
        self._step_probs = []
        self._batch_indices = None
        self.last_step_probs = None

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> Tuple[mx.array, bool, mx.array]:
        if self.temperature == 0:
            next_tokens = logits.argmax(axis=-1)
        else:
            next_tokens = mx.random.categorical(logits=logits / self.temperature)

        # logits already float32 from Inference.logits() — skip redundant cast
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Cache batch indices to avoid per-step mx.arange allocation
        n = logprobs.shape[0]
        if self._batch_indices is None or self._batch_indices.shape[0] != n:
            self._batch_indices = mx.arange(n)

        current_logprobs = logprobs[self._batch_indices, next_tokens]
        self.last_step_probs = mx.exp(current_logprobs)
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        eot_mask = tokens[:, -1] == self.eot
        next_tokens = next_tokens * (1 - eot_mask) + self.eot * eot_mask
        tokens = mx.concatenate([tokens, next_tokens[:, None]], axis=-1)

        completed = mx.all(tokens[:, -1] == self.eot)
        return tokens, completed, sum_logprobs

    def finalize(self, tokens: mx.array, sum_logprobs: mx.array):
        # make sure each sequence has at least one EOT token at the end
        tokens = mx.pad(tokens, [(0, 0), (0, 0), (0, 1)], constant_values=self.eot)
        return tokens, sum_logprobs.tolist()


class LogitFilter:
    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Apply any filtering or masking to logits

        Parameters
        ----------
        logits : mx.array, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : mx.array, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int, n_vocab: int):
        self.sample_begin = sample_begin
        mask = np.zeros(n_vocab, np.float32)
        mask[tokenizer.encode(" ") + [tokenizer.eot]] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        if tokens.shape[1] == self.sample_begin:
            return logits + self.mask
        return logits


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int], n_vocab: int):
        mask = np.zeros(n_vocab, np.float32)
        mask[list(suppress_tokens)] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        return logits + self.mask


class ApplyTimestampRules(LogitFilter):
    def __init__(
        self,
        tokenizer: Tokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index
        # Cache frequently accessed values
        self._ts_begin = tokenizer.timestamp_begin
        self._eot = tokenizer.eot
        self._no_timestamps = tokenizer.no_timestamps
        # Pre-allocate mask buffer (resized on first call)
        self._mask_buf = None
        # Lazy-initialized mask for GPU-only force-timestamp (see apply_cpu)
        self._text_positions = None

    def _get_mask_buf(self, n_batch: int, n_vocab: int) -> np.ndarray:
        """Reuse a pre-allocated buffer instead of np.zeros() every step."""
        if self._mask_buf is None or self._mask_buf.shape != (n_batch, n_vocab):
            self._mask_buf = np.zeros((n_batch, n_vocab), np.float32)
        else:
            self._mask_buf[:] = 0.0
        return self._mask_buf

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        n_batch = tokens.shape[0]

        # Fast path for single-item batches (most common case).
        # Avoids numpy batch operations and 2D mask allocation.
        if n_batch == 1:
            return self._apply_single(logits, tokens)

        return self._apply_batch(logits, tokens, n_batch)

    def _apply_single(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Optimized path for batch_size=1: no batch numpy, minimal allocation."""
        ts_begin = self._ts_begin
        eot = self._eot
        n_vocab = logits.shape[-1]

        mask = self._get_mask_buf(1, n_vocab)

        if self._no_timestamps is not None:
            mask[0, self._no_timestamps] = -np.inf

        seq = tokens[0, self.sample_begin:]
        seq_np = np.array(seq)
        seq_len = seq_np.shape[0]

        if seq_len >= 1:
            last = int(seq_np[-1])
            last_is_ts = last >= ts_begin
            penult_is_ts = True if seq_len < 2 else int(seq_np[-2]) >= ts_begin

            if last_is_ts:
                if penult_is_ts:
                    mask[0, ts_begin:] = -np.inf
                else:
                    mask[0, :eot] = -np.inf

            # Find last timestamp position for monotonicity
            ts_positions = np.where(seq_np >= ts_begin)[0]
            if len(ts_positions) > 0:
                last_timestamp = int(ts_positions[-1])
                if not last_timestamp or penult_is_ts:
                    last_timestamp += 1
                if last_timestamp > 0:
                    mask[0, ts_begin:ts_begin + last_timestamp] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            mask[0, :ts_begin] = -np.inf
            if self.max_initial_timestamp_index is not None:
                last_allowed = ts_begin + self.max_initial_timestamp_index
                mask[0, last_allowed + 1:] = -np.inf

        # Force timestamp if timestamp prob > max text prob
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        ts_logprob = mx.logsumexp(logprobs[0, ts_begin:], axis=-1)
        max_text_logprob = mx.max(logprobs[0, :ts_begin], axis=-1)
        if (ts_logprob > max_text_logprob).item():
            mask[0, :ts_begin] = -np.inf

        return logits + mx.array(mask, logits.dtype)

    def apply_cpu(self, logits: mx.array, cpu_sampled: list) -> mx.array:
        """Optimized single-batch path using CPU token list.

        Avoids GPU→CPU transfer for token sequence and eliminates the GPU sync
        in the force-timestamp check by computing it entirely on GPU.

        Parameters
        ----------
        logits : mx.array, shape = (1, vocab_size)
        cpu_sampled : list of int — decoded tokens after sample_begin (CPU-side)
        """
        ts_begin = self._ts_begin
        eot = self._eot
        n_vocab = logits.shape[-1]

        mask = self._get_mask_buf(1, n_vocab)

        if self._no_timestamps is not None:
            mask[0, self._no_timestamps] = -np.inf

        seq_len = len(cpu_sampled)

        if seq_len >= 1:
            last = cpu_sampled[-1]
            last_is_ts = last >= ts_begin
            penult_is_ts = True if seq_len < 2 else cpu_sampled[-2] >= ts_begin

            if last_is_ts:
                if penult_is_ts:
                    mask[0, ts_begin:] = -np.inf
                else:
                    mask[0, :eot] = -np.inf

            # Enforce monotonically increasing timestamps
            ts_positions = [i for i, t in enumerate(cpu_sampled) if t >= ts_begin]
            if ts_positions:
                last_timestamp = ts_positions[-1]
                if not last_timestamp or penult_is_ts:
                    last_timestamp += 1
                if last_timestamp > 0:
                    mask[0, ts_begin:ts_begin + last_timestamp] = -np.inf

        if seq_len == 0:
            # First sample step — constrain to timestamps only
            mask[0, :ts_begin] = -np.inf
            if self.max_initial_timestamp_index is not None:
                last_allowed = ts_begin + self.max_initial_timestamp_index
                mask[0, last_allowed + 1:] = -np.inf

        mask_mx = mx.array(mask, logits.dtype)

        # Force-timestamp: if P(timestamp) > max P(text_token), suppress all text.
        # Computed entirely on GPU — no .item() sync.
        if seq_len > 0:
            # Lazy-init boolean mask for text token positions
            if self._text_positions is None or self._text_positions.shape[0] != n_vocab:
                tp = np.zeros(n_vocab, dtype=np.float32)
                tp[:ts_begin] = 1.0
                self._text_positions = mx.array(tp, logits.dtype)

            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            ts_logprob = mx.logsumexp(logprobs[0, ts_begin:], axis=-1)
            max_text_logprob = mx.max(logprobs[0, :ts_begin], axis=-1)
            should_force = ts_logprob > max_text_logprob
            # -1e9 instead of -inf to avoid NaN from -inf * 0
            force_val = mx.where(should_force, mx.array(-1e9, logits.dtype), mx.array(0.0, logits.dtype))
            return logits + mask_mx + force_val * self._text_positions

        return logits + mask_mx

    def apply_cpu_batch(self, logits: mx.array, cpu_sampled_np: np.ndarray) -> mx.array:
        """Optimized batch path using CPU token matrix.

        Like apply_cpu() for batch=1, this avoids GPU→CPU transfer for the token
        sequence and computes force-timestamp entirely on GPU.

        Parameters
        ----------
        logits : mx.array, shape = (n_batch, vocab_size)
        cpu_sampled_np : np.ndarray, shape = (n_batch, n_sampled) — CPU token mirror
        """
        n_batch = cpu_sampled_np.shape[0]
        n_vocab = logits.shape[-1]
        ts_begin = self._ts_begin
        eot = self._eot

        mask = self._get_mask_buf(n_batch, n_vocab)

        if self._no_timestamps is not None:
            mask[:, self._no_timestamps] = -np.inf

        seq_len = cpu_sampled_np.shape[1]

        if seq_len >= 1:
            last_tok = cpu_sampled_np[:, -1]
            last_is_ts = last_tok >= ts_begin

            penult_is_ts = np.ones(n_batch, dtype=bool)
            if seq_len >= 2:
                penult_is_ts = cpu_sampled_np[:, -2] >= ts_begin

            both_ts = last_is_ts & penult_is_ts
            for k in np.where(both_ts)[0]:
                mask[k, ts_begin:] = -np.inf

            ts_after_text = last_is_ts & ~penult_is_ts
            for k in np.where(ts_after_text)[0]:
                mask[k, :eot] = -np.inf

            ts_mask = cpu_sampled_np >= ts_begin
            ts_indices = np.where(ts_mask, np.arange(seq_len)[None, :], -1)
            has_any_ts = ts_mask.any(axis=1)

            if has_any_ts.any():
                last_ts_pos = ts_indices.max(axis=1)
                for k in np.where(has_any_ts)[0]:
                    pos = last_ts_pos[k]
                    last_timestamp = pos
                    if not last_timestamp or penult_is_ts[k]:
                        last_timestamp += 1
                    if last_timestamp > 0:
                        mask[k, ts_begin:ts_begin + last_timestamp] = -np.inf

        if seq_len == 0:
            mask[:, :ts_begin] = -np.inf
            if self.max_initial_timestamp_index is not None:
                last_allowed = ts_begin + self.max_initial_timestamp_index
                mask[:, last_allowed + 1:] = -np.inf

        mask_mx = mx.array(mask, logits.dtype)

        # Force-timestamp: GPU-only (no sync). Replaces np.array(ts > text).
        if seq_len > 0:
            if self._text_positions is None or self._text_positions.shape[0] != n_vocab:
                tp = np.zeros(n_vocab, dtype=np.float32)
                tp[:ts_begin] = 1.0
                self._text_positions = mx.array(tp, logits.dtype)

            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            ts_logprobs = mx.logsumexp(logprobs[:, ts_begin:], axis=-1)
            max_text_logprobs = mx.max(logprobs[:, :ts_begin], axis=-1)
            should_force = ts_logprobs > max_text_logprobs  # (n_batch,)
            force_vals = mx.where(should_force, mx.array(-1e9, logits.dtype), mx.array(0.0, logits.dtype))
            force_contribution = force_vals[:, None] * self._text_positions[None, :]
            return logits + mask_mx + force_contribution

        return logits + mask_mx

    def _apply_batch(self, logits: mx.array, tokens: mx.array, n_batch: int) -> mx.array:
        """Fallback batch path when no CPU token mirror is available."""
        n_vocab = logits.shape[-1]
        ts_begin = self._ts_begin
        eot = self._eot

        mask = self._get_mask_buf(n_batch, n_vocab)

        if self._no_timestamps is not None:
            mask[:, self._no_timestamps] = -np.inf

        sampled_np = np.array(tokens[:, self.sample_begin:])
        seq_len = sampled_np.shape[1]

        if seq_len >= 1:
            last_tok = sampled_np[:, -1]
            last_is_ts = last_tok >= ts_begin

            penult_is_ts = np.ones(n_batch, dtype=bool)
            if seq_len >= 2:
                penult_is_ts = sampled_np[:, -2] >= ts_begin

            both_ts = last_is_ts & penult_is_ts
            for k in np.where(both_ts)[0]:
                mask[k, ts_begin:] = -np.inf

            ts_after_text = last_is_ts & ~penult_is_ts
            for k in np.where(ts_after_text)[0]:
                mask[k, :eot] = -np.inf

            ts_mask = sampled_np >= ts_begin
            ts_indices = np.where(ts_mask, np.arange(seq_len)[None, :], -1)
            has_any_ts = ts_mask.any(axis=1)

            if has_any_ts.any():
                last_ts_pos = ts_indices.max(axis=1)
                for k in np.where(has_any_ts)[0]:
                    pos = last_ts_pos[k]
                    last_timestamp = pos
                    if not last_timestamp or penult_is_ts[k]:
                        last_timestamp += 1
                    if last_timestamp > 0:
                        mask[k, ts_begin:ts_begin + last_timestamp] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            mask[:, :ts_begin] = -np.inf
            if self.max_initial_timestamp_index is not None:
                last_allowed = ts_begin + self.max_initial_timestamp_index
                mask[:, last_allowed + 1:] = -np.inf

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        ts_logprobs = mx.logsumexp(logprobs[:, ts_begin:], axis=-1)
        max_text_logprobs = mx.max(logprobs[:, :ts_begin], axis=-1)
        force_ts = np.array(ts_logprobs > max_text_logprobs)

        if force_ts.any():
            for k in np.where(force_ts)[0]:
                mask[k, :ts_begin] = -np.inf

        return logits + mx.array(mask, logits.dtype)


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=options.task,
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = Inference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            raise NotImplementedError("Beam search decoder is not yet implemented")
            # self.decoder = BeamSearchDecoder(
            #    options.beam_size, tokenizer.eot, self.inference, options.patience
            # )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # Pre-compute a single fused suppression mask (SuppressBlank + SuppressTokens).
        # These are constant across all decode steps, so one mx.array addition per step
        # instead of two separate filter calls with their own mx.array additions.
        n_vocab = model.dims.n_vocab
        fused_suppress = np.zeros(n_vocab, np.float32)
        if self.options.suppress_tokens:
            for t in self._get_suppress_tokens():
                fused_suppress[t] = -np.inf
        self._fused_suppress_mask = mx.array(fused_suppress)

        # SuppressBlank only fires on the very first sample step
        if self.options.suppress_blank:
            blank_mask = np.zeros(n_vocab, np.float32)
            blank_mask[self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf
            self._blank_mask = mx.array(blank_mask)
        else:
            self._blank_mask = None

        # Timestamp filter is the only stateful per-step filter
        self._timestamp_filter = None
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self._timestamp_filter = ApplyTimestampRules(
                tokenizer, self.sample_begin, max_initial_timestamp_index
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: mx.array):
        if self.options.fp16:
            mel = mel.astype(mx.float16)

        if mel.shape[-2:] == (
            self.model.dims.n_audio_ctx,
            self.model.dims.n_audio_state,
        ):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            audio_features = self.model.encoder(mel)

        if audio_features.dtype != (mx.float16 if self.options.fp16 else mx.float32):
            raise TypeError(
                f"audio_features has an incorrect dtype: {audio_features.dtype}"
            )

        return audio_features

    def _detect_language(self, audio_features: mx.array, tokens: np.array):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                # write language tokens
                tokens[:, self.sot_index + 1] = np.array(lang_tokens)

        return languages, lang_probs

    def _main_loop(self, audio_features: mx.array, tokens: mx.array):
        n_batch = tokens.shape[0]
        sum_logprobs: mx.array = mx.zeros(n_batch)
        no_speech_probs_mx = None  # deferred — avoid GPU sync on step 0
        active_indices = np.arange(n_batch, dtype=np.int32)
        finalized_tokens = [None] * n_batch
        finalized_sum_logprobs = np.zeros(n_batch, dtype=np.float32)
        # Pre-allocated buffer for token probabilities.
        # Avoids per-step Python for-loop + list.append overhead.
        token_prob_buf = np.zeros((n_batch, sample_len), dtype=np.float32)
        token_prob_len = np.zeros(n_batch, dtype=np.int32)

        # Local references to avoid attribute lookups in the hot loop
        inference_logits = self.inference.logits
        inference_rearrange = self.inference.rearrange_kv_cache
        decoder_update = self.decoder.update
        decoder_last_step_probs = lambda: getattr(self.decoder, "last_step_probs", None)
        suppress_mask = self._fused_suppress_mask
        blank_mask = self._blank_mask
        ts_filter = self._timestamp_filter
        sample_begin = self.sample_begin
        sample_len = self.sample_len
        n_ctx = self.n_ctx
        sot_index = self.sot_index
        no_speech_token = self.tokenizer.no_speech
        eot_id = self.tokenizer.eot
        is_single = n_batch == 1

        # CPU token mirror for ALL batch sizes.
        # Replaces GPU→CPU transfer of the entire (n_batch × seq_len) token matrix
        # in timestamp rules with a single n_batch-int transfer per step.
        # Also provides CPU-side completion check (avoids mx.all GPU sync).
        if ts_filter is not None:
            if is_single:
                cpu_sampled = []          # Python list for batch=1
                cpu_sampled_buf = None
            else:
                cpu_sampled = None
                cpu_sampled_buf = np.empty((n_batch, sample_len), dtype=np.int32)
        else:
            cpu_sampled = None
            cpu_sampled_buf = None
        n_sampled = 0

        try:
            for i in range(sample_len):
                logits = inference_logits(tokens, audio_features)

                if i == 0 and no_speech_token is not None:
                    # Store mx.array — defer .tolist() until after loop
                    no_speech_probs_mx = mx.softmax(
                        logits[:, sot_index].astype(mx.float32), axis=-1
                    )[:, no_speech_token]

                logits = logits[:, -1]

                # Apply fused constant suppression mask (single addition)
                logits = logits + suppress_mask

                # SuppressBlank only on the first sample step
                if i == 0 and blank_mask is not None:
                    logits = logits + blank_mask

                # Timestamp rules (the only stateful per-step filter)
                if ts_filter is not None:
                    if cpu_sampled is not None:
                        logits = ts_filter.apply_cpu(logits, cpu_sampled)
                    elif cpu_sampled_buf is not None:
                        logits = ts_filter.apply_cpu_batch(
                            logits, cpu_sampled_buf[:, :n_sampled]
                        )
                    else:
                        logits = ts_filter.apply(logits, tokens)

                tokens, completed, sum_logprobs = decoder_update(
                    tokens, logits, sum_logprobs
                )

                # Fetch the sampled token id and probability in one sync.
                last_step_probs = decoder_last_step_probs()
                if last_step_probs is not None:
                    step_meta = np.array(
                        mx.stack(
                            [
                                tokens[:, -1].astype(mx.float32),
                                last_step_probs.astype(mx.float32),
                            ],
                            axis=1,
                        )
                    )
                    last_col = step_meta[:, 0].astype(np.int32, copy=False)
                    step_probs_np = step_meta[:, 1]
                else:
                    last_col = np.array(tokens[:, -1])
                    step_probs_np = None

                if step_probs_np is not None:
                    cols = token_prob_len[active_indices]
                    token_prob_buf[active_indices, cols] = step_probs_np[:len(active_indices)]
                    token_prob_len[active_indices] = cols + 1

                if cpu_sampled is not None:
                    cpu_sampled.append(int(last_col[0]))
                elif cpu_sampled_buf is not None:
                    cpu_sampled_buf[:, n_sampled] = last_col
                    n_sampled += 1

                if tokens.shape[-1] > n_ctx:
                    remaining_tokens = np.array(tokens)
                    remaining_sum_logprobs = np.array(sum_logprobs)
                    for row, global_idx in enumerate(active_indices):
                        finalized_tokens[global_idx] = remaining_tokens[row]
                        finalized_sum_logprobs[global_idx] = remaining_sum_logprobs[row]
                    active_indices = np.empty(0, dtype=np.int32)
                    break

                if is_single:
                    if last_col[0] == eot_id:
                        finalized_tokens[active_indices[0]] = np.array(tokens[0])
                        finalized_sum_logprobs[active_indices[0]] = float(
                            np.array(sum_logprobs[0])
                        )
                        active_indices = np.empty(0, dtype=np.int32)
                        break
                    continue

                completed_mask = last_col == eot_id
                if not completed_mask.any():
                    continue

                finished_positions = np.flatnonzero(completed_mask)
                finished_idx = mx.array(finished_positions, dtype=mx.int32)
                finished_tokens = np.array(tokens[finished_idx])
                finished_sum_logprobs = np.array(sum_logprobs[finished_idx])
                for row, local_idx in enumerate(finished_positions):
                    global_idx = active_indices[local_idx]
                    finalized_tokens[global_idx] = finished_tokens[row]
                    finalized_sum_logprobs[global_idx] = finished_sum_logprobs[row]

                if finished_positions.size == active_indices.size:
                    active_indices = np.empty(0, dtype=np.int32)
                    break

                keep_positions = np.flatnonzero(~completed_mask)
                keep_idx = mx.array(keep_positions, dtype=mx.int32)
                tokens = tokens[keep_idx]
                audio_features = audio_features[keep_idx]
                sum_logprobs = sum_logprobs[keep_idx]
                active_indices = active_indices[keep_positions]
                inference_rearrange(keep_positions.tolist())
                if cpu_sampled_buf is not None:
                    cpu_sampled_buf = cpu_sampled_buf[keep_positions]
        finally:
            self.inference.reset()

        if active_indices.size:
            remaining_tokens = np.array(tokens)
            remaining_sum_logprobs = np.array(sum_logprobs)
            for row, global_idx in enumerate(active_indices):
                finalized_tokens[global_idx] = remaining_tokens[row]
                finalized_sum_logprobs[global_idx] = remaining_sum_logprobs[row]

        # Deferred no_speech_probs sync (was previously on step 0)
        if no_speech_probs_mx is not None:
            no_speech_probs = no_speech_probs_mx.tolist()
        else:
            no_speech_probs = [np.nan] * n_batch

        if any(seq is None for seq in finalized_tokens):
            raise RuntimeError("decode loop finalized an incomplete token set")

        max_len = max(len(seq) for seq in finalized_tokens)
        tokens_np = np.full((n_batch, max_len), eot_id, dtype=np.int32)
        for i, seq in enumerate(finalized_tokens):
            tokens_np[i, : len(seq)] = seq

        # Convert pre-allocated prob buffer to per-sequence lists for consumers.
        token_prob_lists = [
            token_prob_buf[i, :token_prob_len[i]] if token_prob_len[i] > 0 else None
            for i in range(n_batch)
        ]

        return (
            mx.array(tokens_np),
            mx.array(finalized_sum_logprobs),
            no_speech_probs,
            token_prob_lists,
        )

    def run(self, mel: mx.array) -> List[DecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: mx.array = self._get_audio_features(mel)  # encoder forward pass
        tokens: np.array = np.array(self.initial_tokens)
        tokens = np.broadcast_to(tokens, (n_audio, len(self.initial_tokens))).copy()

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(language=language, language_probs=probs)
                for language, probs in zip(languages, language_probs)
            ]

        # repeat tokens by the group size, for beam search or best-of-n sampling
        tokens = mx.array(tokens)
        if self.n_group > 1:
            tokens = tokens[:, None, :]
            tokens = mx.broadcast_to(
                tokens, [n_audio, self.n_group, len(self.initial_tokens)]
            )
            tokens = tokens.reshape(
                tokens, (n_audio * self.n_group, len(self.initial_tokens))
            )

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs, token_prob_lists = self._main_loop(
            audio_features, tokens
        )

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens = tokens[..., self.sample_begin :].tolist()
        tokens = [[t[: t.index(tokenizer.eot)] for t in s] for s in tokens]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i] for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        results = []
        for k, (text, language, toks, avg_logprob, no_speech_prob) in enumerate(
            zip(*fields)
        ):
            probs = None
            candidate_idx = k * self.n_group + selected[k]
            if candidate_idx < len(token_prob_lists):
                tok_probs = token_prob_lists[candidate_idx]
                if tok_probs is not None and len(tok_probs) > 0:
                    probs = tok_probs[: len(toks)].copy()
            results.append(
                DecodingResult(
                    language=language,
                    tokens=toks,
                    text=text,
                    avg_logprob=avg_logprob,
                    no_speech_prob=no_speech_prob,
                    temperature=self.options.temperature,
                    token_probs=probs,
                )
            )
        return results


def decode(
    model: "Whisper",
    mel: mx.array,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: mx.array, shape = (80, 3000) or (*, 80, 3000)
        An array containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel[None]

    if kwargs:
        options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(mel)
    return result[0] if single else result
