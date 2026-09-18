# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from string import punctuation
from typing import TypeAlias

import numpy as np
import riva.client.proto.riva_asr_pb2 as riva_asr_pb2
import tensorrt as trt
import tensorrt_llm
import tensorrt_llm.logger as logger
import torch
import triton_python_backend_utils as pb_utils
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.bindings import executor as trtllm
from tensorrt_llm.runtime import ModelConfig, ModelRunnerCpp, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo
from torch.utils.dlpack import from_dlpack
import copy


TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
logging.basicConfig(
    format='%(asctime)s [%(levelname)s]: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ['TLLM_LOG_LEVEL'] = 'ERROR'

FileName: TypeAlias = str | Path

torch_dtypes = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32,
}


def read_config(component, engine_dir, mode='decoder'):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    if mode == 'decoder':
        model_config.update(config['pretrained_config'])
        model_config.update(config['build_config'])
    else:
        model_config.update(config)
    return model_config


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class CanaryTokenizer:
    def __init__(self, engine_dir, prompt_format='canary2'):
        vocab_file = os.path.join(engine_dir, 'decoder/vocab.json')
        decoder_config = read_config('decoder', engine_dir)
        self.prompt_format = decoder_config.get('prompt_format', prompt_format)
        self.blank = '▁'
        self.has_country_code = False

        with open(vocab_file, 'r') as jfp:
            vocab = json.load(jfp)

        self.token_id_offset = vocab['offsets']
        self.langs = [k for k in vocab['tokens']]
        self.__id_to_token__ = {l: {} for l in vocab['tokens']}
        self.spl_tokens = self.__id_to_token__['spl_tokens']

        for lang in vocab['tokens']:
            self.__id_to_token__[lang] = {int(k): v for k, v in vocab['tokens'][lang].items()}

        self.__token_to_id__ = {}
        for lang in self.__id_to_token__:
            self.__token_to_id__[lang] = {v: k for k, v in self.__id_to_token__[lang].items()}

        if self.prompt_format == 'canary_riva_rand':
            self.langs = []
            for token in self.__token_to_id__['spl_tokens']:
                if "-" in token:
                    self.langs.append(token.replace("<|", "").replace("|>", ""))
            self.langs.append("multi")
        else:
            self.langs = [
                t.strip('<').strip('>').strip('|')
                for t in self.__token_to_id__['spl_tokens']
                if re.match(r'^<\|[a-z]{2}(-[A-Z]{2})?\|>$', t)
            ]
            self.has_country_code = any('-' in l for l in self.langs)

        dpl = 'en-US' if self.has_country_code else 'en'

        if self.prompt_format == 'canary2':
            self.default_prompt = (
                f"<|startofcontext|> <|startoftranscript|> <|emo:undefined|> "
                f"<|{dpl}|> <|{dpl}|> <|nopnc|> <|noitn|> <|noromanized|> "
                f"<|notimestamp|> <|nodiarize|>"
            )
        else:
            self.default_prompt = f"<|startoftranscript|> <|{dpl}|> <|transcribe|> <|{dpl}|> <|pnc|>"

        self.id_to_token = {}
        for lang in self.__id_to_token__:
            self.id_to_token.update(self.__id_to_token__[lang])

        self.bos_id = vocab.get('bos_id', self.spl_tokens.get('<|startoftranscript|>'))
        self.eos_id = vocab.get('eos_id', self.spl_tokens.get('<|endoftext|>'))
        self.pad_id = vocab['pad_id']
        self.blank_id = self.__token_to_id__['spl_tokens'][self.blank]

    @staticmethod
    def word_separator(lang):
        if lang in [
            'ja-JP', 'ko-KR', 'zh-CN', 'th-TH', 'km-KH', 'my-MM', 'lo-LA',
            'ja', 'ko', 'zh', 'th', 'km', 'my', 'lo',
        ]:
            return ''
        return " "

    def ids_to_lang(self, token_ids: list):
        langs_count = {k: 0 for k in self.langs}
        max_lang = self.langs[0]
        for token_id in token_ids:
            for l in langs_count:
                if token_id in self.__id_to_token__[l]:
                    langs_count[l] += 1
                    if langs_count[l] > langs_count[max_lang]:
                        max_lang = l
        return max_lang

    def ids_to_tokens(self, token_ids: list):
        return [self.id_to_token[k] if k in self.id_to_token else "" for k in token_ids]

    def token_to_id(self, token: str, lang='spl_tokens'):
        return self.__token_to_id__[lang].get(token, self.token_id_offset[lang])

    def tokens_to_ids(self, tokens: list | str, lang='spl_tokens'):
        if isinstance(tokens, str):
            tokens = tokens.split(' ')
        return [self.token_to_id(k, lang) for k in tokens]

    def _resolve_lang_key(self, lang: str) -> str:
        if lang is None:
            return None
        if lang in self.__id_to_token__:
            return lang
        for key in self.__id_to_token__:
            if key.startswith(lang + '-'):
                return key
        base = lang.split('-')[0]
        if base in self.__id_to_token__:
            return base
        return None

    def ids_to_text(self, ids: list, lang=None):
        MAX_REPEAT = 10
        clean_ids = []
        prev_id = 0
        id_count = 0
        for i in ids:
            if prev_id == i:
                id_count += 1
                if id_count >= MAX_REPEAT:
                    continue
            else:
                id_count = 0
                prev_id = i

            if i == self.eos_id:
                break

            if i not in self.__id_to_token__['spl_tokens']:
                clean_ids.append(i)

        if lang is None:
            return ''.join(self.ids_to_tokens(clean_ids)).replace('▁', ' ').strip()

        resolved_lang = self._resolve_lang_key(lang)
        if resolved_lang is None or resolved_lang not in self.__id_to_token__:
            return ''.join(self.ids_to_tokens(clean_ids)).replace('▁', ' ').strip()

        ws = self.word_separator(lang)
        lookup = self.__id_to_token__[resolved_lang]
        if self.prompt_format == 'canary_riva_rand':
            lookup = self.__id_to_token__.get('unified', lookup)
        tokens = [lookup.get(k, " <unk> ").replace('▁', ws) for k in clean_ids]

        if ws == "":
            return re.sub(r'(?<=[.,;:])(?=[^\s])', r' ', ''.join(tokens).strip())
        return ''.join(tokens).strip()

    def get_prompt_v2(
        self, pnc=True, src_lang='en', tgt_lang=None,
        itn=False, timestamp=False, diarize=False,
    ):
        if not self.has_country_code and '-' in src_lang:
            src_lang = src_lang.split('-')[0]

        lang_token = f"<|{src_lang}|>"
        if lang_token not in self.__token_to_id__['spl_tokens']:
            raise ValueError(
                f"Invalid language {src_lang=} specified — "
                f"{lang_token} not found in spl_tokens"
            )

        tgt_lang = src_lang

        prompt  = "<|startofcontext|> <|startoftranscript|> <|emo:undefined|>"
        prompt += f" <|{src_lang}|> <|{tgt_lang}|>"
        prompt += f" {'<|nopnc|>' if pnc else '<|nopnc|>'}"
        prompt += f" {'<|itn|>' if itn else '<|noitn|>'}"
        prompt += " <|noromanized|>"
        prompt += f" {'<|timestamp|>' if timestamp else '<|notimestamp|>'}"
        prompt += f" {'<|diarize|>' if diarize else '<|nodiarize|>'}"
        return prompt

    def get_prompt_legacy(self, task_type='transcribe', pnc=True, src_lang='en', tgt_lang=None):
        prompt = "<|startoftranscript|>"

        if src_lang not in self.langs and "-" in src_lang and src_lang.split('-')[0] in self.langs:
            src_lang = src_lang.split('-')[0]

        if src_lang not in self.langs:
            raise ValueError(f"Invalid language {src_lang=} specified")

        prompt += f" <|{src_lang}|>"
        pnc_tag = "<|pnc|>" if pnc else "<|nopnc|>"

        if task_type in ('translate', 'ast'):
            if tgt_lang is None:
                tgt_lang = src_lang
            if tgt_lang not in self.langs and "-" in tgt_lang and tgt_lang.split('-')[0] in self.langs:
                tgt_lang = tgt_lang.split('-')[0]
            if tgt_lang not in self.langs:
                raise ValueError(f"Invalid language {tgt_lang=} specified")
            prompt += f" {self.task[task_type]} <|{tgt_lang}|> {pnc_tag}"
        elif task_type in ("transcribe", 'asr'):
            prompt += f" {self.task[task_type]} <|{src_lang}|> {pnc_tag}"
        else:
            raise ValueError(f"Invalid task {task_type=} specified")
        return prompt

    def get_prompt_canary_riva_rand(self, task='asr', pnc=False, src_lang='en-US', tgt_lang="en-US"):
        prompt = "<|startoftranscript|>"
        prompt += f" <|{src_lang}|>"
        if task in ('asr', 'transcribe'):
            prompt += " <|asr|>"
        else:
            prompt += f" <|{tgt_lang}|>"
        return prompt

    def get_prompt_ids_from_cfg(self, cfg):
        if self.prompt_format == 'canary2':
            return self.tokens_to_ids(
                self.get_prompt_v2(cfg['pnc'], cfg['source_language'])
            )
        elif self.prompt_format == 'canary_riva_rand':
            return self.tokens_to_ids(
                self.get_prompt_canary_riva_rand(
                    cfg['task'], cfg['pnc'], cfg['source_language'], cfg['target_language']
                )
            )
        else:
            return self.tokens_to_ids(
                self.get_prompt_legacy(
                    cfg['task'], cfg['pnc'], cfg['source_language'], cfg['target_language']
                )
            )

    def encode(self, prompt):
        return self.tokens_to_ids(prompt.split())

    def decode(self, ids: list, lang=None):
        text = self.ids_to_text(ids, lang)
        return re.sub(r'<\|.*?\|>', '', text)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CanaryEncoder:
    def __init__(self, engine_dir):
        engine_path = os.path.join(engine_dir, 'encoder.plan')
        self.encoder_config = json.load(open(os.path.join(engine_dir, 'config.json'), 'r'))
        logger.info(f"Loading engine from {engine_path}")
        with open(engine_path, "rb") as f:
            engine_buffer = f.read()
        logger.info(f"Creating session from engine {engine_path}")
        self.session = Session.from_serialized_engine(engine_buffer)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
        self.cuda_stream = torch.cuda.current_stream('cuda')

    @staticmethod
    def get_masked_emb(enc_outputs):
        enc_emb = enc_outputs.get('encoded_outputs', enc_outputs.get('outputs'))
        enc_lens = enc_outputs['encoded_lengths']
        batch_size = enc_lens.shape[0]
        max_length = enc_emb.shape[1]

        mask = (
            torch.arange(max_length, device='cuda')
            .unsqueeze(0)
            .expand(batch_size, max_length)
            < enc_lens.unsqueeze(1)
        )
        enc_mask = torch.where(mask.unsqueeze(2), enc_emb, 0.0)
        return enc_mask, enc_lens

    def infer(self, mels, mel_lens):
        if mel_lens is None:
            mel_lens = torch.tensor(
                [mels.shape[2] for _ in range(mels.shape[0])],
                dtype=torch.int64, device='cuda'
            )

        audio_inputs = {'audio_signal': mels, 'length': mel_lens}

        outputs_info = self.session.infer_shapes([
            TensorInfo("audio_signal", trt.DataType.FLOAT, mels.shape),
            TensorInfo("length", trt.DataType.INT64, mel_lens.shape),
        ])
        enc_outputs = {
            t.name: torch.empty(
                tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda:0"
            )
            for t in outputs_info
        }

        is_ok = self.session.run(audio_inputs, enc_outputs, self.cuda_stream.cuda_stream)
        assert is_ok, "Runtime execution failed for Conformer Encoder session"
        self.cuda_stream.synchronize()

        enc_states, enc_lens = self.get_masked_emb(enc_outputs)
        enc_lens = torch.clip(enc_lens, max=enc_states.shape[1])
        return enc_states, enc_lens


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class CanaryDecoding:
    def __init__(
        self,
        engine_dir,
        runtime_mapping,
        tokenizer,
        debug_mode=False,
        device="cuda:0",
        use_py_session=False,
        use_inflight_batching=False,
    ):
        self.tokenizer = tokenizer
        self.decoder_config = read_config('decoder', engine_dir)
        self.prompt_format = self.decoder_config['prompt_format']
        self.dtype = str_dtype_to_torch(self.decoder_config['dtype'])
        self.max_seq_len = self.decoder_config['max_seq_len']
        self.max_input_len = self.decoder_config['max_input_len']
        self.device = device
        self.use_py_session = use_py_session
        self.use_inflight_batching = False  # always False — cpp session handles batching
        self.decoder_generation_session = self.get_cpp_session(engine_dir, runtime_mapping, debug_mode)

    def get_cpp_session(self, engine_dir, runtime_mapping, debug_mode=False):
        runner_kwargs = dict(
            engine_dir=os.path.join(engine_dir, 'decoder'),
            is_enc_dec=False,
            max_batch_size=self.decoder_config['max_batch_size'],
            max_input_len=self.max_input_len,
            max_output_len=self.max_seq_len - self.max_input_len,
            max_beam_width=self.decoder_config['max_beam_width'],
            debug_mode=debug_mode,
            kv_cache_free_gpu_memory_fraction=0.9,
            cross_kv_cache_fraction=0.5,
        )
        return ModelRunnerCpp.from_dir(**runner_kwargs)

    def generate(
        self, decoder_input_ids, encoder_outputs, encoder_input_lengths,
        max_new_tokens, num_beams=1
    ):
        encoder_outputs = encoder_outputs.to(dtype=self.dtype)
        batch_size = decoder_input_ids.shape[0]
        encoder_max_input_length = encoder_outputs.shape[1]

        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1]] * batch_size,
            dtype=torch.int32, device='cuda'
        )
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        assert decoder_max_input_length <= self.max_input_len, (
            f"Decoder input length {decoder_max_input_length} exceeds "
            f"max input length {self.max_input_len}"
        )

        if max_new_tokens > self.max_seq_len:
            logging.warning(
                f"max_new_tokens {max_new_tokens} > max_seq_len {self.max_seq_len}, "
                f"clamping to max_seq_len"
            )
            max_new_tokens = self.max_seq_len
        max_new_tokens -= self.max_input_len

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()

        # cpp session path — stateless, safe for concurrent batch sizes
        cross_attention_masks = [
            torch.ones(
                [decoder_input_lengths[i] + max_new_tokens, encoder_input_lengths[i]],
                dtype=torch.bool,
                device='cuda',
            )
            for i in range(batch_size)
        ]
        decoder_input_ids_list = unpack_tensors(decoder_input_ids, decoder_input_lengths)
        encoder_outputs_list = unpack_tensors(encoder_outputs, encoder_input_lengths)

        out = self.decoder_generation_session.generate(
            batch_input_ids=decoder_input_ids_list,
            encoder_input_features=encoder_outputs_list,
            encoder_output_lengths=encoder_input_lengths,
            cross_attention_masks=cross_attention_masks,
            max_new_tokens=max_new_tokens,
            end_id=self.tokenizer.eos_id,
            pad_id=self.tokenizer.pad_id,
            num_beams=num_beams,
            output_sequence_lengths=True,
            return_dict=True,
        )
        output_ids = out['output_ids'].cpu().numpy().tolist()
        return output_ids, None


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class CanaryTRTLLM(object):
    def __init__(
        self, engine_dir, debug_mode=False, device="cuda:0",
        use_py_session=False, use_inflight_batching=False
    ):
        self.device = device
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)

        self.encoder_config = read_config('encoder', engine_dir, mode='encoder')
        self.decoder_config = read_config('decoder', engine_dir)
        self.max_seq_len = self.decoder_config['max_seq_len']
        self.max_input_len = self.decoder_config['max_input_len']
        self.encoder_config['max_batch_size'] = 64
        self.max_batch_size = self.encoder_config['max_batch_size']
        self.num_beams = self.decoder_config['max_beam_width']
        self.max_output_len = self.max_seq_len - self.max_input_len
        self.decoder_dtype = self.decoder_config['dtype']

        self.tokenizer = CanaryTokenizer(engine_dir)
        self.decoder = CanaryDecoding(
            engine_dir,
            runtime_mapping,
            tokenizer=self.tokenizer,
            debug_mode=debug_mode,
            device=self.device,
            use_py_session=False,
            use_inflight_batching=False,
        )

    def generate(
        self, encoder_output, encoder_output_lengths, prompt_ids,
        prompt_lengths, num_beams=1, max_new_tokens=None,
    ):
        decoder_input_ids = torch.stack(prompt_ids, dim=0)

        if max_new_tokens is None:
            max_new_tokens = self.max_seq_len

        output_ids, request_ids = self.decoder.generate(
            decoder_input_ids,
            encoder_output,
            encoder_output_lengths,
            max_new_tokens=max_new_tokens,
            num_beams=self.num_beams,
        )
        return output_ids, request_ids

    def decode(self, output_ids):
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.ids_to_text(output_ids[i][0]).strip()
            text = text.lstrip(punctuation)
            texts.append(text)
        return texts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unpack_tensors(input_tensors, input_tensor_lengths):
    # Fast path: all same length (common after min-frame padding)
    if input_tensor_lengths.min() == input_tensor_lengths.max():
        max_len = input_tensor_lengths[0].item()
        return [input_tensors[i, :max_len] for i in range(len(input_tensors))]
    return [input_tensors[i, :input_tensor_lengths[i]] for i in range(len(input_tensors))]


def remove_tensor_padding(input_tensor, input_tensor_lengths=None, pad_value=None):
    if pad_value:
        assert input_tensor_lengths is None
        assert torch.all(input_tensor[:, 0] != pad_value)
        mask = input_tensor != pad_value
        return input_tensor[mask].view(1, -1)
    else:
        assert input_tensor_lengths is not None
        valid_sequences = [
            input_tensor[i, :input_tensor_lengths[i]]
            for i in range(input_tensor.shape[0])
        ]
        return torch.cat(valid_sequences, dim=0)


# ---------------------------------------------------------------------------
# Triton model
# ---------------------------------------------------------------------------

class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.is_decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.init_model(self.model_config['parameters'])

    def init_model(self, parameters):
        self.inference_lock = threading.Lock()

        for key, value in parameters.items():
            parameters[key] = value["string_value"]

        self.engine_dir = Path(os.path.dirname(__file__))
        self.encoder_engine_dir = os.path.join(self.engine_dir, 'engine', 'encoder')
        self.decoder_engine_dir = os.path.join(self.engine_dir, 'engine', 'decoder')

        with open(os.path.join(self.encoder_engine_dir, 'config.json'), 'r') as f:
            conf = json.load(f)
            self.encoder_config = conf['builder_config']
            self.feature_config = conf['feature_config']

        self.encoder = CanaryEncoder(engine_dir=self.encoder_engine_dir)
        self.decoder = CanaryTRTLLM(
            engine_dir=Path(os.path.dirname(__file__)) / "engine",
            debug_mode=False,
            device="cuda:0",
            use_py_session=False,
            use_inflight_batching=False,
        )

        self.max_beam_width = self.decoder.decoder.decoder_config['max_beam_width']

        if "en-US" in self.decoder.tokenizer.langs:
            self.default_language = "en-US"
        elif "en" in self.decoder.tokenizer.langs:
            self.default_language = "en"
        else:
            self.default_language = parameters["language_code"]

        if self.default_language not in self.decoder.tokenizer.langs and '-' in self.default_language:
            self.default_language = self.default_language.split('-')[0]

        if self.default_language not in self.decoder.tokenizer.langs:
            if self.decoder.tokenizer.langs[0] == 'spl_tokens':
                self.default_language = self.decoder.tokenizer.langs[1]
            else:
                self.default_language = self.decoder.tokenizer.langs[0]

        self.subsampling_rate = 8
        self.left_padding_size = self.feature_config['left_padding_size']
        self.right_padding_size = self.feature_config['right_padding_size']
        self.chunk_size = self.feature_config['chunk_size']

        assert self.chunk_size > self.left_padding_size + self.right_padding_size, (
            "Chunk size should be greater than left padding size + right padding size."
        )

        self.ms_per_timestep_out = self.feature_config['ms_per_timestep']
        assert self.ms_per_timestep_out % self.subsampling_rate == 0, (
            "ms_per_timestep expected to be a multiple of subsampling_rate for canary."
        )

        # Correct mel frame rate: window_stride (10ms), not ms_per_timestep_out/subsampling
        self.ms_per_timestep_in = int(self.feature_config.get('window_stride', 0.01) * 1000)

        self.chunk_frames_in = int(self.chunk_size * 1000 // self.ms_per_timestep_in)
        self.left_padding_frames_in = int(self.left_padding_size * 1000 // self.ms_per_timestep_in)
        self.right_padding_frames_in = int(self.right_padding_size * 1000 // self.ms_per_timestep_in)
        self.buffer_frames_in = (
            self.left_padding_frames_in + self.right_padding_frames_in + self.chunk_frames_in
        )
        self.chunk_frames_out = int(self.chunk_size * 1000 // self.ms_per_timestep_out)
        self.left_padding_frames_out = int(self.left_padding_size * 1000 // self.ms_per_timestep_out)
        self.right_padding_frames_out = int(self.right_padding_size * 1000 // self.ms_per_timestep_out)
        self.buffer_frames_out = (
            self.left_padding_frames_out + self.right_padding_frames_out + self.chunk_frames_out
        )
        self.num_features = self.feature_config['num_features']
        self.stddev_floor = self.feature_config['stddev_floor']
        self.norm_per_feature = self.feature_config['norm_per_feature']
        self.precalc_norm_params = self.feature_config['precalc_norm_params']

        # Minimum mel frames = 3 seconds worth (matches working offline script)
        self.min_mel_frames = int(3.0 * 1000 / self.ms_per_timestep_in)

        logging.info(
            f"CanaryTRTLLM initialized: ms_per_timestep_in={self.ms_per_timestep_in}, "
            f"chunk_frames_in={self.chunk_frames_in}, min_mel_frames={self.min_mel_frames}, "
            f"default_language={self.default_language}, "
            f"cpp_session=True, is_decoupled={self.is_decoupled}"
        )

    def execute(self, requests):
        if self.is_decoupled:
            responses = None
        else:
            responses = []

        for request in requests:
            response_sender = request.get_response_sender()
            try:
                inputs = self.prepare_inputs(request)
                mels = inputs['audio_signal']
                lengths = inputs['length']
                serialized_configs = inputs['config']
                batch_size = mels.shape[0]

                # CPU work — outside lock
                prompt_ids, prompt_lengths, tgt_langs, src_langs, num_beams = \
                    self.prepare_prompts(batch_size, serialized_configs)

                # GPU work — serialized
                with self.inference_lock:
                    enc_outs, enc_lens = self.forward_encoder(mels, lengths)
                    output_ids, _ = self.forward_decoder(
                        enc_outs, enc_lens, prompt_ids, prompt_lengths
                    )

                transcripts_str = self.postprocess_output(output_ids, tgt_langs)
                lang_ids = src_langs
                inference_response = self.create_inference_response(transcripts_str, lang_ids)
            
                if self.is_decoupled:
                    response_sender.send(
                        response=inference_response,
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                else:
                    responses.append(inference_response)

            except Exception as e:
                import traceback
                traceback.print_exc()
                error = pb_utils.TritonError(str(e))
                err_response = pb_utils.InferenceResponse(output_tensors=[], error=error)
                if self.is_decoupled:
                    response_sender.send(
                        response=err_response,
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                else:
                    responses.append(err_response)

        return None if self.is_decoupled else responses

    def prepare_inputs(self, request):
        fw_params = pb_utils.get_input_tensor_by_name(request, "fw_params")
        if fw_params is not None:
            fw_params = fw_params.as_numpy()

        mels = (
            from_dlpack(pb_utils.get_input_tensor_by_name(request, "audio_signal"))
            .clone()
            .to(device=self.device, dtype=torch.float32)
        )
        lengths = (
            from_dlpack(pb_utils.get_input_tensor_by_name(request, "length"))
            .clone()
            .to(self.device, dtype=torch.int64)
        )
        lengths = torch.clip(lengths, max=self.chunk_frames_in)
        max_feat_len = max(65, int(lengths.max()))
        mels = mels[:, :, :max_feat_len]
        batched_serialized_config = np.array(pb_utils.get_input_tensor_by_name(request, "config").as_numpy())
        if int(torch.min(lengths)) < 1 or int(torch.max(lengths)) > 3001:
            raise Exception(f"Invalid audio feature length should be between 1 and 3001 (30s) frames but is {lengths}")

        if len(mels.shape) == 2:
            # This shouldn't be required but just in case the backend does something or
            # we need to implement ragged batches we need to implement stuff here
            nmels, frames = mels.shape
            mels = torch.reshape(mels, (1, nmels, frames))

        mels = self.per_feature_normalize_batch(mels, lengths)

        inputs = {'fw_params': fw_params, 'audio_signal': mels, 'length': lengths, 'config': batched_serialized_config}
        return inputs

        """
        # Clip to max chunk size
        lengths = torch.clip(lengths, max=self.chunk_frames_in)

        # Validate before any padding
        if int(torch.min(lengths)) < 1:
            raise Exception(f"Invalid audio: length < 1 frame: {lengths}")

        # Pad to minimum 3s or batch max, whichever is larger
        max_feat_len = max(self.min_mel_frames, int(lengths.max()))

        if mels.shape[2] < max_feat_len:
            pad = torch.zeros(
                mels.shape[0], mels.shape[1],
                max_feat_len - mels.shape[2],
                dtype=mels.dtype, device=mels.device
            )
            mels = torch.cat([mels, pad], dim=2)
        else:
            mels = mels[:, :, :max_feat_len]

        # Update lengths to reflect padded minimum
        lengths = torch.clamp(lengths, min=self.min_mel_frames)

        if len(mels.shape) == 2:
            nmels, frames = mels.shape
            mels = torch.reshape(mels, (1, nmels, frames))

        batched_serialized_config = np.array(
            pb_utils.get_input_tensor_by_name(request, "config").as_numpy()
        )

        mels = self.per_feature_normalize_batch(mels, lengths)

        return {
            'fw_params': fw_params,
            'audio_signal': mels,
            'length': lengths,
            'config': batched_serialized_config
        }"""

    def prepare_prompts(self, batch_size, serialized_configs):
        prompt_ids = []
        tgt_langs = []
        src_langs = []
        num_beams = []

        for i in range(batch_size):
            config_len = int.from_bytes(bytes(serialized_configs[i][:4]), 'little')
            serialized_config = bytes(serialized_configs[i][4: 4 + config_len])
            req_obj = riva_asr_pb2.StreamingRecognizeRequest()
            req_obj.ParseFromString(serialized_config)
            req_cfg = {d.name: v for d, v in req_obj.streaming_config.config.ListFields()}
            req_cfg['pnc'] = req_cfg.get('enable_automatic_punctuation', False)
            req_cfg['num_beams'] = req_cfg.get('num_beams', self.max_beam_width)
            num_beams.append(req_cfg['num_beams'])
            req_cfg.update(req_obj.streaming_config.config.custom_configuration)

            src_lang_code = req_obj.streaming_config.config.language_code

            if not src_lang_code:
                src_lang_code = self.default_language
            elif src_lang_code not in self.decoder.tokenizer.langs:
                if src_lang_code.split('-')[0] in self.decoder.tokenizer.langs:
                    src_lang_code = src_lang_code.split('-')[0]
                else:
                    logging.warning(f"Invalid source language {src_lang_code}, using default")
                    src_lang_code = self.default_language

            if "source_language" not in req_cfg:
                req_cfg["source_language"] = src_lang_code
            else:
                if req_cfg["source_language"] not in self.decoder.tokenizer.langs:
                    req_cfg["source_language"] = req_cfg["source_language"].split("-")[0]

            if "task" not in req_cfg:
                req_cfg["task"] = "asr"

            # target language always mirrors source for ASR/transcribe
            req_cfg["target_language"] = req_cfg["source_language"]

            if req_cfg["source_language"] not in self.decoder.tokenizer.langs:
                logging.warning(
                    f"Invalid source language {req_cfg['source_language']}, "
                    f"defaulting to {self.default_language}"
                )
                req_cfg["source_language"] = self.default_language
                req_cfg["target_language"] = self.default_language

            if req_cfg["target_language"] not in self.decoder.tokenizer.langs:
                logging.warning(
                    f"Invalid target language {req_cfg['target_language']}, "
                    f"defaulting to {self.default_language}"
                )
                req_cfg["target_language"] = self.default_language

            tgt_langs.append(req_cfg["target_language"])
            src_langs.append(req_cfg["source_language"])

            prompt_id = self.decoder.tokenizer.get_prompt_ids_from_cfg(req_cfg)
            prompt_id = torch.tensor(prompt_id, dtype=torch.int32, device=self.device)
            prompt_ids.append(prompt_id)

        prompt_lengths = [len(p) for p in prompt_ids]
        return prompt_ids, prompt_lengths, tgt_langs, src_langs, num_beams

    def forward_encoder(self, mels, lengths):
        enc_out, enc_lens = self.encoder.infer(mels=mels, mel_lens=lengths)
        enc_out = enc_out.to(self.device, dtype=torch_dtypes[self.decoder.decoder_dtype])
        return enc_out, enc_lens

    def forward_decoder(self, enc_out, enc_lens, prompt_ids, prompt_lengths, num_beams=1):
        output_ids, request_ids = self.decoder.generate(enc_out, enc_lens, prompt_ids, prompt_lengths)
        return output_ids, request_ids

    def postprocess_output(self, output_ids, tgt_langs):
        transcripts_str = []
        for i, tgt_lang in enumerate(tgt_langs):
            transcript = output_ids[i][0]
            transcript_str = self.decoder.tokenizer.decode(transcript, tgt_lang).strip()
            transcripts_str.append(transcript_str)
        return transcripts_str

    def extract_lang_ids(self, output_ids, prompt_lengths=None):
        if prompt_lengths is None:
            prompt_lengths = [0] * len(output_ids)
        lang_ids = []
        for output_id, prompt_length in zip(output_ids, prompt_lengths):
            is_lang_id = True
            lang_id = ""
            while is_lang_id and prompt_length < len(output_id[0]):
                raw_temp_lang = self.decoder.tokenizer.ids_to_tokens([output_id[0][prompt_length]])[0]
                temp_lang = raw_temp_lang.replace("<|", "").replace("|>", "")
                if temp_lang in self.decoder.tokenizer.langs:
                    lang_id = temp_lang
                elif "<|" not in raw_temp_lang:
                    is_lang_id = False
                prompt_length += 1
            lang_ids.append(lang_id)
        return lang_ids

    def create_inference_response(self, transcripts_str, lang_ids=None):
        if lang_ids is None:
            lang_ids = [""] * len(transcripts_str)
        responses = []
        for transcript_str, lang_id in zip(transcripts_str, lang_ids):
            response = riva_asr_pb2.RecognizeResponse()
            result = riva_asr_pb2.SpeechRecognitionResult()
            alternative = riva_asr_pb2.SpeechRecognitionAlternative()
            alternative.transcript = transcript_str
            alternative.language_code.append(lang_id)
            result.alternatives.append(alternative)
            response.results.append(result)
            responses.append([response.SerializeToString()])
        out = pb_utils.Tensor("logprobs", np.array(responses, dtype=np.bytes_))
        return pb_utils.InferenceResponse(output_tensors=[out])

    def per_feature_normalize_batch(self, x, seq_len):
        # Fully batched GPU normalization — replaces slow per-item loop
        # x: [B, F, T],  seq_len: [B]
        seq_len = seq_len.to(x.device)
        # Build validity mask [B, T] → expand to [B, F, T]
        mask = (
            torch.arange(x.shape[2], device=x.device).unsqueeze(0)
            < seq_len.unsqueeze(1)
        ).unsqueeze(1).expand_as(x)  # [B, F, T]

        counts = seq_len.float().unsqueeze(1)     # [B, 1]
        x_masked = x * mask
        mean = x_masked.sum(dim=2) / counts        # [B, F]

        diff = (x - mean.unsqueeze(2)) * mask
        std = (diff.pow(2).sum(dim=2) / counts).sqrt()  # [B, F]
        std = torch.clamp(std, min=self.stddev_floor)

        normalized = (x - mean.unsqueeze(2)) / std.unsqueeze(2)
        return normalized * mask  # zero out padding frames

    def finalize(self):
        logging.info('Cleaning up CanaryTRTLLM model...')

