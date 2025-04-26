#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch
from funasr.utils.torch_function import sequence_mask


def export_rebuild_model(model, **kwargs):
    model.device = kwargs.get("device")
    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)
    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    model.export_name = types.MethodType(export_name, model)
    return model

def export_forward(
    self,
    speech: torch.Tensor,
    masks: torch.Tensor,
    position_encoding: torch.Tensor,
    **kwargs,
):
    encoder_out, encoder_out_lens = self.encoder(speech, masks, position_encoding)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]

    ctc_logits = self.ctc.ctc_lo(encoder_out)
    
    return ctc_logits, encoder_out_lens

def export_dummy_inputs(self):
    speech = torch.randn(1, self.seq_len, 560)
    masks = torch.zeros(1, 1, self.seq_len, dtype=torch.float32)
    position_encoding = torch.randn(1, self.seq_len, 560, dtype=torch.float32)
    return (speech, masks, position_encoding)

def export_input_names(self):
    # return ["speech", "speech_lengths", "language", "textnorm"]
    return ["speech", "masks", "position_encoding"]

def export_output_names(self):
    return ["ctc_logits", "encoder_out_lens"]

def export_dynamic_axes(self):
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {0: "batch_size"},
        "language": {0: "batch_size"},
        "textnorm": {0: "batch_size"},
        "ctc_logits": {0: "batch_size", 1: "logits_length"},
        "encoder_out_lens":  {0: "batch_size"},
    }

def export_name(self):
    return "model.onnx"

