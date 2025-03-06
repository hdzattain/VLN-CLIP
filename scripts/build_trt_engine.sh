#!/bin/bash
trtexec --onnx=clip_vln.onnx \
        --saveEngine=clip_vln.engine \
        --fp16 \
        --workspace=4096 \
        --best \
        --sparsity=enable \
        --verbose