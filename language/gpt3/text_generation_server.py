import os
import sys

sys.path.append(os.environ["MEGATRON_PATH"])

import datetime
import torch
import json
import threading
from flask import Flask, request, jsonify, current_app
from flask_restful import Resource, Api
from megatron import get_args
from megatron.text_generation.generation import (
    generate_tokens_probs_and_return_on_first_stage,
)
from megatron.text_generation import beam_search_and_post_process


from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
import torch


GENERATE_NUM = 0
BEAM_NUM = 1
lock = threading.Lock()


class MegatronGenerate(Resource):
    def __init__(self, model, gen_kwargs, log=None):
        self.model = model
        self.log = log
        self.gen_kwargs = gen_kwargs
        self.beam_width = self.gen_kwargs.get("beam_width", None)

    @staticmethod
    def send_do_generate():
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)

    @staticmethod
    def send_do_beam_search():
        choice = torch.cuda.LongTensor([BEAM_NUM])
        torch.distributed.broadcast(choice, 0)

    @staticmethod
    def sync_input(input_ids, input_length):
        input_length_tensor = torch.cuda.LongTensor(input_length)
        torch.distributed.broadcast(input_length_tensor, 0)
        input_ids_tensor = torch.cuda.LongTensor(input_ids)
        torch.distributed.broadcast(input_ids_tensor, 0)
        return input_ids_tensor, input_length_tensor

    def put(self):
        args = get_args()
        if not "input_ids" in request.get_json():
            return "input_ids argument required", 400

        if not "input_length" in request.get_json():
            return "input_length is required", 400

        input_ids = request.get_json()["input_ids"]
        input_length = request.get_json()["input_length"]

        with lock:  # Need to get lock to keep multiple threads from hitting code

            if self.log:
                print("request IP: " + str(request.remote_addr))
                print("start time: ", datetime.datetime.now())

            try:
                if self.beam_width is not None:
                    MegatronGenerate.send_do_beam_search()  # Tell other ranks we're doing beam_search
                    # TODO: implement beam_search
                    return jsonify({})
                else:
                    MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
                    input_ids_tensor, input_length_tensor = MegatronGenerate.sync_input(
                        input_ids, input_length
                    )
                    (
                        output_tokens,
                        _,
                        _,
                    ) = generate_tokens_probs_and_return_on_first_stage(
                        self.model,
                        input_ids_tensor,
                        input_length_tensor,
                        top_k=self.gen_kwargs.get("top_k", 4),
                        temperature=self.gen_kwargs.get("temperature", 0.0),
                    )
                    output_batch_truncated = []
                    for data, source_len in zip(output_tokens, input_length_tensor):
                        output_batch_truncated.append(data[source_len:])
                    # TODO: encode output
                    if self.log:
                        print("end time: ", datetime.datetime.now())
                    return jsonify({"output": output_batch_truncated.cpu().numpy().tolist()})

            except ValueError as ve:
                return ve.args[0]


class MegatronServer(object):
    def __init__(self, model, gen_kwargs):
        self.app = Flask(__name__, static_url_path="")
        api = Api(self.app)
        api.add_resource(
            MegatronGenerate, "/api", resource_class_args=[model, gen_kwargs]
        )

    def run(self, url):
        self.app.run(url, threaded=True, debug=False)


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0("building GPT model ...")
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process,
    )

    return model


if __name__ == "__main__":
    initialize_megatron(
        args_defaults={
            "tokenizer_type": "GPT2BPETokenizer",
            "no_load_rng": True,
            "no_load_optim": True,
        }
    )

    args = get_args()
    gen_kwargs = {
        "early_stopping": True,
        "max_new_tokens": 128,
        "min_new_tokens": 30,
        "top_k": 4,
        "temperature": 0.0,
    }
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        server = MegatronServer(model)
        server.run("0.0.0.0")

    while True:
        choice = torch.cuda.LongTensor(0)
        input_length_tensor = torch.cuda.LongTensor([0])
        input_ids_tensor = torch.cuda.LongTensor(0)
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 0:
            try:
                torch.distributed.broadcast(input_length_tensor, 0)
                input_ids_tensor = torch.cuda.LongTensor(
                    [
                        0
                        for _ in range(
                            input_length_tensor[0].item()
                            + gen_kwargs.get("max_new_tokens")
                        )
                    ]
                )
                torch.distributed.broadcast(input_ids_tensor, 0)
                generate_tokens_probs_and_return_on_first_stage(
                    input_ids_tensor,
                    input_length_tensor,
                    top_k=gen_kwargs.get("top_k", 4),
                    temperature=gen_kwargs.get("temperature", 0.0),
                )
            except ValueError as ve:
                pass
        elif choice[0].item() == 1:
            try:
                # TODO: implement beam search
                pass
            except ValueError as ve:
                pass
