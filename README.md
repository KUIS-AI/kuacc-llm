# kuacc-llm
Common LLM setup for the KUACC cluster in Koc University, Istanbul.

tldr: `$ source setenv.sh` to set up your environment and access models/datasets without downloading them.

This currently sets the TORCH_HOME and HF_HOME and directs the following commands to use the cache under /datasets/NLP:
```
transformers.AutoModel.from_pretrained("gpt2")
datasets.load_dataset("tiny_shakespeare")
torchvision.get_model("resnet50", weights="DEFAULT")
```

To prompt a llm:
```
TODO
```

To use a llm+adapter:
```
TODO
```


## Downloaded resources:

### huggingface/transformers
* EleutherAI/gpt-j-6b (23G) (from open-llms)
* EleutherAI/gpt-neo-125m (505M) (from open-llms)
* EleutherAI/gpt-neox-20b (39G) (from open-llms)
* EleutherAI/pythia-* (70m (160M), 160m (360M), 410m (873M), 1b (2.0G), 1.4b (2.8G), 2.8b (5.4G), 6.9b (13G), 12b (23G)) (from open-llms)
* OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 (23G) (from open-llms)
* StabilityAI/stablelm-tuned-alpha-* (3b (14G), 7b (30G)) (from open-llms)
* aisquared/chopt-* (125m (240M), 350m (635M), 1_3b (2.5G), 2_7b (5.0G)) 
* aisquared/chopt-research-* (125m (240M), 350m (635M), 1_3b (2.5G), 2_7b (5.0G)) 
* aisquared/dlite-v2-* (124m (253M), 355m (704M), 774m (1.5G), 1_5b (3.0G)) (from open-llms, lightweight gpt-2 based, finetuned)
* bigscience/bloom-560m (1.1G) 
* cerebras/Cerebras-GPT-* (111M (467M), 256M (1.1G), 590M (2.3G), 1.3B (5.1G), 2.7B (11G), 6.7B (26G), 13B (49G)) (from open-llms)
* databricks/dolly-* (v1-6b (12G), v2-3b (5.4G), v2-7b (13G), v2-12b (23G)) (from open-llms)
* decapoda-research/llama-* (7b-hf (13G), 13b-hf (37G), 30b-hf (77G)) AutoConfig:ok, AutoTokenizer:wrong-name-error, AutoModel:ok
* google/flan-t5-* (small (298M), base (949M), large (3.0G), xl (11G), xxl (43G)) 
* google/flan-ul2 (37G), google/ul2 (37G) 
* openai-gpt (461M), gpt2 (528M), gpt2-medium (1.5G), gpt2-large (3.1G), gpt2-xl (6.1G)
* h2oai/h2ogpt-oig-oasst1-512-6.9b (13G) (from open-llms)
* huggyllama/llama-* (7b (13G), 13b (25G), 30b (61G), 65b (?)) 
* huggyllama/llama-65b ==
* lmsys/fastchat-t5-3b-v1.0 (6.3G) 
* lmsys/vicuna-* (7b-delta-v1.1 (13G), 13b-delta-v1.1 (25G)) 
* mosaicml/mpt-* (1b-redpajama-200b (5.0G), 1b-redpajama-200b-dolly (5.0G), 7b (13G), 7b-chat (13G), 7b-instruct (13G), 7b-storywriter (13G)) (from open-llms, requires einops, trust_remote_code=True, see hf page for details)
* nomic-ai/gpt4all-13b-snoozy ==
* nomic-ai/gpt4all-j (23G) (from open-llms)
* nomic-ai/gpt4all-j-lora -- FAILED! missing config
* nomic-ai/gpt4all-lora -- FAILED! missing config (gururise refers to it but don't know how to download, seems worse than tloen/alpaca-lora-7b)
* openlm-research/open_llama_7b_preview_300bt (13G) (from open-llms, AutoModel does not work, use `AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_7b_preview_300bt", subfolder="open_llama_7b_preview_300bt_transformers_weights")`)
* togethercomputer/GPT-JT-* (6B-v0 (12G), Moderation-6B (12G)) 
* togethercomputer/GPT-NeoXT-Chat-Base-20B ==
* togethercomputer/Pythia-Chat-Base-7B (13G) 
* togethercomputer/RedPajama-INCITE-* (Base-3B-v1 (5.4G), Base-7B-v0.1 (13G), Chat-3B-v1 (5.4G), Chat-7B-v0.1 (13G), Instruct-3B-v1 (5.4G), Instruct-7B-v0.1 (13G)) (from open-llms)


### huggingface/datasets

* bookcorpus (4.6G) 
* databricks/databricks-dolly-15k (12M) 
* enwik8 (99M) 
* glue (232M) 
* imdb (128M) 
* mosaicml/dolly_hhrlhf (46M) 
* openwebtext (38G) 
* piqa (5.2M) 
* ptb_text_only (5.8M) 
* sciq (7.4M) 
* squad (87M) 
* super_glue (285M) 
* tatsu-lab/alpaca (45M) (the original)
* tiny_shakespeare (1.2M) 
* wikitext (1.1G) 
* yahma/alpaca-cleaned (39M) (https://github.com/gururise/AlpacaDataCleaned as of 2023-04-10)


### torchvision.models

* all 121 models listed in torchvision.models.list_models() (25G):

['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet101', 'deeplabv3_resnet50', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fcn_resnet101', 'fcn_resnet50', 'fcos_resnet50_fpn', 'googlenet', 'inception_v3', 'keypointrcnn_resnet50_fpn', 'lraspp_mobilenet_v3_large', 'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 'maxvit_t', 'mc3_18', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mvit_v1_b', 'mvit_v2_s', 'quantized_googlenet', 'quantized_inception_v3', 'quantized_mobilenet_v2', 'quantized_mobilenet_v3_large', 'quantized_resnet18', 'quantized_resnet50', 'quantized_resnext101_32x8d', 'quantized_resnext101_64x4d', 'quantized_shufflenet_v2_x0_5', 'quantized_shufflenet_v2_x1_0', 'quantized_shufflenet_v2_x1_5', 'quantized_shufflenet_v2_x2_0', 'r2plus1d_18', 'r3d_18', 'raft_large', 'raft_small', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 's3d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 'swin3d_b', 'swin3d_s', 'swin3d_t', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']


## List of companies/users with sota models/datasets we should follow:

* https://huggingface.co/h2oai (h2ogpt)
* https://huggingface.co/databricks (dolly)
* https://huggingface.co/mosaicml (mpt)
* https://huggingface.co/bigscience (bloom)
* https://huggingface.co/togethercomputer (redpajama)
* https://huggingface.co/lmsys (fastchat, vicuna)
* https://huggingface.co/aisquared (dlite)
* https://huggingface.co/stabilityai (stablelm)
* https://huggingface.co/OpenAssistant (openassistant)
* https://huggingface.co/EleutherAI (pythia, gpt-neox)
* https://huggingface.co/young-geng (koala)
* https://huggingface.co/CarperAI (stable-vicuna)
* https://huggingface.co/cerebras (cerebras-gpt)
* https://huggingface.co/nomic-ai (gpt4all)
* https://huggingface.co/facebook (opt)
* https://huggingface.co/google (flan-t5, flan-ul2, everything (-:)
