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

### huggingface/datasets

* bookcorpus (4.6G) ++
* databricks/databricks-dolly-15k (12M) ++
* enwik8 (99M) ++
* glue (232M) ++
* imdb (128M) ++
* mosaicml/dolly_hhrlhf (46M) ++
* openwebtext (38G) ++
* piqa (5.2M) ++
* ptb_text_only (5.8M) ++
* sciq (7.4M) ++
* squad (87M) ++
* super_glue (285M) ++
* tatsu-lab/alpaca (45M) ++ (the original)
* tiny_shakespeare (1.2M) ++
* wikitext (1.1G) ++
* yahma/alpaca-cleaned (39M) ++ (https://github.com/gururise/AlpacaDataCleaned as of 2023-04-10)

### huggingface/transformers
* EleutherAI/gpt-j-6b (23G) ++ (from open-llms)
* EleutherAI/gpt-neo-125m (505M) ++ (from open-llms)
* EleutherAI/gpt-neox-20b (39G) ++ (from open-llms)
* EleutherAI/pythia-* (70m (160M), 160m (360M), 410m (873M), 1b (2.0G), 1.4b (2.8G), 2.8b (5.4G), 6.9b (13G), 12b (?)) ++ (from open-llms)
* EleutherAI/pythia-12b () == (from open-llms)
* OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 (23G) ++ (from open-llms)
* StabilityAI/stablelm-tuned-alpha-* (3b (14G), 7b (30G)) ++ (from open-llms)
* aisquared/chopt-* (125m (240M), 350m (635M), 1_3b (2.5G), 2_7b (5.0G)) ++
* aisquared/chopt-research-* (125m (240M), 350m (635M), 1_3b (2.5G), 2_7b (5.0G)) ++
* aisquared/dlite-v2-* (124m (253M), 355m (704M), 774m (1.5G), 1_5b (3.0G)) ++ (from open-llms, lightweight gpt-2 based, finetuned)
* bigscience/bloom-560m (1.1G) ++
* cerebras/Cerebras-GPT-* (111M (467M), 256M (1.1G), 590M (2.3G), 1.3B (5.1G), 2.7B (11G), 6.7B (26G), 13B (49G)) ++ (from open-llms)
* databricks/dolly-* (v1-6b (12G), v2-3b (5.4G), v2-7b (13G), v2-12b (23G)) ++ (from open-llms)
* decapoda-research/llama-* (7b-hf (13G), 13b-hf (37G), 30b-hf (77G)) ++ AutoConfig:ok, AutoTokenizer:wrong-name-error, AutoModel:ok
* google/flan-t5-* (small (298M), base (949M), large (3.0G), xl (11G), xxl (43G)) ++
* google/flan-ul2 (37G) ++
* google/ul2 (37G) ++
* gpt2 (528M) ++
* gpt2-* (medium (1.5G), large (3.1G), xl (6.1G)) ++
* h2oai/h2ogpt-oig-oasst1-512-6.9b (13G) ++ (from open-llms)
* huggyllama/llama-* (7b (13G), 13b (25G), 30b (61G), 65b (?)) ++
* huggyllama/llama-65b ==
* lmsys/fastchat-t5-3b-v1.0 (6.3G) ++
* lmsys/vicuna-* (7b-delta-v1.1 (13G), 13b-delta-v1.1 (25G)) ++
* mosaicml/mpt-1b-redpajama-* (200b (5.0G), 200b-dolly (5.0G)) ++ (from open-llms, requires einops, trust_remote_code=True, see hf page for details)
* mosaicml/mpt-7b (13G) ++
* mosaicml/mpt-7b-* (chat (13G), instruct (13G), storywriter (13G)) ++
* nomic-ai/gpt4all-13b-snoozy ==
* nomic-ai/gpt4all-j (23G) ++ (from open-llms)
* nomic-ai/gpt4all-j-lora -- FAILED! missing config
* nomic-ai/gpt4all-lora -- FAILED! missing config (gururise refers to it but don't know how to download, seems worse than tloen/alpaca-lora-7b)
* openai-gpt (461M) ++
* openlm-research/open_llama_7b_preview_300bt (13G) ++ (from open-llms, AutoModel does not work, use `AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_7b_preview_300bt", subfolder="open_llama_7b_preview_300bt_transformers_weights")`)
* togethercomputer/GPT-JT-* (6B-v0 (12G), Moderation-6B (12G)) ++
* togethercomputer/GPT-NeoXT-Chat-Base-20B ==
* togethercomputer/Pythia-Chat-Base-7B (13G) ++
* togethercomputer/RedPajama-INCITE-* (Base-3B-v1 (5.4G), Base-7B-v0.1 (13G), Chat-3B-v1 (5.4G), Chat-7B-v0.1 (13G), Instruct-3B-v1 (5.4G), Instruct-7B-v0.1 (13G)) ++ (from open-llms)

### torchvision.models

* all 121 models listed in torchvision.models.list_models() (25G): ['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet101', 'deeplabv3_resnet50', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fcn_resnet101', 'fcn_resnet50', 'fcos_resnet50_fpn', 'googlenet', 'inception_v3', 'keypointrcnn_resnet50_fpn', 'lraspp_mobilenet_v3_large', 'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 'maxvit_t', 'mc3_18', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mvit_v1_b', 'mvit_v2_s', 'quantized_googlenet', 'quantized_inception_v3', 'quantized_mobilenet_v2', 'quantized_mobilenet_v3_large', 'quantized_resnet18', 'quantized_resnet50', 'quantized_resnext101_32x8d', 'quantized_resnext101_64x4d', 'quantized_shufflenet_v2_x0_5', 'quantized_shufflenet_v2_x1_0', 'quantized_shufflenet_v2_x1_5', 'quantized_shufflenet_v2_x2_0', 'r2plus1d_18', 'r3d_18', 'raft_large', 'raft_small', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 's3d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 'swin3d_b', 'swin3d_s', 'swin3d_t', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']


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

## TODO:

* https://arxiv.org/abs/2210.17323 (GPTQ quantized)
* https://bair.berkeley.edu/blog/2023/04/03/koala/
* https://chat.lmsys.org/
* https://github.com/EleutherAI/lm-evaluation-harness
* https://github.com/ZrrSkywalker/LLaMA-Adapter
* https://github.com/eugeneyan/open-llms
* https://github.com/gururise/AlpacaDataCleaned 
* https://github.com/imaurer/awesome-decentralized-llm
* https://github.com/lm-sys/FastChat#vicuna-weights
* https://github.com/young-geng/EasyLM
* https://huggingface.co/OpenAssistant (from open-llms: training set and fine tuned models based on pythia and Llama)
* https://huggingface.co/google/flan-t5-xxl (from open-llms)
* https://huggingface.co/google/flan-ul2 (from open-llms)
* https://huggingface.co/young-geng/koala
* https://www.semianalysis.com/p/google-we-have-no-moat-and-neither
* huggingface/datasets: h2oai/h2ogpt-* (instruction tuning datasets)
* huggingface/datasets: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T
* huggingface/transformers: CarperAI/stable-vicuna-13b-delta (depends on llama-13b)
* huggingface/transformers: EleutherAI/pythia-XXX  (from open-llms, open source alternative to Llama?)
* huggingface/transformers: bigscience/bloom (176B) 
* huggingface/transformers: bigscience/bloom-1b1 
* huggingface/transformers: bigscience/bloom-1b7 
* huggingface/transformers: bigscience/bloom-3b 
* huggingface/transformers: bigscience/bloom-7b1 
* huggingface/transformers: bigscience/bloomz 
* huggingface/transformers: bigscience/bloomz-1b1 
* huggingface/transformers: bigscience/bloomz-1b7 
* huggingface/transformers: bigscience/bloomz-3b 
* huggingface/transformers: bigscience/bloomz-560m
* huggingface/transformers: bigscience/bloomz-7b1 
* huggingface/transformers: bigscience/bloomz-7b1-mt 
* huggingface/transformers: bigscience/bloomz-mt 
* huggingface/transformers: bigscience/mt0-base 
* huggingface/transformers: bigscience/mt0-large 
* huggingface/transformers: bigscience/mt0-small 
* huggingface/transformers: bigscience/mt0-xl 
* huggingface/transformers: bigscience/mt0-xxl 
* huggingface/transformers: bigscience/mt0-xxl-mt 
* huggingface/transformers: decapoda-research/llama-13b-hf-int4
* huggingface/transformers: decapoda-research/llama-30b-hf-int4
* huggingface/transformers: decapoda-research/llama-65b-hf
* huggingface/transformers: decapoda-research/llama-65b-hf-int4
* huggingface/transformers: decapoda-research/llama-7b-hf-int4
* huggingface/transformers: decapoda-research/llama-7b-hf-int8
* huggingface/transformers: decapoda-research/llama-smallint-pt
* huggingface/transformers: facebook/opt-1.3b 
* huggingface/transformers: facebook/opt-125m 
* huggingface/transformers: facebook/opt-13b 
* huggingface/transformers: facebook/opt-2.7b 
* huggingface/transformers: facebook/opt-30b 
* huggingface/transformers: facebook/opt-350m 
* huggingface/transformers: facebook/opt-6.7b 
* huggingface/transformers: facebook/opt-66b 
* huggingface/transformers: facebook/opt-iml-1.3b 
* huggingface/transformers: facebook/opt-iml-30b 
* huggingface/transformers: facebook/opt-iml-max-1.3b 
* huggingface/transformers: facebook/opt-iml-max-30b 
* huggingface/transformers: h2oai/h2ogpt-* (from open-llms)
* huggingface/transformers: huggyllama/llama-65b 
* huggingface/transformers: lmsys/vicuna-13b-delta-v0 
* huggingface/transformers: lmsys/vicuna-7b-delta-v0  
* huggingface/transformers: mosaicml/mosaic-bert-base
* huggingface/transformers: mosaicml/mosaic-bert-base-seqlen-1024
* huggingface/transformers: mosaicml/mosaic-bert-base-seqlen-2048
* huggingface/transformers: mosaicml/mosaic-bert-base-seqlen-256
* huggingface/transformers: mosaicml/mosaic-bert-base-seqlen-512
* huggingface/transformers: nomic-ai/gpt4all-lora-epoch-3 
* huggingface/transformers: samwit/alpaca7B-lora  (original alpaca: gururise refers to it but don't know how to download)
* huggingface/transformers: tloen/alpaca-lora-7b  (trained with yahma/alpaca-cleaned 2023-03-26: gururise refers to it but don't know how to download)

## adapter.hub


## torch.hub

https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved
The locations are used in the order of
* Calling hub.set_dir(<PATH_TO_HUB_DIR>)
* $TORCH_HOME/hub, if environment variable TORCH_HOME is set.
* $XDG_CACHE_HOME/torch/hub, if environment variable XDG_CACHE_HOME is set.
* ~/.cache/torch/hub

## torchvision

https://pytorch.org/vision/stable/models.html

Uses torch.hub.

## torchtext

https://pytorch.org/text/stable/datasets.html

Seems deprecated, its datasets probably exist in huggingface anyway.

## huggingface/datasets

https://huggingface.co/docs/datasets/cache:

The default cache directory is ~/.cache/huggingface/datasets. Change the cache location by
setting the shell environment variable, HF_DATASETS_CACHE to another directory. (It seems
like setting HF_HOME may also solve transformers).

https://github.com/huggingface/datasets/blob/e3f4f124a1b118a5bfff5bae76b25a68aedbebbc/src/datasets/config.py#L147:
```
# Cache location
DEFAULT_XDG_CACHE_HOME = "~/.cache"
XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_HF_CACHE_HOME = os.path.join(XDG_CACHE_HOME, "huggingface")
HF_CACHE_HOME = os.path.expanduser(os.getenv("HF_HOME", DEFAULT_HF_CACHE_HOME))
DEFAULT_HF_DATASETS_CACHE = os.path.join(HF_CACHE_HOME, "datasets")
HF_DATASETS_CACHE = Path(os.getenv("HF_DATASETS_CACHE", DEFAULT_HF_DATASETS_CACHE))
DEFAULT_HF_METRICS_CACHE = os.path.join(HF_CACHE_HOME, "metrics")
HF_METRICS_CACHE = Path(os.getenv("HF_METRICS_CACHE", DEFAULT_HF_METRICS_CACHE))
DEFAULT_HF_MODULES_CACHE = os.path.join(HF_CACHE_HOME, "modules")
HF_MODULES_CACHE = Path(os.getenv("HF_MODULES_CACHE", DEFAULT_HF_MODULES_CACHE))
DOWNLOADED_DATASETS_DIR = "downloads"
DEFAULT_DOWNLOADED_DATASETS_PATH = os.path.join(HF_DATASETS_CACHE, DOWNLOADED_DATASETS_DIR)
DOWNLOADED_DATASETS_PATH = Path(os.getenv("HF_DATASETS_DOWNLOADED_DATASETS_PATH", DEFAULT_DOWNLOADED_DATASETS_PATH))
EXTRACTED_DATASETS_DIR = "extracted"
DEFAULT_EXTRACTED_DATASETS_PATH = os.path.join(DEFAULT_DOWNLOADED_DATASETS_PATH, EXTRACTED_DATASETS_DIR)
EXTRACTED_DATASETS_PATH = Path(os.getenv("HF_DATASETS_EXTRACTED_DATASETS_PATH", DEFAULT_EXTRACTED_DATASETS_PATH))
```

## huggingface/transformers

HF_HOME should fix this one as well as datasets:

https://github.com/huggingface/transformers/blob/01734dba842c29408c96caa5c345c9e415c7569b/src/transformers/utils/hub.py#L74

```
hf_cache_home = os.path.expanduser(os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface")))
default_cache_path = os.path.join(hf_cache_home, "hub")
PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", PYTORCH_TRANSFORMERS_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", HUGGINGFACE_HUB_CACHE)
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules"))
```

huggingface/transformers environment variables full list:
```
AWS_REGION
BRANCH
CLEARML_LOG_MODEL
CLEARML_PROJECT
CLEARML_TASK
COMET_LOG_ASSETS
COMET_MODE
COMET_OFFLINE_DIRECTORY
COMET_PROJECT_NAME
DISABLE_MLFLOW_INTEGRATION
DISABLE_TELEMETRY
HF_DAGSHUB_LOG_ARTIFACTS
HF_DAGSHUB_MODEL_NAME
HF_HOME
HF_MLFLOW_LOG_ARTIFACTS
HF_MODULES_CACHE
HUGGINGFACE_HUB_CACHE
MLFLOW_EXPERIMENT_NAME
MLFLOW_FLATTEN_PARAMS
MLFLOW_NESTED_RUN
MLFLOW_RUN_ID
MLFLOW_TAGS
MLFLOW_TRACKING_URI
PYTORCH_PRETRAINED_BERT_CACHE
PYTORCH_TRANSFORMERS_CACHE
RUN_SLOW
RUN_TOKENIZER_INTEGRATION
SMDATAPARALLEL_LOCAL_RANK
SM_FRAMEWORK_MODULE
SM_FRAMEWORK_PARAMS
SM_HP_MP_PARAMETERS
SM_NUM_CPUS
SM_NUM_GPUS
TEST_SAGEMAKER
TORCH_HOME
TRAINING_JOB_ARN
TRANSFORMERS_CACHE
TRANSFORMERS_IS_CI
TRANSFORMERS_NO_ADVISORY_WARNINGS
TRANSFORMERS_USE_MULTIPROCESSING
TRANSFORMERS_VERBOSITY
USE_REAL_DATA
USE_TF
WANDB_DISABLED
WANDB_LOG_MODEL
WANDB_PROJECT
WANDB_WATCH
XDG_CACHE_HOME
```

