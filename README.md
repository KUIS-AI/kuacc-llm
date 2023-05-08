# kuacc-llm
Common LLM setup for the KUACC cluster in Koc University, Istanbul.

tldr: `$ source setenv.sh` to set up your environment and access models/datasets without downloading them.

## Downloaded resources:

* huggingface/datasets: bookcorpus ++
* huggingface/datasets: enwik8 ++
* huggingface/datasets: openwebtext ==
* huggingface/datasets: ptb_text_only ++
* huggingface/datasets: tiny_shakespeare ++
* huggingface/datasets: wikitext ++
* huggingface/transformers: gpt2 ++
* huggingface/transformers: gpt2-large ==
* huggingface/transformers: gpt2-medium ==
* huggingface/transformers: gpt2-xl ==
* huggingface/transformers: openai-gpt ++
* huggingface/transformers: CarperAI/stable-vicuna-13b-delta (depends on llama-13b)
* huggingface/transformers: bigscience/bloom (176B) 
* huggingface/transformers: bigscience/bloom-1b1 
* huggingface/transformers: bigscience/bloom-1b7 
* huggingface/transformers: bigscience/bloom-3b 
* huggingface/transformers: bigscience/bloom-560m ++
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
* https://github.com/gururise/AlpacaDataCleaned 
* https://github.com/EleutherAI/lm-evaluation-harness
* huggingface/datasets: tatsu-lab/alpaca ++ (the original)
* huggingface/datasets: yahma/alpaca-cleaned ++ (https://github.com/gururise/AlpacaDataCleaned as of 2023-04-10)
* huggingface/transformers: samwit/alpaca7B-lora  (original alpaca: gururise refers to it but don't know how to download)
* huggingface/transformers: tloen/alpaca-lora-7b  (trained with yahma/alpaca-cleaned 2023-03-26: gururise refers to it but don't know how to download)
* huggingface/transformers: lmsys/vicuna-7b-delta-v0  
* huggingface/transformers: lmsys/vicuna-7b-delta-v1.1 
* huggingface/transformers: lmsys/vicuna-13b-delta-v1.1 (9GB)
* huggingface/transformers: lmsys/vicuna-13b-delta-v0 
* huggingface/transformers: lmsys/fastchat-t5-3b-v1.0 
* huggingface/transformers: nomic-ai/gpt4all-j 
* huggingface/transformers: nomic-ai/gpt4all-13b-snoozy 
* huggingface/transformers: nomic-ai/gpt4all-lora-epoch-3 
* huggingface/transformers: nomic-ai/gpt4all-lora  (gururise refers to it but don't know how to download, seems worse than tloen/alpaca-lora-7b)
* huggingface/transformers: nomic-ai/gpt4all-j-lora 
* huggingface/transformers: cerebras/Cerebras-GPT-13B 
* huggingface/transformers: cerebras/Cerebras-GPT-6.7B 
* huggingface/transformers: cerebras/Cerebras-GPT-2.7B 
* huggingface/transformers: cerebras/Cerebras-GPT-1.3B 
* huggingface/transformers: cerebras/Cerebras-GPT-590M 
* huggingface/transformers: cerebras/Cerebras-GPT-256M 
* huggingface/transformers: cerebras/Cerebras-GPT-111M 
* huggingface/transformers: EleutherAI/pythia-XXX  (open source alternative to Llama?)
* https://github.com/ZrrSkywalker/LLaMA-Adapter
* https://github.com/lm-sys/FastChat#vicuna-weights
* https://github.com/imaurer/awesome-decentralized-llm
* https://arxiv.org/abs/2210.17323 (GPTQ quantized)
* https://bair.berkeley.edu/blog/2023/04/03/koala/
* https://huggingface.co/young-geng/koala
* https://huggingface.co/OpenAssistant (training set and fine tuned models based on pythia and Llama)
* https://www.semianalysis.com/p/google-we-have-no-moat-and-neither


## stability.ai

## huggingface.hub

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

