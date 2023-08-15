# kuacc-llm
Common LLM setup for the KUACC cluster in Koc University, Istanbul.

tldr: `$ source /datasets/NLP/setenv.sh` to set up your environment and access models/datasets without downloading them.

This currently sets the TORCH_HOME and HF_HOME and directs the following commands to use the read-only cache under /datasets/NLP:
* Here is some info on the huggingface cache structure: https://huggingface.co/docs/huggingface_hub/guides/manage-cache

```
import transformers, datasets, torchvision
transformers.AutoModelForCausalLM.from_pretrained("gpt2")
datasets.load_dataset("tiny_shakespeare")
torchvision.get_model("resnet50", weights="DEFAULT")
```

To build the python environment to use these models and datasets use:
```
$ conda env create -f environment.yml
# OR
$ conda create --name llm --file spec-file.txt
```

To generate text:
```
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])
```

Alternative method to generate text:
```
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
generator("Hello, I'm a language model,")
```

To investigate weights:
```
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
for n,p in model.named_parameters():
  print((n,p.shape))

('transformer.wte.weight', torch.Size([50257, 768]))
('transformer.wpe.weight', torch.Size([1024, 768]))
...
```

To use a model with lower precision (32, 16, 8, 4 bit): For setup and tips see:
* https://huggingface.co/blog/hf-bitsandbytes-integration (basic theory)
* https://huggingface.co/blog/4bit-transformers-bitsandbytes (4-bit types, NF4 etc)
* https://huggingface.co/docs/transformers/main_classes/quantization (advanced options)
* https://huggingface.co/docs/transformers/perf_infer_gpu_one (efficient inference on single gpu)
* https://huggingface.co/docs/accelerate/usage_guides/big_modeling (handling big models for inference)

```
from transformers import AutoModelForCausalLM
m = "facebook/opt-350m"  # gpt2 is not supported with 4/8 bit
AutoModelForCausalLM.from_pretrained(m)  # fp32, defaults to cpu
AutoModelForCausalLM.from_pretrained(m, device_map="auto")     # fp32, gpu if available
AutoModelForCausalLM.from_pretrained(m, device_map="auto", dtype=torch.float16)   # fp16
AutoModelForCausalLM.from_pretrained(m, device_map="auto", dtype=torch.bfloat16)  # bf16, better with overflows
AutoModelForCausalLM.from_pretrained(m, device_map="auto", load_in_8bit=True)
AutoModelForCausalLM.from_pretrained(m, device_map="auto", load_in_4bit=True)
```

To use a llm+adapter:
```
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import snapshot_download

snapshot_download(repo_id="esokullu/llama-13B-ft-sokullu-lora")
base_model_name_or_path = "huggyllama/llama-13b"
lora_model_name_or_path = "/datasets/NLP/huggingface/hub/models--esokullu--llama-13B-ft-sokullu-lora/snapshots/542c2f91183ac5bc5ed13d5130161b11b7bcc9b8" # "sokullu/sokullu-lora-13b"
tokenizer = LlamaTokenizer.from_pretrained(base_model_name_or_path)
model = LlamaForCausalLM.from_pretrained(base_model_name_or_path, load_in_8bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, lora_model_name_or_path)
```

To investigate activations:
```
TODO
```



## Downloaded resources:

### huggingface/transformers

* Aeala/GPT4-x-AlpacaDente2-30b (61G) (from open_llm_leaderboard)
* ai-forever/mGPT (3.3G)
* aisquared/chopt-* (125m (240M), 350m (635M), 1_3b (2.5G), 2_7b (5.0G)) 
* aisquared/chopt-research-* (125m (240M), 350m (635M), 1_3b (2.5G), 2_7b (5.0G)) 
* aisquared/dlite-v2-* (124m (253M), 355m (704M), 774m (1.5G), 1_5b (3.0G)) (from open-llms, open_llm_leaderboard, lightweight gpt-2 based, finetuned)
* ausboss/llama-30b-supercot (61G) (needs 128G, out-of-memory error with 64G, high on open_llm_leaderboard)
* bigscience/bloom-560m (1.1G), bigscience/bloomz-3b (5.7G)
* CarperAI/stable-vicuna-13b-delta (25G) (from open_llm_leaderboard)
* cerebras/Cerebras-GPT-* (111M (467M), 256M (1.1G), 590M (2.3G), 1.3B (5.1G), 2.7B (11G), 6.7B (26G), 13B (49G)) (from open-llms, open_llm_leaderboard)
* chainyo/alpaca-lora-7b (13G) (from open_llm_leaderboard)
* chavinlo/gpt4-x-alpaca (49G) (from open_llm_leaderboard)
* databricks/dolly-* (v1-6b (12G), v2-3b (5.4G), v2-7b (13G), v2-12b (23G)) (from open-llms, open_llm_leaderboard)
* decapoda-research/llama-* (7b-hf (13G), 13b-hf (37G), 30b-hf (77G)) AutoConfig:ok, AutoTokenizer:wrong-name-error, AutoModel:ok
* digitous/Alpacino30b (61G) (from open_llm_leaderboard)
* eachadea/vicuna-* (7b-1.1 (13G), 13b (25G)) (from open_llm_leaderboard)
* ehartford/Wizard-Vicuna-* (7B-Uncensored (26G), 13B-Uncensored (49G)) (see https://t.co/9vrPyktaIz)
* ehartford/WizardLM-* (7B-Uncensored (13G), 13B-Uncensored (25G), 30B-Uncensored (61G)) (see https://t.co/9vrPyktaIz)
* EleutherAI/gpt-* (j-6b (23G), neo-125m (505M), neox-20b (39G)) (from open-llms, open_llm_leaderboard)
* EleutherAI/pythia-* (70m (160M), 160m (360M), 410m (873M), 1b (2.0G), 1.4b (2.8G), 2.8b (5.4G), 6.9b (13G), 12b (23G)) (from open-llms)
* facebook/llama-* (7B (13G), 13B (25G)) (the originals, not an hf repo, to load use e.g. AutoModelForCausalLM.from_pretrained("/datasets/NLP/huggingface/hub/models--facebook--llama-7B"))
* facebook/opt-* (125m (242M), 350m (636M), 1.3b (2.5G), 13b (25G)) (from open_llm_leaderboard)
* facebook/xglm-* (564M (1.1G), 1.7B (3.3G), 2.9B (5.6G), 4.5B (9.6G), 7.5B (15G))
* google/flan-t5-* (small (298M), base (949M), large (3.0G), xl (11G), xxl (43G)) 
* google/flan-ul2 (37G), google/ul2 (37G) 
* h2oai/h2ogpt-oig-oasst1-512-6.9b (13G) (from open-llms)
* hakurei/instruct-12b (45G)
* HuggingFaceH4/starchat-alpha (30G) (from open_llm_leaderboard)
* huggyllama/llama-* (7b (13G), 13b (25G), 30b (61G), 65b (123G)) 
* KoboldAI/OPT-13B-Nerybus-Mix (25G) (from open_llm_leaderboard)
* lamini/instruct-tuned-3b (5.7G)
* lmsys/fastchat-t5-3b-v1.0 (6.3G) 
* lmsys/vicuna-* (7b-delta-v1.1 (13G), 13b-delta-v1.1 (25G), 7b (13G), 13b (25G)) (for 7b/13b use e.g. AutoModelForCausalLM.from_pretrained("/datasets/NLP/huggingface/hub/models--lmsys--vicuna-7b"))
* meta-llama/Llama-2-* (7b-hf (13G), 7b-chat-hf (13G), 13b-hf (25G), 13b-chat-hf (25G), 70b-hf (129G), 70b-chat-hf (129G))
* MetaIX/GPT4-X-Alpasta-30b (61G) (from open_llm_leaderboard)
* mosaicml/mpt-* (1b-redpajama-200b (5.0G), 1b-redpajama-200b-dolly (5.0G), 7b (13G), 7b-chat (13G), 7b-instruct (13G), 7b-storywriter (13G)) (from open-llms, requires einops, trust_remote_code=True, see hf page for details)
* nomic-ai/gpt4all-* (13b-snoozy (49G), j (23G)) (gururise refers to it but don't know how to download -lora, seems worse than tloen/alpaca-lora-7b) (from open_llm_leaderboard)
* openaccess-ai-collective/manticore-13b (25G)
* openai-gpt (461M), gpt2 (528M), gpt2-medium (1.5G), gpt2-large (3.1G), gpt2-xl (6.1G), distilgpt2 (341M) (from open_llm_leaderboard)
* OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 (23G) (from open-llms, open_llm_leaderboard)
* openlm-research/open_llama_7b_preview_300bt (13G) (from open-llms, open_llm_leaderboard, use `AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_7b_preview_300bt", subfolder="open_llama_7b_preview_300bt_transformers_weights")`)
* Pirr/pythia-13b-deduped-green_devil (23G) (from open_llm_leaderboard)
* pythainlp/wangchanglm-7.5B-sft-en-sharded (32G) (from open_llm_leaderboard)
* Salesforce/codegen-16B-multi (31G) (from open_llm_leaderboard)
* stabilityai/stablelm-* (base-alpha-3b (14G), tuned-alpha-3b (14G), tuned-alpha-7b (30G)) (from open-llms, open_llm_leaderboard)
* TheBloke/dromedary-65b-lora-HF (123G) (from open_llm_leaderboard)
* TheBloke/vicuna-13B-1.1-HF (25G) (from open_llm_leaderboard)
* TheBloke/wizardLM-7B-HF (13G)
* tiiuae/falcon-* (rw-1b (2.5G), rw-7b (15G), 7b (14G), 7b-instruct (14G), 40b (79G), 40b-instruct (79G))
* togethercomputer/GPT-* (JT-6B-v0 (12G), JT-Moderation-6B (12G), NeoXT-Chat-Base-20B (39G)) 
* togethercomputer/Pythia-Chat-Base-7B (13G) 
* togethercomputer/RedPajama-INCITE-* (Base-3B-v1 (5.4G), Base-7B-v0.1 (13G), Chat-3B-v1 (5.4G), Chat-7B-v0.1 (13G), Instruct-3B-v1 (5.4G), Instruct-7B-v0.1 (13G)) (from open-llms, open_llm_leaderboard)
* vicgalle/gpt2-alpaca-gpt4 (492M) (from open_llm_leaderboard)
* wordcab/llama-natural-instructions-13b (37G) (from open_llm_leaderboard)


### downloading

* bigcode/starcoder
* https://huggingface.co/conceptofmind/Flan-Open-Llama-7b
* nomic-ai/gpt4all-lora -- error
* nomic-ai/gpt4all-j-lora -- error
* ehartford/alpaca1337-13b-4bit -- error: OSError: ehartford/alpaca1337-13b-4bit does not appear to have a file named config.json. Checkout 'https://huggingface.co/ehartford/alpaca1337-13b-4bit/main' for available files.
* ehartford/alpaca1337-7b-4bit -- error: OSError: ehartford/alpaca1337-7b-4bit does not appear to have a file named config.json. Checkout 'https://huggingface.co/ehartford/alpaca1337-7b-4bit/main' for available files.
n
* laion/OIG () (from emirhan) -- error: Generating train split: 14113288 examples [36:34, 4918.58 examples/s]Failed to read file '/datasets/NLP/huggingface/datasets/downloads/extracted/13d1404eac66ab41c857612e073018ab83a1dcd1293cc32464dead7b4ce933ba' with error <class 'pyarrow.lib.ArrowInvalid'>: JSON parse error: Missing a comma or '}' after an object member. in row 10
* EleutherAI/pile () -- error: json.decoder.JSONDecodeError: Unterminated string starting at: line 1 column 10 (char 9)
* gsm8k (4.6M) (from emirhan) -- downloaded 'main', 'socratic', but gives error with load_dataset: FileNotFoundError: Unable to resolve any data file that matches '['train[-._ 0-9/]**', ...
* Hello-SimpleAI/HC3 (from emirhan) -- AttributeError: 'NoneType' object has no attribute 'name'


### huggingface/datasets

* allenai/prosocial-dialog (92M) (from emirhan)
* amazon_reviews_multi (368M) (from emirhan)
* big_patent (40G) (from emirhan)
* billsum (261M) (from emirhan)
* bookcorpus (4.6G) 
* ccdv/cnn_dailymail (1.3G) (from emirhan)
* checkai/instruction-poems (27M) (from emirhan)
* databricks/databricks-dolly-15k (12M) (from emirhan)
* dctanner/oa_recipes (7.4M) (from emirhan)
* donfu/oa-stackexchange (6.2G) (from emirhan)
* ehartford/oa_leet10k (46M) (from emirhan)
* emozilla/soda_synthetic_dialogue (1.8G) (from emirhan)
* enwik8 (99M) 
* glue (232M) 
* imdb (128M) 
* MBZUAI/LaMini-instruction (1.1G) (2M chatGPT outputs for different prompts, from emirhan)
* mikegarts/oa_tell_a_joke_10000 (5.9G) (from emirhan)
* mosaicml/dolly_hhrlhf (46M)
* multi_news (668M) (from emirhan)
* nomic-ai/gpt4all-j-prompt-generations (1.7G) (from emirhan)
* OllieStanley/humaneval-mbpp-codegen-qa (244K) (from emirhan)
* OllieStanley/humaneval-mbpp-testgen-qa (320K) (from emirhan)
* OllieStanley/oa_camel (227M) (from emirhan)
* openwebtext (38G) 
* piqa (5.2M) 
* ptb_text_only (5.8M) 
* samsum (11M) (from emirhan)
* sciq (7.4M) 
* squad (87M) 
* super_glue (285M) 
* tatsu-lab/alpaca (45M) (the original)
* tiny_shakespeare (1.2M) 
* totuta/youtube_subs_howto100M (1.2G) (from emirhan)
* victor123/evol_instruct_70k (126M) (from emirhan)
* wikitext (1.1G) 
* xsum (510M) (from emirhan)
* yahma/alpaca-cleaned (39M) (https://github.com/gururise/AlpacaDataCleaned as of 2023-04-10)


### torchvision.models

* all 121 models listed in torchvision.models.list_models() (25G):

['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet101', 'deeplabv3_resnet50', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fcn_resnet101', 'fcn_resnet50', 'fcos_resnet50_fpn', 'googlenet', 'inception_v3', 'keypointrcnn_resnet50_fpn', 'lraspp_mobilenet_v3_large', 'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 'maxvit_t', 'mc3_18', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mvit_v1_b', 'mvit_v2_s', 'quantized_googlenet', 'quantized_inception_v3', 'quantized_mobilenet_v2', 'quantized_mobilenet_v3_large', 'quantized_resnet18', 'quantized_resnet50', 'quantized_resnext101_32x8d', 'quantized_resnext101_64x4d', 'quantized_shufflenet_v2_x0_5', 'quantized_shufflenet_v2_x1_0', 'quantized_shufflenet_v2_x1_5', 'quantized_shufflenet_v2_x2_0', 'r2plus1d_18', 'r3d_18', 'raft_large', 'raft_small', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 's3d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 'swin3d_b', 'swin3d_s', 'swin3d_t', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']


## List of companies/users with sota models/datasets we should follow:

* https://huggingface.co/aisquared (dlite)
* https://huggingface.co/bigscience (bloom)
* https://huggingface.co/CarperAI (stable-vicuna)
* https://huggingface.co/cerebras (cerebras-gpt)
* https://huggingface.co/databricks (dolly)
* https://huggingface.co/EleutherAI (pythia, gpt-neox)
* https://huggingface.co/facebook (opt)
* https://huggingface.co/google (flan-t5, flan-ul2, everything (-:)
* https://huggingface.co/h2oai (h2ogpt)
* https://huggingface.co/lmsys (fastchat, vicuna)
* https://huggingface.co/mosaicml (mpt)
* https://huggingface.co/nomic-ai (gpt4all)
* https://huggingface.co/OpenAssistant (openassistant)
* https://huggingface.co/stabilityai (stablelm)
* https://huggingface.co/togethercomputer (redpajama)
* https://huggingface.co/young-geng (koala)


## Model Lists and Evaluation

* https://crfm.stanford.edu/helm/latest/?models=1
* https://github.com/EleutherAI/lm-evaluation-harness
* https://github.com/eugeneyan/open-llms
* https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
* https://chat.lmsys.org/?arena
* https://lmsys.org/blog/2023-05-03-arena/


## Some model sizes

| model         | layer | embd | nhead | vocab | nctx | params         |
| -----         | ----- | ---- | ----- | ----- | ---- | ------         |
| openai-gpt    | 12    | 768  | 12    | 40478 | 512  |    116_534_784 |
| gpt2          | 12    | 768  | 12    | 50257 | 1024 |    124_439_808 |
| gpt2-medium   | 24    | 1024 | 16    | 50257 | 1024 |    354_823_168 |
| gpt2-large    | 36    | 1280 | 20    | 50257 | 1024 |    774_030_080 |
| gpt2-xl       | 48    | 1600 | 25    | 50257 | 1024 |  1_557_611_200 | 
| pythia-70m    |  6    | 512  |  8    | 50304 | 2048 |     70_426_624 |
| pythia-160m   | 12    | 768  | 12    | 50304 | 2048 |    162_322_944 |
| pythia-410m   | 24    | 1024 | 16    | 50304 | 2048 |    405_334_016 |
| pythia-1b     | 16    | 2048 |  8    | 50304 | 2048 |  1_011_781_632 |
| pythia-1.4b   | 24    | 2048 | 16    | 50304 | 2048 |  1_414_647_808 |
| pythia-2.8b   | 32    | 2560 | 32    | 50304 | 2048 |  2_775_208_960 |
| pythia-6.9b   | 32    | 4096 | 32    | 50432 | 2048 |  6_857_302_016 |
| pythia-12b    | 36    | 5120 | 40    | 50688 | 2048 | 11_846_072_320 |
| chopt-125m    | 12    | 768  | 12    | 50268 | 2048 |    125_236_224 |
| chopt-350m    | 24    | 1024 | 16    | 50268 | 2048 |    331_194_368 |
| chopt-1_3b    | 24    | 2048 | 32    | 50268 | 2048 |  1_315_749_888 |
| chopt-2_7b    | 32    | 2560 | 32    | 50268 | 2048 |  2_651_586_560 |
| dlite-v2-124m | 12    | 768  | 12    | 50260 | 1024 |    124_442_112 |
| dlite-v2-355m | 24    | 1024 | 16    | 50260 | 1024 |    354_826_240 |
| dlite-v2-774m | 36    | 1280 | 20    | 50260 | 1024 |    774_033_920 |
| dlite-v2-1_5b | 48    | 1600 | 25    | 50260 | 1024 |  1_557_616_000 |
