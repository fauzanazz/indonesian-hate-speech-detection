---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:11444
- loss:TripletLoss
base_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
widget:
- source_sentence: USER mungut dimana lagi tuh orang..?? Tampang Cebong kok ngaku2
    ulama muda...'
  sentences:
  - Pilkada Serentak, Megawati Hanya Minta Jangan ada Intimidasi dan Politik Uang
    https:\/\/t.co\/zHhvJ4qrrY
  - USER Main taplak meja bukannya?'
  - RT USER USER namanya juga sandirawa para pecundang. Bubarkan ajah USER klo cma
    urus kasus ecek2 g brani urus BLBI RSSW dll
- source_sentence: persiapan 2019 \nsaya berharap cendana mau bantu abis abisan untuk
    hambalang\n\nkalo 2019 gagal\nsemua tamat\n\nkarna si sipit sudah menguasai lebih
    dari 50%\n\ndan akan menjadi \nY\nU\nA\nN'
  sentences:
  - Bang Adian pernah bilang, sebelum berniat Ganti Presiden tentukan dulu siapa lawannya
    Pak Jokowi di 2019.\nbaru adu program.'
  - 'Kopi susu: \n- Kapal Api sachet tanpa gula bungkus kecil diseduh pake air panas\n-
    satu bungkus susu kpbs\n- madu\nCampur semua, terus masukin plastik. Masukkan
    frezzer kurang lebih 1 jam, ampas kopinya beku tanpa perlu nyaring lagi.'''
  - Orang ini akan nyunsep bersama jokowi 2019. URL
- source_sentence: Bukan saja pemilik Abu Tours itu pendonor dana utk 212, ternyata
    pendukung Anies-Sandi. Rakyat DKI seharusnya prihatin gubernur mrk berhubungan
    dgn laknat Abu Tours yg menelan 96.000 korban dgn total penipuan Rp1.8T atau 16.439
    tahun menabung Rp30rb sehari
  sentences:
  - Saya termasuk orang yang tidak percaya sejarah yg ditulis/difilmkan ttg G30S /PKI\nSaat
    rame2nya saya kultwit ttg itu saya block akun2 keluarga cendana yg sebelumnya
    follow saya.\nDan saya tersentuh dg nasib puluhan juta keluarga yg distigma ...\nHUMANIT
  - He he bego bgt... dr tglnya aja udh gk ada di peredaran..
  - Hm menyampah pula tengok live FB Paan Khair***din tu. Tak habis2 pusing pasal
    manifesto PH takde pentingkan agama. Golongan DAP tu ? Cina Kristian hey memang
    la dah namanya DAP bukan PAS. Eh geram pula aku. Lepas tu dok kata motif terbesar
    nak jadikan Sel
- source_sentence: 'USER Nyari monyet ya bang? #findinghyukjae'''
  sentences:
  - Bajingan sekali kok drivernya itu. Sorry nggak pantes disebut driver, karena driver
    sifatnya universal. Kasihan yg lain yg udah kerja bener nanti kena, kita ganti
    selanjuttnya jadi bangsat aja'
  - Harinibpuasa pastu beli nasi ayam berempah tapi lupa cakap kuah asing so saya
    dah boleh expect nasi dia macam mana nanti'
  - Gausah kepedean bilang kamu bs sembuhin, lu punya titit, bukan meki'
- source_sentence: USER USER Hahahaha.... Iya, bener jugak. \n\nDan memang sekolah
    Katolik lebih beragam. Mulai dari anak Pak Kebon smp Pengusaha ada disana. Later
    belakang, suku, agama apapun diterima di sekolah Katolik. Pernah ngalamin dulu
    &amp; ngga berub
  sentences:
  - Kontol sange lagi. Ngaceng lagi'
  - 'RT @USER: Kami tdk akan terprovokasi dgn sentimen dan jualan SARA mu itu, karena
    kami cinta Indonesia #IklanAhokJahat'
  - USER Emang loe siapa cebong ???'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/paraphrase-multilingual-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) <!-- at revision 4328cf26390c98c5e3c738b4460a05b95f4911f5 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False, 'architecture': 'XLMRobertaModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'USER USER Hahahaha.... Iya, bener jugak. \\n\\nDan memang sekolah Katolik lebih beragam. Mulai dari anak Pak Kebon smp Pengusaha ada disana. Later belakang, suku, agama apapun diterima di sekolah Katolik. Pernah ngalamin dulu &amp; ngga berub',
    'RT @USER: Kami tdk akan terprovokasi dgn sentimen dan jualan SARA mu itu, karena kami cinta Indonesia #IklanAhokJahat',
    "USER Emang loe siapa cebong ???'",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.6856, -0.2199],
#         [ 0.6856,  1.0000, -0.1158],
#         [-0.2199, -0.1158,  1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 11,444 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | sentence_2                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 34.16 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 35.11 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 33.58 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                  | sentence_1                                                                                                                                                                  | sentence_2                                                                                                                                                                                                                                   |
  |:--------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>USER kamu silit ardhog\xe2\x99\xa5\xef\xb8\x8f'</code>                                | <code>Apalagi mereka partai koalisi pendukung pemerintah yg juga diketahui sudah menjalin kerjasama politik dengan partai komunis china</code>                              | <code>JIMIN GANTENG BANGET BANGKAI'</code>                                                                                                                                                                                                   |
  | <code>USER Saya dukung 200 % pernyataan Najwa soal KPK. Lebih baik DPR kita bubarkan</code> | <code>USER USER USER Hahaha...\nGue saranin lu sering2 show topeng monyet keliling kampung sama bapaklu...spy gak DONGO...monyet...\xf0\x9f\x98\x83\xf0\x9f\x98\x86'</code> | <code>USER USER USER Hehehe akhirnya.....ternyata pasar kita global jg,tdk terbatas lokal. Akhirnya masi ada aseng dan asing juga. Kita jujur saja!..\xf0\x9f\x98\x86\xf0\x9f\x98\x86\xf0\x9f\x98\x86\xf0\x9f\x98\x86\xf0\x9f\x98\x86</code> |
  | <code>disini sinyal simpati kacrut abissss parah mau nyetriming suliddddd'</code>           | <code>Habib sinting kan anonim. Ga usah di dengerin. Sakit jiwa dia URL</code>                                                                                              | <code>RT USER: Bukan main dah ah! Wkwkwk. URL</code>                                                                                                                                                                                         |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.COSINE",
      "triplet_margin": 0.5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.3966 | 500  | 0.2782        |
| 2.7933 | 1000 | 0.1032        |


### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.9.1+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->