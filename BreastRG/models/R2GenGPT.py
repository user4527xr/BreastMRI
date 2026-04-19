import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from models.models_backup_withbn import mask_vit_small_patch16_224



class R2GenGPT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # -------- Vision encoder --------
        self.visual_encoder = mask_vit_small_patch16_224()

        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                r=args.vis_r,
                lora_alpha=args.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
        elif args.freeze_vm:
            for _, p in self.visual_encoder.named_parameters():
                p.requires_grad = False

        # -------- LLM (from args) --------
        model_path = args.llm_model_name_or_path
        self.prompt = '根据给定的影像学数据，生成乳腺病灶描述及诊断结果'

        self.llama_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0

        if args.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(model_path).half()

        if args.llm_use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
        else:
            for _, p in self.llama_model.named_parameters():
                p.requires_grad = False

        self.embed_tokens = self.llama_model.get_input_embeddings()

        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym

        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0


        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'),weights_only=False)['model']
            print(state_dict)
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')
        
    def encode_img(self, image_dce, image_dwi, image_t2):
        image_embed = self.visual_encoder(image_dce, image_dwi, image_t2)
        inputs_llama = self.llama_proj(image_embed)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long, device=image_dce.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img):
        prompt = f"Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:"
        bs = img_embeds.shape[0]
        p_before, p_after = prompt.split("<ImageHere>")
        p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens  = self.llama_tokenizer(p_after,  return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(bs, -1, -1)
        p_after_embeds  = self.embed_tokens(p_after_tokens.input_ids).expand(bs, -1, -1)

        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def decode(self, output_token):
        if output_token[0] in [0, 1]:
            output_token = output_token[1:]
        text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        text = text.split(self.end_sym)[0].replace("<unk>", "").strip()
        return text

    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['report'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image_dce, image_dwi, image_t2 = samples["sub"], samples["dwi"], samples["t2"]
        img_embeds, atts_img = self.encode_img(image_dce, image_dwi, image_t2)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        bs = img_embeds.shape[0]
        bos = torch.ones([bs, 1], dtype=atts_img.dtype, device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )

        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])
        
        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        
        # Use tokenizer to calculate BLEU scores
        eval_res = {}
        
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        
        bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
        smooth = SmoothingFunction().method4
        
        for ref_text, hypo_text in zip(ref.values(), hypo.values()):
            if not ref_text or not hypo_text:
                continue
                
            ref_tokens = self.llama_tokenizer.tokenize(ref_text[0])
            hypo_tokens = self.llama_tokenizer.tokenize(hypo_text[0])
            
            bleu1 = sentence_bleu([ref_tokens], hypo_tokens, weights=[1.0, 0, 0, 0], smoothing_function=smooth)
            bleu2 = sentence_bleu([ref_tokens], hypo_tokens, weights=[0.5, 0.5, 0, 0], smoothing_function=smooth)
            bleu3 = sentence_bleu([ref_tokens], hypo_tokens, weights=[0.33, 0.33, 0.33, 0], smoothing_function=smooth)
            bleu4 = sentence_bleu([ref_tokens], hypo_tokens, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=smooth)
            
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)
        
        if bleu1_scores:
            eval_res['bleu1'] = sum(bleu1_scores) / len(bleu1_scores)
            eval_res['bleu2'] = sum(bleu2_scores) / len(bleu2_scores)
            eval_res['bleu3'] = sum(bleu3_scores) / len(bleu3_scores)
            eval_res['bleu4'] = sum(bleu4_scores) / len(bleu4_scores)
        
        # Use tokenizer to calculate ROUGE
        from rouge import Rouge
        
        rouge = Rouge()
        ref_texts = [text[0] for text in ref.values() if text]
        hypo_texts = [text[0] for text in hypo.values() if text]
        
        if ref_texts and hypo_texts:
            scores = rouge.get_scores(hypo_texts, ref_texts, avg=True)
            eval_res['rouge_l'] = scores['rouge-l']['f']
        
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}.json"), 'w', encoding='utf-8'), ensure_ascii=False)
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w', encoding='utf-8'), ensure_ascii=False)

        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            if score_type in eval_res:
                val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['report'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image_dce = samples['sub']
        image_dwi = samples['dwi']
        image_t2 = samples['t2']
        img_embeds, atts_img = self.encode_img(image_dce, image_dwi, image_t2)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_test_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        
        eval_res = {}
        
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge import Rouge
        
        smooth = SmoothingFunction().method4
        rouge = Rouge()
        
        bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
        
        for ref_text, hypo_text in zip(ref.values(), hypo.values()):
            if not ref_text or not hypo_text:
                continue
                
            ref_tokens = self.llama_tokenizer.tokenize(ref_text[0])
            hypo_tokens = self.llama_tokenizer.tokenize(hypo_text[0])
            
            bleu1 = sentence_bleu([ref_tokens], hypo_tokens, weights=[1.0, 0, 0, 0], smoothing_function=smooth)
            bleu2 = sentence_bleu([ref_tokens], hypo_tokens, weights=[0.5, 0.5, 0, 0], smoothing_function=smooth)
            bleu3 = sentence_bleu([ref_tokens], hypo_tokens, weights=[0.33, 0.33, 0.33, 0], smoothing_function=smooth)
            bleu4 = sentence_bleu([ref_tokens], hypo_tokens, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=smooth)
            
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)
        
        if bleu1_scores:
            eval_res['bleu1'] = sum(bleu1_scores) / len(bleu1_scores)
            eval_res['bleu2'] = sum(bleu2_scores) / len(bleu2_scores)
            eval_res['bleu3'] = sum(bleu3_scores) / len(bleu3_scores)
            eval_res['bleu4'] = sum(bleu4_scores) / len(bleu4_scores)
        
        ref_texts = [text[0] for text in ref.values() if text]
        hypo_texts = [text[0] for text in hypo.values() if text]
        
        if ref_texts and hypo_texts:
            scores = rouge.get_scores(hypo_texts, ref_texts, avg=True)
            eval_res['rouge_l'] = scores['rouge-l']['f']
        
        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w', encoding='utf-8'), ensure_ascii=False)
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w', encoding='utf-8'), ensure_ascii=False)
        
        self.print(f"Test result: {eval_res}")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
    
    def save_checkpoint(self, eval_res):
        checkpoint_path = os.path.join(
            self.hparams.savedmodel_path,
            f"best_model_epoch{self.trainer.current_epoch}_score{self.val_score:.4f}.ckpt"
        )
        torch.save({
            'state_dict': self.state_dict(),
            'hyper_parameters': self.hparams,
            'val_score': self.val_score,
            'eval_res': eval_res,
        }, checkpoint_path)
        self.print(f"Saved best model to {checkpoint_path}")
