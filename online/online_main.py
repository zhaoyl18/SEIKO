import torch
from PIL import Image
import sys
import os
import copy
import gc
cwd = os.getcwd()
sys.path.append(cwd)

from tqdm import tqdm
import random
from collections import defaultdict
import prompts as prompts_file
import torch.distributed as dist
import numpy as np
import torch.utils.checkpoint as checkpoint
import wandb
import contextlib
import torchvision
from transformers import AutoProcessor, AutoModel
import sys
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import datetime

from accelerate.logging import get_logger    
from accelerate import Accelerator
from absl import app, flags
from ml_collections import config_flags
import time

from diffusers_patch.ddim_with_kl import ddim_step_KL
from online.model_utils import generate_embeds_fn, evaluate_loss_fn, evaluate, prepare_pipeline, generate_new_x, online_aesthetic_loss_fn
from online.dataset import D_explored

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/online.py:aesthetic", "Training configuration.")

from accelerate.utils import set_seed, ProjectConfiguration
logger = get_logger(__name__)
    


def main(_):
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
        
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )


    if accelerator.is_main_process:
        wandb_args = {}
        wandb_args["name"] = config.run_name
        if config.debug:
            wandb_args.update({'mode':"disabled"})        
        accelerator.init_trackers(
            project_name="Online", config=config.to_dict(), init_kwargs={"wandb": wandb_args}
        )

        accelerator.project_configuration.project_dir = os.path.join(config.logdir, config.run_name)
        accelerator.project_configuration.logging_dir = os.path.join(config.logdir, wandb.run.name)    


    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    if config.pretrained.model.endswith(".safetensors") or config.pretrained.model.endswith(".ckpt"):
        pipeline = StableDiffusionPipeline.from_single_file(config.pretrained.model)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)

    # freeze parameters of models to save more memory
    inference_dtype = torch.float32

    unet_list, Unet2d_models = prepare_pipeline(pipeline, accelerator, config, inference_dtype)
    
    embedding_fn = generate_embeds_fn(device = accelerator.device, torch_dtype = inference_dtype)    
    
    online_loss_fn = online_aesthetic_loss_fn(grad_scale=config.grad_scale,
                                    aesthetic_target=config.aesthetic_target,
                                    config=config,
                                    accelerator = accelerator,
                                    torch_dtype = inference_dtype,
                                    device = accelerator.device)
    
    eval_loss_fn = evaluate_loss_fn(grad_scale=config.grad_scale,
                                aesthetic_target=config.aesthetic_target,
                                accelerator = accelerator,
                                torch_dtype = inference_dtype,
                                device = accelerator.device)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True 

    prompt_fn = getattr(prompts_file, config.prompt_fn)
    samping_prompt_fn = getattr(prompts_file, config.samping_prompt_fn)

    if config.eval_prompt_fn == '':
        eval_prompt_fn = prompt_fn
    else:
        eval_prompt_fn = getattr(prompts_file, config.eval_prompt_fn)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
            pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
        )[0]

    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size_per_gpu_available, 1, 1)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size_per_gpu_available, 1, 1)

    autocast = contextlib.nullcontext          
    #################### TRAINING ####################        

    num_fresh_samples = config.num_samples  # 64 samples take 4 minutes to generate
    assert len(num_fresh_samples) == config.train.num_outer_loop, "Number of outer loops must match the number of data counts"
    
    exp_dataset = D_explored(config, accelerator.device).to(accelerator.device, dtype=inference_dtype)
    exp_dataset.model = accelerator.prepare(exp_dataset.model)

    global_step = 0
    for outer_loop in range(config.train.num_outer_loop):        
        ##### Generate a new sample x(i) ∼ p(i)(x) by running {p(i) and get a feedback y(i) =r(x(i)) + ε.
        current_unet = unet_list[outer_loop]
        training_unet = unet_list[outer_loop+1]
        num_new_x = num_fresh_samples[outer_loop]
        print(num_new_x)

        current_unet.eval()
        
        # Freeze the parameter of current model
        if outer_loop == 0:
            for param in current_unet.parameters():
                param.requires_grad = False
        else:
            for name, attn_processor in current_unet.named_children():
                for param in attn_processor.parameters():
                    param.requires_grad = False
        logger.info(f"Freezing current model: {outer_loop}")
        logger.info(f"Start training model: {outer_loop+1}")
        
        if outer_loop > 0: # load the previous model to the training model
            logger.info(f"Load previous model: {outer_loop} weight to training model: {outer_loop+1}")
            training_unet.load_state_dict(current_unet.state_dict())
        
        for name, attn_processor in training_unet.named_children():
                for param in attn_processor.parameters():
                    assert param.requires_grad == True, "All LoRA parameters should be trainable"
        
        if outer_loop == 0 and 'restore_initial_data_from' in config.keys():
            logger.info(f"Restore initial data from {config.restore_initial_data_from}")
            all_new_x = torch.load(config.restore_initial_data_from)
            all_new_x = all_new_x.to(accelerator.device)
        
        else:
            new_x = generate_new_x(
                current_unet, 
                num_new_x // config.train.num_gpus, 
                pipeline, 
                accelerator, 
                config, 
                inference_dtype, 
                samping_prompt_fn, 
                sample_neg_prompt_embeds, 
                embedding_fn)  
                 
            all_new_x = accelerator.gather(new_x)  # gather samples and distribute to all GPUs
        
        assert(len(all_new_x) == num_new_x), "Number of fresh online samples does not match the target number" 
        
        ##### Construct a new dataset: D(i) = D(i−1) + (x(i), y(i))
        exp_dataset.update(all_new_x)
        del all_new_x
        
        # Train a pessimistic reward model r(x; D(i)) and a pessimistic bonus term g(i)(x; D(i))
        if config.train.optimism in ['none', 'UCB']:
            exp_dataset.train_MLP(accelerator, config)
        elif config.train.optimism == 'bootstrap':
            exp_dataset.train_bootstrap(accelerator, config)
        else:
            raise ValueError(f"Unknown optimism {config.train.optimism}")
        
        if accelerator.num_processes > 1:
            # sanity check model weight sync
            if config.train.optimism == 'bootstrap':
                print(f"Process {accelerator.process_index} model 0 layer 0 bias: {exp_dataset.model.module.models[0].layers[0].bias.data}")
            else:
                print(f"Process {accelerator.process_index} layer 0 bias: {exp_dataset.model.module.layers[0].bias.data}")
            print(f"Process {accelerator.process_index} x: {exp_dataset.x.shape}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        ##### Update a diffusion model as {p(i)} by finetuning.
        optimizer = torch.optim.AdamW(
            training_unet.parameters(),
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )

        # Prepare everything with our `accelerator`.
        training_unet, optimizer = accelerator.prepare(training_unet, optimizer)
        
        timesteps = pipeline.scheduler.timesteps #[981, 961, 941, 921,]
        
        eval_prompts, eval_prompt_metadata = zip(
            *[eval_prompt_fn() for _ in range(config.train.batch_size_per_gpu_available * config.max_vis_images)]
        )    

        for epoch in list(range(0, config.num_epochs)):
            training_unet.train()
            info = defaultdict(list)
            info_vis = defaultdict(list)
            image_vis_list = []
            
            for inner_iters in tqdm(
                    list(range(config.train.data_loader_iterations)),
                    position=0,
                    disable=not accelerator.is_local_main_process
                ):
                latent = torch.randn((config.train.batch_size_per_gpu_available, 4, 64, 64),
                    device=accelerator.device, dtype=inference_dtype)    

                if accelerator.is_main_process:
                    logger.info(f"{config.run_name.rsplit('/', 1)[0]} Loop={outer_loop}/Epoch={epoch}/Iter={inner_iters}: training")

                prompts, prompt_metadata = zip(
                    *[prompt_fn() for _ in range(config.train.batch_size_per_gpu_available)]
                )

                prompt_ids = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device)   

                pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
                
            
                with accelerator.accumulate(training_unet):
                    with autocast():
                        with torch.enable_grad(): # important b/c don't have on by default in module                        
                            keep_input = True
                            
                            kl_loss = 0
                            
                            for i, t in tqdm(
                                enumerate(timesteps), 
                                total=len(timesteps),
                                disable=not accelerator.is_local_main_process,
                            ):
                                t = torch.tensor([t],
                                        dtype=inference_dtype,
                                        device=latent.device
                                    )
                                t = t.repeat(config.train.batch_size_per_gpu_available)
                                
                                if config.grad_checkpoint:
                                    noise_pred_uncond = checkpoint.checkpoint(training_unet, latent, t, train_neg_prompt_embeds, use_reentrant=False).sample
                                    noise_pred_cond = checkpoint.checkpoint(training_unet, latent, t, prompt_embeds, use_reentrant=False).sample
                                    
                                    old_noise_pred_uncond = checkpoint.checkpoint(current_unet,latent, t, train_neg_prompt_embeds, use_reentrant=False).sample
                                    old_noise_pred_cond = checkpoint.checkpoint(current_unet,latent, t, prompt_embeds, use_reentrant=False).sample
                                    
                                else:
                                    noise_pred_uncond = training_unet(latent, t, train_neg_prompt_embeds).sample
                                    noise_pred_cond = training_unet(latent, t, prompt_embeds).sample
                                
                                    old_noise_pred_uncond = current_unet(latent, t, train_neg_prompt_embeds).sample
                                    old_noise_pred_cond = current_unet(latent, t, prompt_embeds).sample    
                                        
                                if config.truncated_backprop:
                                    if config.truncated_backprop_rand:
                                        timestep = random.randint(
                                            config.truncated_backprop_minmax[0],
                                            config.truncated_backprop_minmax[1]
                                        )
                                        if i < timestep:
                                            noise_pred_uncond = noise_pred_uncond.detach()
                                            noise_pred_cond = noise_pred_cond.detach()
                                            old_noise_pred_uncond = old_noise_pred_uncond.detach()
                                            old_noise_pred_cond = old_noise_pred_cond.detach()
                                    else:
                                        if i < config.trunc_backprop_timestep:
                                            noise_pred_uncond = noise_pred_uncond.detach()
                                            noise_pred_cond = noise_pred_cond.detach()
                                            old_noise_pred_uncond = old_noise_pred_uncond.detach()
                                            old_noise_pred_cond = old_noise_pred_cond.detach()

                                grad = (noise_pred_cond - noise_pred_uncond)
                                old_grad = (old_noise_pred_cond - old_noise_pred_uncond)
                                
                                noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
                                old_noise_pred = old_noise_pred_uncond + config.sd_guidance_scale * old_grad 
                                            
                                # latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
                                
                                latent, kl_terms = ddim_step_KL(
                                    pipeline.scheduler,
                                    noise_pred,   # (2,4,64,64),
                                    old_noise_pred, # (2,4,64,64),
                                    t[0].long(),
                                    latent,
                                    eta=config.sample_eta,  # 1.0
                                )
                                kl_loss += torch.mean(kl_terms).to(inference_dtype)
                                        
                            ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample  # latent entries around -5 - +7
                            
                            loss, rewards = online_loss_fn(ims, config, exp_dataset)
                            loss = loss.mean() * config.train.loss_coeff
                            
                            total_loss = loss + config.train.kl_weight*kl_loss
                            
                            rewards_mean = rewards.mean()
                            rewards_std = rewards.std()
                            
                            if len(info_vis["image"]) < config.max_vis_images:
                                info_vis["image"].append(ims.clone().detach())
                                info_vis["rewards_img"].append(rewards.clone().detach())
                                info_vis["prompts"] = list(info_vis["prompts"]) + list(prompts)
                            
                            info["loss"].append(total_loss)
                            info["KL-entropy"].append(kl_loss)
                            
                            info["rewards"].append(rewards_mean)
                            info["rewards_std"].append(rewards_std)
                            
                            # backward pass
                            accelerator.backward(total_loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(training_unet.parameters(), config.train.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()                        

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    assert (
                        inner_iters + 1
                    ) % config.train.gradient_accumulation_steps == 0
                    # log training and evaluation 
                    if config.visualize_eval and (global_step % config.vis_freq ==0):

                        all_eval_images = []
                        all_eval_rewards = []
                        if config.same_evaluation:
                            generator = torch.cuda.manual_seed(config.seed)
                            latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype, generator=generator)    
                        else:
                            latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)                                
                        with torch.no_grad():
                            for index in range(config.max_vis_images):
                                ims, rewards = evaluate(
                                    training_unet,
                                    latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)],
                                    train_neg_prompt_embeds,
                                    eval_prompts[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)], 
                                    pipeline, 
                                    accelerator, 
                                    inference_dtype,
                                    config, 
                                    eval_loss_fn)
                                
                                all_eval_images.append(ims)
                                all_eval_rewards.append(rewards)
                                
                        eval_rewards = torch.cat(all_eval_rewards)
                        eval_reward_mean = eval_rewards.mean()
                        eval_reward_std = eval_rewards.std()
                        eval_images = torch.cat(all_eval_images)
                        eval_image_vis = []
                        if accelerator.is_main_process:
                            name_val = config.run_name
                            log_dir = f"logs/{name_val}/eval_vis"
                            os.makedirs(log_dir, exist_ok=True)
                            for i, eval_image in enumerate(eval_images):
                                eval_image = (eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                                pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                                prompt = eval_prompts[i]
                                pil.save(f"{log_dir}/{outer_loop:01d}_{epoch:03d}_{inner_iters:03d}_{i:03d}_{prompt}.png")
                                pil = pil.resize((256, 256))
                                reward = eval_rewards[i]
                                eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))                    
                            accelerator.log({"eval_images": eval_image_vis},step=global_step)
                    
                    logger.info("Logging")
                    
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = accelerator.reduce(info, reduction="mean")
                    logger.info(f"loss: {info['loss']}, rewards: {info['rewards']}")

                    info.update({"outer_loop": outer_loop,
                                 "epoch": epoch, 
                                 "inner_epoch": inner_iters,
                                 "eval_rewards":eval_reward_mean,
                                 "eval_rewards_std":eval_reward_std,
                                 "dataset_size": len(exp_dataset),
                                 "dataset_y_avg": torch.mean(exp_dataset.y),
                                 })
                    accelerator.log(info, step=global_step)

                    if config.visualize_train:
                        ims = torch.cat(info_vis["image"])
                        rewards = torch.cat(info_vis["rewards_img"])
                        prompts = info_vis["prompts"]
                        images  = []
                        for i, image in enumerate(ims):
                            image = (image.clone().detach() / 2 + 0.5).clamp(0, 1)
                            pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                            pil = pil.resize((256, 256))
                            prompt = prompts[i]
                            reward = rewards[i]
                            images.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))
                        
                        accelerator.log(
                            {"images": images},
                            step=global_step,
                        )

                    global_step += 1
                    info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients
            
            if epoch % config.save_freq == 0 and accelerator.is_main_process:
                def save_model_hook(models, weights, output_dir):
                    if isinstance(models[-1], AttnProcsLayers):
                        Unet2d_models[outer_loop+1].save_attn_procs(output_dir)
                    else:
                        raise ValueError(f"Unknown model type {type(models[-1])}")
                    for _ in range(len(weights)):
                        weights.pop()
                accelerator.register_save_state_pre_hook(save_model_hook)
                accelerator.save_state()
        
        del optimizer 
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  

if __name__ == "__main__":
    app.run(main)
