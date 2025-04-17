import gradio as gr
import json
import os
import subprocess
import sys # <-- Add import
from .common_gui import ( # Keep existing imports
    # ... (existing imports from common_gui) ...
    get_folder_path,
    get_file_path,
    get_save_folder_path,
)
from .class_gui_config import KohyaSSGUIConfig
from .custom_logging import setup_logging

# Setup logging
log = setup_logging()

# Keras imports if used (keep them)
# ...

# Default parameters (keep them)
# ...

folder_symbol = "\U0001f4c2"  # 📂
refresh_symbol = "\U0001f504"  # 🔄
save_symbol = "\U0001f4be"  # 💾
load_symbol = "\U0001f4c1"  # 📁
delete_symbol = "\U0001f5d1"  # 🗑️


# Define the location of the sd-scripts folder relative to the kohya_ss folder
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'sd-scripts')


# Function to save config (keep existing)
# ... save_configuration(...) ...

# Function to load config (keep existing)
# ... load_configuration(...) ...

# Function to open config file (keep existing)
# ... open_config_file(...) ...

# Function to open folder (keep existing)
# ... open_folder(...) ...

# Function to get file path (keep existing)
# ... get_file_path_func(...) ...

# Function to get folder path (keep existing)
# ... get_folder_path_func(...) ...

# Function to UI elements for config file (keep existing)
# ... config_file_ui(...) ...


# THE KEY FUNCTION TO MODIFY
# NOTE: This function needs modification to correctly launch the subprocess
def train_model(
    headless,
    print_only,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_accumulation_steps,
    max_token_length,
    save_model_as,
    min_bucket_reso,
    max_bucket_reso,
    keep_tokens,
    shuffle_caption,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    lr_scheduler_args,
    max_train_steps,
    max_data_loader_n_workers,
    max_grad_norm,
    network_weights,
    network_module,
    network_dim,
    network_alpha,
    network_dropout,
    # network_args, # This might be needed depending on how args are passed
    unet_lr,
    text_encoder_lr,
    network_train_unet_only,
    network_train_text_encoder_only,
    training_comment,
    stop_text_encoder_training,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    ip_noise_gamma,
    ip_noise_gamma_random_strength,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,
    scale_v_pred_loss_like_noise_pred,
    weighted_captions,
    save_state,
    resume,
    save_every_n_steps,
    use_wandb,
    wandb_api_key,
    log_tracker_name,
    log_tracker_config,
    async_upload,
    config_file_path,
):
    # Check for printing option
    if print_only:
        print(
            'Calling python sd-scripts/train_network.py...'
        )  # This is just a print, not the actual call yet
    else:
        log.info('Start training LoRA Standard...')

    # Get the repo root and resolve paths
    current_gui_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(current_gui_dir, '..'))
    log.info(f"Repo root determined as: {repo_root}")

    # Construct the absolute path to the accelerate executable
    python_executable = sys.executable
    venv_bin_path = os.path.dirname(python_executable)
    accelerate_executable = os.path.join(venv_bin_path, 'accelerate')
    log.info(f"Using accelerate: {accelerate_executable}")

    # Ensure accelerate executable exists
    if not os.path.exists(accelerate_executable):
        log.error(f"Accelerate executable not found at {accelerate_executable}")
        gr.Warning(f"Accelerate executable not found at {accelerate_executable}")
        return # Stop execution

    # Construct the absolute path to the accelerate config file
    # Use the MPS specific one we created
    config_file = os.path.join(repo_root, "accelerate_config_mps.yaml")
    log.info(f"Using config file: {config_file}")
    if not os.path.exists(config_file):
        # Use a default config or error out
        log.warning(f"Config file {config_file} not found. Consider creating it or checking install steps.")
        # Let accelerate handle default config for now
        # Alternatively, you could error here:
        # gr.Warning(f"Accelerate config file {config_file} not found!")
        # return

    # Construct the absolute path to the training script
    train_script_relative = os.path.join("sd-scripts", "train_network.py")
    train_script_absolute = os.path.join(repo_root, train_script_relative)
    log.info(f"Using train script: {train_script_absolute}")
    if not os.path.exists(train_script_absolute):
        log.error(f"Train script not found at {train_script_absolute}")
        gr.Warning(f"Train script not found at {train_script_absolute}")
        return # Stop execution

    # Build the command list for accelerate launch
    run_cmd_list = [
        accelerate_executable,
        'launch',
        '--config_file', config_file,
        # Add other accelerate launch args if necessary, e.g.,
        '--num_cpu_threads_per_process', str(num_cpu_threads_per_process),
        train_script_absolute # Use absolute path to script
    ]

    # Add arguments from the GUI to the list
    # Use a helper function or build the list carefully here
    # Example (ensure variable names match function parameters):
    run_cmd_list.append("--pretrained_model_name_or_path")
    run_cmd_list.append(pretrained_model_name_or_path)
    if v2: run_cmd_list.append("--v2")
    if v_parameterization: run_cmd_list.append("--v_parameterization")
    if sdxl: run_cmd_list.append("--sdxl")
    if logging_dir:
        run_cmd_list.append("--logging_dir")
        run_cmd_list.append(logging_dir)
    run_cmd_list.append("--train_data_dir")
    run_cmd_list.append(train_data_dir)
    if reg_data_dir:
        run_cmd_list.append("--reg_data_dir")
        run_cmd_list.append(reg_data_dir)
    run_cmd_list.append("--output_dir")
    run_cmd_list.append(output_dir)
    if max_resolution: # Example: Check if value is provided
        run_cmd_list.append("--resolution")
        run_cmd_list.append(max_resolution)
    if learning_rate:
        run_cmd_list.append("--learning_rate")
        run_cmd_list.append(str(learning_rate))
    if lr_scheduler:
        run_cmd_list.append("--lr_scheduler")
        run_cmd_list.append(lr_scheduler)
    if lr_warmup:
         run_cmd_list.append("--lr_warmup_steps") # Note: Script uses steps, GUI might use ratio
         # Simple heuristic: if < 1 treat as ratio of steps, else absolute steps
         # This might need refinement based on GUI logic
         try:
            lr_warmup_val = float(lr_warmup)
            if lr_warmup_val < 1.0 and max_train_steps: # Need max_train_steps to calculate from ratio
                 warmup_steps = int(lr_warmup_val * float(max_train_steps))
            else:
                 warmup_steps = int(lr_warmup_val)
            run_cmd_list.append(str(warmup_steps))
            log.info(f"LR warmup steps calculated/set to: {warmup_steps}")
         except ValueError:
             log.error(f"Invalid LR warmup value: {lr_warmup}")
             # Handle error or skip argument

    run_cmd_list.append("--train_batch_size")
    run_cmd_list.append(str(train_batch_size))
    if epoch: # Script uses max_train_epochs
        run_cmd_list.append("--max_train_epochs")
        run_cmd_list.append(str(epoch))
    if save_every_n_epochs:
        run_cmd_list.append("--save_every_n_epochs")
        run_cmd_list.append(str(save_every_n_epochs))
    if mixed_precision: # Should match accelerate config usually
        run_cmd_list.append("--mixed_precision")
        run_cmd_list.append(mixed_precision)
    if save_precision:
        run_cmd_list.append("--save_precision")
        run_cmd_list.append(save_precision)
    if seed:
        run_cmd_list.append("--seed")
        run_cmd_list.append(str(seed))
    if cache_latents: run_cmd_list.append("--cache_latents")
    if cache_latents_to_disk: run_cmd_list.append("--cache_latents_to_disk")
    if caption_extension:
        run_cmd_list.append("--caption_extension")
        run_cmd_list.append(caption_extension)
    if enable_bucket: run_cmd_list.append("--enable_bucket")
    if gradient_accumulation_steps > 1: # Only include if > 1
        run_cmd_list.append("--gradient_accumulation_steps")
        run_cmd_list.append(str(gradient_accumulation_steps))
    if max_token_length:
        run_cmd_list.append("--max_token_length")
        run_cmd_list.append(str(max_token_length))
    if save_model_as:
        run_cmd_list.append("--save_model_as")
        run_cmd_list.append(save_model_as)
    if min_bucket_reso:
        run_cmd_list.append("--min_bucket_reso")
        run_cmd_list.append(str(min_bucket_reso))
    if max_bucket_reso:
        run_cmd_list.append("--max_bucket_reso")
        run_cmd_list.append(str(max_bucket_reso))
    if keep_tokens:
        run_cmd_list.append("--keep_tokens")
        run_cmd_list.append(str(keep_tokens))
    if shuffle_caption: run_cmd_list.append("--shuffle_caption")
    if caption_dropout_every_n_epochs:
        run_cmd_list.append("--caption_dropout_every_n_epochs")
        run_cmd_list.append(str(caption_dropout_every_n_epochs))
    if caption_dropout_rate:
        run_cmd_list.append("--caption_dropout_rate")
        run_cmd_list.append(str(caption_dropout_rate))
    if optimizer:
        run_cmd_list.append("--optimizer_type")
        run_cmd_list.append(optimizer)
    if optimizer_args: # Pass as string, script parses it
        run_cmd_list.append("--optimizer_args")
        run_cmd_list.extend(optimizer_args.split()) # Split args string into list elements
        log.info(f"Optimizer args parsed as: {optimizer_args.split()}")
    if lr_scheduler_args: # Pass as string, script parses it
        run_cmd_list.append("--lr_scheduler_args")
        run_cmd_list.extend(lr_scheduler_args.split()) # Split args string into list elements
        log.info(f"LR scheduler args parsed as: {lr_scheduler_args.split()}")
    if max_train_steps:
        run_cmd_list.append("--max_train_steps")
        run_cmd_list.append(str(max_train_steps))
    if max_data_loader_n_workers:
        run_cmd_list.append("--max_data_loader_n_workers")
        run_cmd_list.append(str(max_data_loader_n_workers))
    if float(max_grad_norm) > 0: # Check value before adding
        run_cmd_list.append("--max_grad_norm")
        run_cmd_list.append(str(max_grad_norm))
    if network_weights:
        run_cmd_list.append("--network_weights")
        run_cmd_list.append(network_weights)
    if network_module:
        run_cmd_list.append("--network_module")
        run_cmd_list.append(network_module)
    if network_dim:
        run_cmd_list.append("--network_dim")
        run_cmd_list.append(str(network_dim))
    if network_alpha:
        run_cmd_list.append("--network_alpha")
        run_cmd_list.append(str(network_alpha))
    if network_dropout > 0: # Check value before adding
        run_cmd_list.append("--network_dropout")
        run_cmd_list.append(str(network_dropout))
    # if network_args: # How are network_args passed? Need to replicate
    #    run_cmd_list.append("--network_args")
    #    run_cmd_list.extend(network_args.split()) # Example split
    if unet_lr:
        run_cmd_list.append("--unet_lr")
        run_cmd_list.append(str(unet_lr))
    if text_encoder_lr:
        run_cmd_list.append("--text_encoder_lr")
        run_cmd_list.append(str(text_encoder_lr))
    if network_train_unet_only: run_cmd_list.append("--network_train_unet_only")
    if network_train_text_encoder_only: run_cmd_list.append("--network_train_text_encoder_only")
    if training_comment:
        run_cmd_list.append("--training_comment")
        run_cmd_list.append(training_comment)
    if stop_text_encoder_training > 0: # Check value before adding
         run_cmd_list.append("--stop_text_encoder_training")
         run_cmd_list.append(str(stop_text_encoder_training))
    if noise_offset_type:
        run_cmd_list.append("--noise_offset_type")
        run_cmd_list.append(noise_offset_type)
    if noise_offset > 0: # Check value before adding
         run_cmd_list.append("--noise_offset")
         run_cmd_list.append(str(noise_offset))
    if adaptive_noise_scale != 0: # Check value before adding
         run_cmd_list.append("--adaptive_noise_scale")
         run_cmd_list.append(str(adaptive_noise_scale))
    if multires_noise_iterations > 0: # Check value before adding
         run_cmd_list.append("--multires_noise_iterations")
         run_cmd_list.append(str(multires_noise_iterations))
    if multires_noise_discount > 0: # Check value before adding
         run_cmd_list.append("--multires_noise_discount")
         run_cmd_list.append(str(multires_noise_discount))
    if ip_noise_gamma > 0:
         run_cmd_list.append("--ip_noise_gamma")
         run_cmd_list.append(str(ip_noise_gamma))
    if ip_noise_gamma_random_strength:
         run_cmd_list.append("--ip_noise_gamma_random_strength")
         run_cmd_list.append(str(ip_noise_gamma_random_strength))
    if sample_every_n_steps > 0: # Check value before adding
         run_cmd_list.append("--sample_every_n_steps")
         run_cmd_list.append(str(sample_every_n_steps))
    if sample_every_n_epochs > 0: # Check value before adding
         run_cmd_list.append("--sample_every_n_epochs")
         run_cmd_list.append(str(sample_every_n_epochs))
    if sample_sampler:
        run_cmd_list.append("--sample_sampler")
        run_cmd_list.append(sample_sampler)
    if sample_prompts:
        run_cmd_list.append("--sample_prompts")
        run_cmd_list.append(sample_prompts) # Assume it's already correct format
    if vae_batch_size > 0: # Check value before adding
         run_cmd_list.append("--vae_batch_size")
         run_cmd_list.append(str(vae_batch_size))
    if min_snr_gamma > 0: # Check value before adding
         run_cmd_list.append("--min_snr_gamma")
         run_cmd_list.append(str(min_snr_gamma))
    if scale_v_pred_loss_like_noise_pred: run_cmd_list.append("--scale_v_pred_loss_like_noise_pred")
    if weighted_captions: run_cmd_list.append("--weighted_captions")
    if save_state: run_cmd_list.append("--save_state")
    if resume:
        run_cmd_list.append("--resume")
        run_cmd_list.append(resume)
    if save_every_n_steps > 0: # Check value before adding
        run_cmd_list.append("--save_every_n_steps")
        run_cmd_list.append(str(save_every_n_steps))
    if use_wandb: run_cmd_list.append("--use_wandb")
    if wandb_api_key:
        run_cmd_list.append("--wandb_api_key")
        run_cmd_list.append(wandb_api_key)
    if log_tracker_name:
        run_cmd_list.append("--log_tracker_name")
        run_cmd_list.append(log_tracker_name)
    if log_tracker_config:
        run_cmd_list.append("--log_tracker_config")
        run_cmd_list.append(log_tracker_config) # Assume correct format
    if async_upload: run_cmd_list.append("--async_upload")
    # Add any additional parameters from the GUI field
    if additional_parameters:
        run_cmd_list.extend(additional_parameters.split()) # Split space-separated string

    # Log the constructed command
    log.info(f"Executing command list: {run_cmd_list}")

    # Prepare environment (ensure venv bin is in PATH for accelerate and its dependencies)
    current_env = os.environ.copy()
    current_env['PATH'] = f"{venv_bin_path}:{current_env.get('PATH', '')}"

    # Run the command using subprocess.run
    try:
        # Use shell=False and pass the command list and environment
        # Set CWD to repo root for consistency, although paths are absolute
        process = subprocess.run(run_cmd_list, env=current_env, shell=False, check=True, cwd=repo_root)
        log.info("Training process completed.")
    except subprocess.CalledProcessError as e:
        log.error(f"Training process failed with error code {e.returncode}")
        log.error(f"Stderr: {e.stderr}")
        log.error(f"Stdout: {e.stdout}")
        gr.Error(f"Training process failed. Check logs. Error code: {e.returncode}")
    except Exception as e:
        log.error(f"An unexpected error occurred during training: {e}")
        gr.Error(f"An unexpected error occurred: {e}")

# UI Elements (Keep existing UI definition code)
# ... lora_tab(...) ...
