import gradio as gr
import subprocess
import os
import sys # <-- Added import
from .common_gui import (
    get_folder_path,
    get_file_path,
    get_save_folder_path,
    get_executable_path,
    tk_msgbox,
    tk_msgbox_askyesno,
)  # Keep existing imports
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Constants
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'sd-scripts') # Used to find the script
# Default values
DEFAULT_REPO_ID = 'SmilingWolf/wd-v1-4-convnextv2-tagger-v2'
DEFAULT_MODEL_FILENAME = 'model.onnx'
DEFAULT_TAGGER_THRESHOLD = 0.35


# Function to get the path for the model file
def get_model_path(model_dir):
    return os.path.join(model_dir, DEFAULT_MODEL_FILENAME)


# Function to check if the model file exists
def model_exists(model_dir):
    return os.path.exists(get_model_path(model_dir))


# Function to download the model using git LFS
def download_model(model_dir, repo_id):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    repo_id_parts = repo_id.split('/')
    if len(repo_id_parts) != 2:
        log.error(f'Invalid repo_id format. Expected owner/name, got {repo_id}')
        return

    owner, name = repo_id_parts
    git_clone_command = [
        'git',
        'clone',
        f'https://huggingface.co/{owner}/{name}',
        model_dir,
    ]
    log.info(f'Cloning repository {repo_id} to {model_dir}')
    subprocess.run(git_clone_command)

    git_lfs_pull_command = ['git', 'lfs', 'pull']
    log.info(f'Running git lfs pull in {model_dir}')
    subprocess.run(git_lfs_pull_command, cwd=model_dir)


# Captioning function
# NOTE: This function is modified to correctly find and use the python executable
#       from the virtual environment instead of generating 'launch ...'
def caption_images(
    image_dir,
    caption_file_ext,
    batch_size,
    max_data_loader_n_workers,
    max_length,
    model_dir,
    force_download,
    caption_separator,
    prefix,
    postfix,
    tag_threshold,
    undescored_threshold,
    onnx,
    repo_id,
    debug,
    frequency_tags,
    remove_underscore,
):
    # Check if the image directory exists
    if not image_dir:
        log.error('Image folder is required.')
        return

    # Construct the model directory path based on repo_id
    if not model_dir:
        if repo_id:
            normalized_repo_id = repo_id.replace('/', '_')  # Normalize for directory name
            # Place model dir inside the main repo folder for clarity
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            model_dir = os.path.join(repo_root, 'wd14_models', normalized_repo_id) # Changed path
            log.info(f"Model directory not specified, using derived path: {model_dir}")
        else:
            log.error('Either Model folder or Repo ID must be provided.')
            return

    # Check if the model needs to be downloaded
    if force_download or not model_exists(model_dir):
        log.info(
            f'Downloading model {repo_id if repo_id else "provided in model_dir"}...'
        )
        download_model(model_dir, repo_id)
        if not model_exists(model_dir):
            log.error(
                'Model download failed or model file not found in the specified directory.'
            )
            return

    log.info(f'Captioning files in {image_dir}...')

    # Find the tagger script using a relative path from this file's location
    tagger_script_path_relative = os.path.join("sd-scripts", "finetune", "tag_images_by_wd14_tagger.py")
    # Get the repo root assuming this script is in kohya_gui subdirectory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tagger_script_path_absolute = os.path.join(repo_root, tagger_script_path_relative)

    # Get the python executable from the current venv
    python_executable = sys.executable
    log.info(f"Using python: {python_executable}")
    log.info(f"Using tagger script: {tagger_script_path_absolute}")


    # Build the command list correctly
    run_cmd_list = [python_executable, tagger_script_path_absolute]

    # Add arguments based on GUI inputs
    run_cmd_list.append("--batch_size")
    run_cmd_list.append(str(batch_size))
    run_cmd_list.append("--max_data_loader_n_workers")
    run_cmd_list.append(str(max_data_loader_n_workers))
    run_cmd_list.append("--caption_extension")
    run_cmd_list.append(caption_file_ext)
    if caption_separator:
         run_cmd_list.append("--caption_separator")
         run_cmd_list.append(caption_separator)
    if prefix:
         run_cmd_list.append("--prefix")
         run_cmd_list.append(prefix)
    if postfix:
         run_cmd_list.append("--postfix")
         run_cmd_list.append(postfix)
    if tag_threshold > 0:
         run_cmd_list.append("--thresh")
         run_cmd_list.append(str(tag_threshold))
    if undescored_threshold > 0:
         run_cmd_list.append("--undescored_threshold")
         run_cmd_list.append(str(undescored_threshold))
    if onnx:
         run_cmd_list.append("--onnx")
         run_cmd_list.append("--model_dir") # ONNX needs model_dir
         run_cmd_list.append(model_dir)
    if repo_id and not onnx: # repo_id is used if not using ONNX
        run_cmd_list.append("--repo_id")
        run_cmd_list.append(repo_id)
    if force_download:
        run_cmd_list.append("--force_download")
    if debug:
        run_cmd_list.append("--debug")
    if frequency_tags:
        run_cmd_list.append("--frequency_tags")
    if remove_underscore:
        run_cmd_list.append("--remove_underscore")

    # Add the image directory path
    run_cmd_list.append(image_dir)

    # Log the command to be executed
    log.info(f'Executing command list: {run_cmd_list}') # Use the list here

    # Prepare environment
    env = os.environ.copy()
    # Optional: Ensure venv bin is in PATH, though may not be needed if calling python directly
    venv_bin_path = os.path.dirname(python_executable)
    env['PATH'] = f"{venv_bin_path}:{env.get('PATH', '')}"

    # Run the command using subprocess.run with shell=False
    try:
        # Use shell=False and pass the command as a list
        subprocess.run(run_cmd_list, env=env, check=True, shell=False) # Add check=True to raise error on failure
        log.info('...captioning done')
    except subprocess.CalledProcessError as e:
        log.error(f"Subprocess failed with error code {e.returncode}: {e.stderr}")
        # Optionally raise the error or return an error message to Gradio
        raise e # Re-raise the exception to make it visible in Gradio logs potentially
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}")
        raise e


# Gradio UI definition function
def wd14_caption_ui(headless=False):
    with gr.Row():
        image_dir = gr.Textbox(
            label='Image folder to caption',
            placeholder='Directory containing the images to caption',
            interactive=True,
        )
        button_image_dir = gr.Button(
            '📂', elem_id='open_folder_small', visible=not headless
        )
        button_image_dir.click(
            get_folder_path, outputs=image_dir, show_progress=False
        )
        repo_id = gr.Textbox(
            label='Repo ID',
            placeholder='Specify the repo ID for the WD14 tagger model (e.g., SmilingWolf/wd-v1-4-convnextv2-tagger-v2)',
            value=DEFAULT_REPO_ID,
            interactive=True,
        )

    with gr.Row():
        model_dir = gr.Textbox(
            label='Model folder',
            placeholder='Directory where the model is stored or will be downloaded',
            interactive=True,
        )
        button_model_dir = gr.Button(
            '📂', elem_id='open_folder_small', visible=not headless
        )
        button_model_dir.click(
            get_folder_path, outputs=model_dir, show_progress=False
        )
        force_download = gr.Checkbox(
            label='Force model re-download',
            value=False,
            info='Useful to re-download when switching to onnx',
        )
        onnx = gr.Checkbox(label='Use ONNX model', value=False)

    with gr.Row():
        caption_file_ext = gr.Textbox(
            label='Caption file extension',
            placeholder='Extension for caption files (e.g., .caption, .txt)',
            value='.txt',
            interactive=True,
        )
        caption_separator = gr.Textbox(
            label='Caption Separator',
            value=',',
            interactive=True,
            placeholder='Separator character for captions',
        )
        prefix = gr.Textbox(
            label='Prefix to add to WD14 caption',
            placeholder='(Optional) Add a prefix to the caption',
            interactive=True,
        )
        postfix = gr.Textbox(
            label='Postfix to add to WD14 caption',
            placeholder='(Optional) Add a postfix to the caption',
            interactive=True,
        )
    with gr.Row():
        batch_size = gr.Number(
            label='Batch size', value=1, interactive=True, precision=0
        )
        max_data_loader_n_workers = gr.Number(
            label='Max dataloader workers',
            value=2,
            interactive=True,
            precision=0,
        )
        max_length = gr.Number(
            label='Max length', value=75, interactive=True, precision=0
        )
    with gr.Row():
        tag_threshold = gr.Slider(
            label='Tag threshold',
            minimum=0,
            maximum=1,
            step=0.01,
            value=DEFAULT_TAGGER_THRESHOLD,
            interactive=True,
        )
        undescored_threshold = gr.Slider(
            label='Underscored threshold',
            minimum=0,
            maximum=1,
            step=0.01,
            value=0,
            interactive=True,
        )
    with gr.Row():
        frequency_tags = gr.Checkbox(
            label='Frequency tags', value=True, interactive=True
        )
        remove_underscore = gr.Checkbox(
            label='Remove underscore', value=True, interactive=True
        )
        debug = gr.Checkbox(label='Debug mode', value=False, interactive=True)

    caption_button = gr.Button('Caption images')

    caption_button.click(
        caption_images,
        inputs=[
            image_dir,
            caption_file_ext,
            batch_size,
            max_data_loader_n_workers,
            max_length,
            model_dir,
            force_download,
            caption_separator,
            prefix,
            postfix,
            tag_threshold,
            undescored_threshold,
            onnx,
            repo_id,
            debug,
            frequency_tags,
            remove_underscore,
        ],
        show_progress=False,  # Set to False because we're running a blocking subprocess
    )

# Interface initialization
# This part is usually called from the main kohya_gui.py script
# if __name__ == '__main__':
#     interface = gr.Blocks()
#     with interface:
#         with gr.Tab("WD14 Captioning"):
#              wd14_caption_ui()
#     interface.launch()
