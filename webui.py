import os
import sys
import shutil

from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed


from pydub import AudioSegment
import pysrt
import tqdm

import gradio as gr
import webbrowser
import socket
import tqdm
import json
from pathlib import Path
from datetime import datetime
import subprocess
import time
import math

from multiprocessing import Pool, cpu_count
import pysrt
from pydub import AudioSegment

from gradio_utils.utils import get_available_items, refresh_dropdown_proxy, move_existing_folder, get_port_available, launch_tensorboard_proxy

import torchaudio
import soundfile as sf
import torch

# --- ROCm PATCH: Bypass torchcodec's hardcoded NVIDIA dependency ---
def _patched_torchaudio_load(filepath, *args, **kwargs):
    audio_np, sr = sf.read(filepath, dtype='float32')
    tensor = torch.from_numpy(audio_np)
    # torchaudio expects (channels, samples)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    return tensor, sr

def _patched_torchaudio_save(filepath, src, sample_rate, *args, **kwargs):
    # torchaudio expects (channels, samples), soundfile expects (samples, channels)
    sf.write(filepath, src.cpu().t().numpy(), sample_rate)
# --- PYTORCH 2.6 PATCH: Bypass strict weights_only check ---
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
torchaudio.load = _patched_torchaudio_load
torchaudio.save = _patched_torchaudio_save

# --- LOCAL PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Force all PyTorch Hub downloads (Silero VAD, SpeechMOS) into local project folder
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
torch.hub.set_dir(CACHE_DIR)

# 2. Local temp scratchpad for intermediate audio chunks
LOCAL_TEMP_DIR = os.path.join(BASE_DIR, "temp_processing")

def get_port_available(start_port=7860, end_port=7865):
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return sock.connect_ex(('localhost', port)) == 0
    webui_port = None         
    while webui_port == None:
        for i in range (start_port, end_port):
            if is_port_in_use(i):
                print(f"Port {i} is in use, moving 1 up")
            else:
                webui_port = i
                break
    return webui_port

def is_correct_dataset_structure(folder_to_analyze):
    if len(os.listdir(folder_to_analyze)) <= 0:
        return False
    for item in os.listdir(folder_to_analyze):
        path_to_item = os.path.join(folder_to_analyze, item)
        if os.path.isdir(path_to_item):
            pass
        else:
            return False
    return True

def folder_to_process_proxy(folder_to_analyze):
    folder_check = is_correct_dataset_structure(folder_to_analyze)
    if folder_check==False:
        raise gr.Error("Please check the folder structure and make sure it contains ONLY folders and that it's NOT empty")
    return gr.Dropdown(value=folder_to_analyze)
gpu_model_queue = Queue()


gpu_model_queue = Queue()

def load_whisperx(model_name='large-v3', parallel_mode=False):
    import stable_whisper
    num_instances = 2 if parallel_mode else 1
    print(f"Loading {num_instances} Whisper model instance(s) into VRAM via ROCm...")
    
    while not gpu_model_queue.empty():
        gpu_model_queue.get()
        
    for i in range(num_instances):
        try:
            model = stable_whisper.load_model(
                model_name, 
                device="cuda", 
                download_root=os.path.join(CACHE_DIR, "whisper")
            )
            gpu_model_queue.put(model)
            print(f"Loaded model instance {i+1}/{num_instances} into VRAM")
        except Exception as e: 
            print(f"Failed to load on GPU: {e}. Falling back to CPU...")
            model = stable_whisper.load_model(
                model_name, 
                device="cpu", 
                download_root=os.path.join(CACHE_DIR, "whisper")
            )
            gpu_model_queue.put(model)
    return True

def prepare_and_chunk_audio(src_file_path, dest_dir, max_chunk_sec=180):
    """
    Reads audio in-memory and writes temporary 3-minute working chunks
    into the local temp directory.
    """
    audio = AudioSegment.from_file(src_file_path).set_channels(1)
    duration_ms = len(audio)
    max_chunk_ms = max_chunk_sec * 1000
    base_name = os.path.splitext(os.path.basename(src_file_path))[0]
    
    tasks = []
    
    if duration_ms <= max_chunk_ms:
        chunk_wav_path = os.path.join(dest_dir, f"{base_name}.wav")
        chunk_srt_path = os.path.join(dest_dir, f"{base_name}.srt")
        audio.export(chunk_wav_path, format="wav")
        tasks.append((chunk_wav_path, chunk_srt_path))
    else:
        for idx, start_ms in enumerate(range(0, duration_ms, max_chunk_ms)):
            end_ms = min(start_ms + max_chunk_ms, duration_ms)
            chunk_audio = audio[start_ms:end_ms]
            
            chunk_name = f"{base_name}_part{idx:03d}"
            chunk_wav_path = os.path.join(dest_dir, f"{chunk_name}.wav")
            chunk_srt_path = os.path.join(dest_dir, f"{chunk_name}.srt")
            
            chunk_audio.export(chunk_wav_path, format="wav")
            tasks.append((chunk_wav_path, chunk_srt_path))
            
    return tasks

def transcribe_and_save(audio_path, srt_path):
    model = gpu_model_queue.get()
    try:
        result = model.transcribe(
            audio_path, 
            word_timestamps=True, 
            verbose=None,
            temperature=0.0,
            vad=True
        )
        result.to_srt_vtt(srt_path)
    finally:
        gpu_model_queue.put(model)

def split_and_export_slices(audio_path, srt_path, final_dest_dir):
    """
    Slices the chunked audio according to the generated SRT and saves
    the final <=8s training slices named after the speaker folder.
    """
    audio = AudioSegment.from_file(audio_path)
    subs = pysrt.open(srt_path)
    
    # 1. Extract speaker name from the destination folder path
    speaker_name = os.path.basename(os.path.normpath(final_dest_dir))
    
    # 2. Offset the counter by existing files to avoid overwriting across multiple audio chunks
    existing_wavs = len([f for f in os.listdir(final_dest_dir) if f.lower().endswith(".wav")])
    segment_counter = existing_wavs + 1

    for sub in subs:
        start_time = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
        end_time = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
        duration = end_time - start_time
        max_segment_duration = 8000  

        while duration > max_segment_duration:
            segment_end_time = start_time + max_segment_duration
            segment = audio[start_time:segment_end_time]
            output_file = os.path.join(final_dest_dir, f"{speaker_name}_seg{segment_counter:05d}.wav")
            segment.export(output_file, format="wav")
            start_time = segment_end_time
            duration = end_time - start_time
            segment_counter += 1

        if duration > 0:
            segment = audio[start_time:end_time]
            output_file = os.path.join(final_dest_dir, f"{speaker_name}_seg{segment_counter:05d}.wav")
            segment.export(output_file, format="wav")
            segment_counter += 1

def process_speaker_folder(file_info, progress_bar=None):
    folder_path, audio_file, srt_file = file_info

    audio = AudioSegment.from_file(audio_file)
    subs = pysrt.open(srt_file)

    file_stem = os.path.splitext(os.path.basename(audio_file))[0]
    segment_counter = 1 

    for idx, sub in enumerate(tqdm.tqdm(subs, desc="Processing Subtitles", leave=False, file=sys.stdout)):
        start_time = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
        end_time = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
        duration = end_time - start_time

        max_segment_duration = 8000  

        while duration > max_segment_duration:
            segment_end_time = start_time + max_segment_duration
            segment = audio[start_time:segment_end_time]
            output_file = f"{folder_path}/{file_stem}_seg{segment_counter}.wav"
            segment.export(output_file, format="wav")
            start_time = segment_end_time
            duration = end_time - start_time
            segment_counter += 1

        if duration > 0:
            segment = audio[start_time:end_time]
            output_file = f"{folder_path}/{file_stem}_seg{segment_counter}.wav"
            segment.export(output_file, format="wav")
            segment_counter += 1

        if progress_bar:
            progress_bar.update(1)

    os.remove(audio_file)
    os.remove(srt_file)

def split_by_srt(folder_path, progress_bar=None):
    file_pairs = []
    for file in os.listdir(folder_path):
        if file.endswith(('.wav', '.mp3', '.m4a', ".mp4")): 
            audio_file = os.path.join(folder_path, file)
            srt_file = os.path.join(folder_path, file.rsplit('.', 1)[0] + '.srt')
            if os.path.exists(srt_file):
                file_pairs.append((folder_path, audio_file, srt_file))

    with Pool(cpu_count()) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_speaker_folder, file_pairs), total=len(file_pairs), desc="Processing Files", file=sys.stdout))

def process_proxy(folder_to_process_path, parallel_toggle, progress=gr.Progress(track_tqdm=False)):
    training_root = os.path.join(BASE_DIR, "training")
    training_destination = os.path.join(training_root, os.path.basename(folder_to_process_path))
    
    if os.path.exists(training_destination):
        raise gr.Error("Training destination folder already exists. Please delete or rename it.")
    os.makedirs(training_destination, exist_ok=True)

    # Initialize model instances inside local cache
    load_whisperx('large-v3', parallel_mode=parallel_toggle)
    
    if not is_correct_dataset_structure(folder_to_process_path):
        raise gr.Error("Invalid folder structure. Ensure the folder contains ONLY speaker subfolders.")

    speaker_folders = [
        os.path.join(folder_to_process_path, f) 
        for f in os.listdir(folder_to_process_path) 
        if os.path.isdir(os.path.join(folder_to_process_path, f))
    ]
    
    # Ensure clean local scratch space
    if os.path.exists(LOCAL_TEMP_DIR):
        shutil.rmtree(LOCAL_TEMP_DIR)
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

    try:
        for speaker_path in tqdm.tqdm(speaker_folders, desc="Processing Speakers", file=sys.stdout):
            speaker_name = os.path.basename(speaker_path)
            speaker_dest = os.path.join(training_destination, speaker_name)
            os.makedirs(speaker_dest, exist_ok=True)

            speaker_temp_dir = os.path.join(LOCAL_TEMP_DIR, speaker_name)
            os.makedirs(speaker_temp_dir, exist_ok=True)

            # 1. Pre-chunk into local temporary workspace
            all_transcription_tasks = []
            for file in tqdm.tqdm(os.listdir(speaker_path), desc="Pre-chunking Audio", file=sys.stdout, leave=False):
                file_path = os.path.join(speaker_path, file)
                tasks = prepare_and_chunk_audio(file_path, speaker_temp_dir, max_chunk_sec=180)
                all_transcription_tasks.extend(tasks)

            # 2. Transcribe using parallel/sequential dynamic worker pool
            max_workers = 2 if parallel_toggle else 1
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(transcribe_and_save, audio_path, srt_path) 
                    for audio_path, srt_path in all_transcription_tasks
                ]
                for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Transcribing Tasks", file=sys.stdout, leave=False):
                    future.result()

            # 3. Slice each transcribed chunk directly into the final dataset folder
            for audio_path, srt_path in tqdm.tqdm(all_transcription_tasks, desc="Generating Slices", file=sys.stdout, leave=False):
                split_and_export_slices(audio_path, srt_path, speaker_dest)
                
    finally:
        # 4. Clean up temporary uncompressed files immediately
        if os.path.exists(LOCAL_TEMP_DIR):
            shutil.rmtree(LOCAL_TEMP_DIR)
        
    return "Dataset creation completed successfully! All temporary files purged."

def count_items_in_directory(root):
        file_count = 0
        
        # Use rglob to iterate through all files recursively
        for item in root.rglob('*'):  # The '*' pattern matches everything
            if item.is_file():
                file_count += 1
        
        return file_count
    
def find_largest_folder(directory):
    max_file_count = 0
    largest_folder = None
    
    # Iterate through each folder in the given directory
    for root, dirs, files in os.walk(directory):
        # If there are files in the current folder, compare file counts
        if files:
            file_count = len(files)
            if file_count > max_file_count:
                max_file_count = file_count
                largest_folder = root
    
    return max_file_count
    
def training_calculations(total_audio_files, batch_size, epochs):
    batches_per_epoch = total_audio_files // batch_size
    n_steps = epochs * batches_per_epoch
    return [batches_per_epoch, n_steps]

def recommendation_proxy(data_dir, epochs):
    recommended_exposure = 10000
    largest_file_count = find_largest_folder(data_dir)
    user_value = largest_file_count * epochs
    recommended_epochs = math.ceil(recommended_exposure / largest_file_count)
    if user_value < recommended_exposure:
        return (f"Your largest dataset folder contains {largest_file_count} files, so at {epochs} epochs, "
            f"the model will be exposed to this speaker {user_value} times.\n"
            f"Recommended exposure is {recommended_exposure} times, so I'd suggest {recommended_epochs} epochs.\n\n"
            f"You may find it not necessary to train longer but that is up to you to decide.")
    else:
        return f"The amount of training is sufficient.  If the model is lacking after, I recommend either adding more audio files, or training for longer."

def training_proxy(data_dir, batch_size, epochs, num_workers, resume, save_interval, log_interval, progress=gr.Progress(track_tqdm=True)):
    # pathlib used here cuz of beatrice trainer
    from beatrice_trainer.src.train import run_training
    output_name = os.path.basename(data_dir)
    output_dir = os.path.join("trained_models", output_name)
    models_output_dir = Path(os.path.join("trained_models", output_name, "models"))
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    
    total_audio_files = count_items_in_directory(data_dir)
    
    training_values = training_calculations(total_audio_files, batch_size, epochs)
    batches_per_epoch = training_values[0]
    n_steps = training_values[1]
    warmup_steps = n_steps // 2
    
    if warmup_steps > 10000:
        warmup_steps = 10000
    
    def update_configurations(config, batch_size, n_steps, num_workers, warmup_steps):
        config['batch_size'] = batch_size
        config['n_steps'] = n_steps
        config['num_workers'] = num_workers
        config['warmup_steps'] = warmup_steps
        return config
    
    config_path = Path('assets/default_config.json')
    with config_path.open('r') as file:
        config = json.load(file)
        
    config = update_configurations(config, batch_size, n_steps, num_workers, warmup_steps)
    updated_config_path = output_dir / 'updated_config.json'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with updated_config_path.open('w') as file:
        json.dump(config, file, indent=4)
    
    try:
        run_training(data_dir, models_output_dir, batches_per_epoch, save_interval, log_interval , resume, updated_config_path)
    except Exception as e:
        raise gr.Error(e)
    
if __name__ == "__main__":
    import torch

    whisper_model = None
    VALID_AUDIO_EXT = [
        ".mp3",   
        ".wav",   
        ".aac",   
        ".flac",  
        ".ogg",   
        ".m4a",   
        ".opus",  
        ".mp4"
    ]
    
    def load_settings():
        settings_file = 'configs/settings.json'
        
        if not os.path.exists(settings_file):
            settings = {"custom_theme": True, "dark_mode": True}
            save_settings(settings) 
        else:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
        
        return settings

    def save_settings(settings):
        os.makedirs(os.path.dirname('configs/settings.json'), exist_ok=True)
        with open('configs/settings.json', 'w') as f:
            json.dump(settings, f, indent=4)

    settings = load_settings()
    if settings.get("custom_theme", True):
        theme = gr.themes.Glass(
            primary_hue="zinc",
            secondary_hue="slate",
            neutral_hue="orange",
            text_size="lg"
        ).set(
            body_background_fill_dark='*primary_900',
            body_text_color='*primary_950',
            body_text_color_subdued='*neutral_500',
            embed_radius='*radius_md',
            border_color_accent_subdued_dark='*neutral_950',
            border_color_primary_dark='*secondary_800',
            color_accent_soft='*primary_400',
            block_border_width_dark='0',
            block_label_border_width_dark='None',
            block_shadow_dark='*primary_600 0px 0px 5px 0px',
            button_border_width='2px',
            button_border_width_dark='0px',
            button_shadow_hover='*block_shadow',
            button_large_radius='*radius_md',
            button_small_radius='*radius_md',
            button_small_text_weight='500',
            button_primary_border_color='*primary_500',
            button_primary_border_color_dark='*primary_950'
            
        )
    else:
        theme = gr.themes.Default()

    def toggle_theme():
        settings = load_settings()
        settings["custom_theme"] = not settings.get("custom_theme", False)
        save_settings(settings)
        if settings['custom_theme']:
            gr.Info("Gradio will boot up with custom theme on next start up.")
        else:
            gr.Info("Gradio will boot up with the default theme on next start up.")
            
    def toggle_dark_mode():
        settings = load_settings()
        settings["dark_mode"] = not settings.get("dark_mode", True)
        save_settings(settings)
        if settings['dark_mode']:
            gr.Info("Gradio will boot up with dark mode on next start up.")
        else:
            gr.Info("Gradio will boot up with light mode on next start up.")

    js_dark_mode = "document.querySelector('body').classList.add('dark');" if settings.get("dark_mode", True) else "document.querySelector('body').classList.remove('dark');"

    js = f"""
        function createGradioAnimation() {{
            var container = document.createElement('div');
            container.id = 'gradio-animation';
            container.style.fontSize = '2em';
            container.style.fontWeight = 'bold';
            container.style.textAlign = 'center';
            container.style.marginBottom = '20px';
            container.style.position = 'absolute';
            container.style.left = '-100%'; 
            container.style.top = '20px'; 
            container.style.transition = 'left 1s ease-out'; 
            container.style.zIndex = '1000'; 
            container.style.whiteSpace = 'nowrap'; 
            container.style.overflow = 'hidden'; 
            container.style.textOverflow = 'ellipsis'; 

            var text = 'Beatrice Voice Changer Training Webui';
            container.innerText = text;

            var gradioContainer = document.querySelector('.gradio-container');
            gradioContainer.style.position = 'relative'; 
            gradioContainer.style.paddingTop = '60px'; 
            gradioContainer.insertBefore(container, gradioContainer.firstChild);

            setTimeout(function() {{
                container.style.left = '50%';
                container.style.transform = 'translateX(-50%)'; 
            }}, 100);

            {js_dark_mode} 
            return 'Animation created';
        }}
    """
        
    with gr.Blocks(js=js, theme=theme) as demo:
        with gr.Tab("Create Dataset"):
            with gr.Row():
                with gr.Column():
                    hidden_dataset_textbox = gr.Textbox(value="datasets", visible=False)
                    list_of_datasets = get_available_items(root="datasets", directory_only=True)
                    folder_to_process = gr.Dropdown(choices=list_of_datasets, value=None, label="Dataset to Process")
                    
                    parallel_toggle = gr.Checkbox(label="Enable 2x GPU Parallelism (Double Model Load)", value=False)

                    refresh_datasets_button = gr.Button(value="Refresh Datasets Available")
                    move_existing_folder_button = gr.Button(value="Move Existing Folder")
                    process_button = gr.Button(value="Begin Process", variant="primary")
                with gr.Column():
                    console_output = gr.Textbox(label="Progress Console")

                process_button.click(fn=process_proxy,
                                     inputs=[folder_to_process, parallel_toggle],
                                     outputs=console_output
                                     )
                folder_to_process.change(fn=folder_to_process_proxy,
                                         inputs=folder_to_process,
                                         outputs=folder_to_process
                                         )
                
                destination_root = gr.Textbox(value="training/moved_training_datasets", visible=False)
                source_root = gr.Textbox(value="training", visible=False)
                move_existing_folder_button.click(fn=move_existing_folder,
                                                  inputs=[source_root, 
                                                          folder_to_process, 
                                                          destination_root]
                )
                
        with gr.Tab("Train"):
            with gr.Row():
                with gr.Column():
                    hidden_train_textbox = gr.Textbox(value="training", visible=False)
                    TRAINING_SETTINGS = {}
                    list_of_training_datasets = get_available_items(root="training", directory_only=True)
                    TRAINING_SETTINGS["dataset_name"] = gr.Dropdown(label="Dataset to Train", choices=list_of_training_datasets, value=list_of_training_datasets[0] if list_of_training_datasets else '')
                    refresh_training_available_button = gr.Button(value="Refresh Training Datasets Available")
                    TRAINING_SETTINGS["batch_size"] = gr.Slider(label="Batch Size", minimum=1, maximum=64, value=4, step=1)
                    TRAINING_SETTINGS["epochs"] = gr.Slider(label="Number of Epochs", minimum=1, maximum=1000, value=20, step=1)
                    TRAINING_SETTINGS["num_workers"] = gr.Slider(label="Number of Workers",minimum=1, maximum=32, value=4, step=1)
                    TRAINING_SETTINGS["save_interval"] = gr.Slider(label="Save Interval in Epochs", minimum=1, maximum= 200, value= 5, step=1)
                    TRAINING_SETTINGS["log_interval"] = gr.Slider(label="Console Log Interval", minimum=10, maximum=1000, step=10)
                    TRAINING_SETTINGS["resume"] = gr.Checkbox(label="Resume Training", value=False)

                    html_value = '''<h2>What are Batches</h2>
                                    <p>Bunches or groups of files that are processed at once by the model before updating gradients (model predictions --> loss calc --> gradient update). A batch size of 1 trains on a single audio file at a time, a batch size of 8 trains on 8 audio files at a time.</p>

                                    <h3>Batch Size:</h3>
                                    <p>The number of audio files processed per step. The higher the value, the faster training is but also incurs more VRAM usage.</p>

                                    <h2>What are Epochs</h2>
                                    <p>A complete pass through the entire dataset where the model has "been trained on" all of the audio samples a single time.</p>

                                    <h3>Number of Epochs:</h3>
                                    <p>The amount of epochs you want to train the model for. The higher the value, the better the model may sound but it will take longer to finish.</p>

                                    <h2>What are Workers</h2>
                                    <p>Processes or "sorters" that go through the dataset to curate and create the batches needed for training.</p>

                                    <h3>Number of Workers:</h3>
                                    <p>The amount of workers created to sort the data. The higher the value, the faster data gets prepared, but may cause unnecessary overhead if your GPU isn't fast enough. Recommend to leave at default value.</p>

                                    <h3>Save Interval in Epochs:</h3>
                                    <p>How often a model is saved. For larger datasets, I'd save at lower intervals as you will need fewer epochs to complete training. For smaller datasets, I'd save at larger intervals to reduce how much space training will take up to complete.</p>

                                    <h3>Console Log Interval:</h3>
                                    <p>How often training loss is output to the terminal and for tensorboard. Recommend 10-100.</p>
                                    '''
                with gr.Column():   
                    recommendation_console = gr.Textbox(label="Jarods's Recommendation")
            with gr.Row():
                output_console = gr.Textbox(label="Training Console")
            with gr.Row():
                with gr.Column():
                    start_train_button = gr.Button(value="Start Training", variant="primary")
                with gr.Column():
                    launch_tb_button = gr.Button(value="Launch Tensorboard")
            with gr.Row():
                gr.HTML(value=html_value)
        with gr.Tab("Settings"):
            dark_mode_btn = gr.Button("Dark Mode", variant="primary")
            toggle_theme_btn = gr.Button("Toggle Custom Theme", variant="primary")
            
        for key, component in TRAINING_SETTINGS.items():
                if isinstance(component, gr.Dropdown):
                    component.change(
                        fn=recommendation_proxy, 
                        inputs=[TRAINING_SETTINGS["dataset_name"], 
                                TRAINING_SETTINGS["epochs"]],
                        outputs=recommendation_console
                        )
                elif isinstance(component, gr.Slider):
                    component.release(
                        fn=recommendation_proxy, 
                        inputs=[TRAINING_SETTINGS["dataset_name"], 
                                TRAINING_SETTINGS["epochs"]],
                        outputs=recommendation_console
                        )

        start_train_button.click(fn=training_proxy,
                                 inputs=[
                                     TRAINING_SETTINGS["dataset_name"],
                                     TRAINING_SETTINGS["batch_size"],
                                     TRAINING_SETTINGS["epochs"],
                                     TRAINING_SETTINGS["num_workers"],
                                     TRAINING_SETTINGS["resume"],
                                     TRAINING_SETTINGS["save_interval"],
                                     TRAINING_SETTINGS["log_interval"]
                                         ],
                                 outputs=output_console
                                 )
        
        launch_tb_button.click(fn=launch_tensorboard_proxy)
        
        hidden_option1 = gr.Textbox(value="directory", visible=False)
        hidden_option2 = gr.Textbox(value="files", visible=False)
        
        hidden_extensions1 = gr.Textbox(value="[]", visible=False)
        
        refresh_training_available_button.click(fn=refresh_dropdown_proxy,
                                                inputs=[
                                                    hidden_train_textbox, hidden_extensions1, hidden_option1
                                                    ],
                                                outputs=[
                                                    TRAINING_SETTINGS["dataset_name"]
                                                ]
        )
        
        refresh_datasets_button.click(fn=refresh_dropdown_proxy,
                                                inputs=[
                                                    hidden_dataset_textbox, hidden_extensions1, hidden_option1
                                                    ],
                                                outputs=[
                                                    folder_to_process
                                                ]
        )
        
        toggle_theme_btn.click(toggle_theme)
        dark_mode_btn.click(toggle_dark_mode)

        dark_mode_btn.click(
            None,
            None,
            None,
            js="""() => {
            if (document.querySelectorAll('.dark').length) {
                document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
            } else {
                document.querySelector('body').classList.add('dark');
            }
        }""",
            show_api=False,
        )
            
    port = get_port_available()
    webbrowser.open(f"http://localhost:{port}")
    demo.launch()