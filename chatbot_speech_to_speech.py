'''
    *** In The Name Of GOD ***
    author:  AliBinary
    Email: AliGhanbariCs@gmail.com
    GitHub: https://github.com/AliBinary
    created: 07.07.2025 19:14:24
'''

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
from io import BytesIO
import sounddevice as sd
import webrtcvad
import whisper
import queue
import threading
import sys
import time
import json
import pysbd
import py3langid
import subprocess
import requests
import array


import numpy as np
from scipy.signal import resample
from piper import PiperVoice      # The main engine
#from parler_tts import ParlerTTSForConditionalGeneration

# LM Studio client import (optional)
try:
    from lm_studio_client import LMStudioClient
except ImportError:
    print("⚠️ Install lm_studio_client module for LM Studio support")
    LMStudioClient = None


class ChatBotSpeech:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Check for LM Studio backend
        lm_studio_cfg = self.config.get("lm_studio", {})
        self.use_lm_studio = lm_studio_cfg.get("enabled", False)
        
        # Load LLM
        llm_cfg = self.config["llm"]
        
        if self.use_lm_studio:
            print("🔄 Using LM Studio backend...")
            api_url = lm_studio_cfg.get("api_url", "http://localhost:1234/v1/chat/completions")
            model_name = lm_studio_cfg.get("model_name", "")
            timeout = lm_studio_cfg.get("timeout", 60)
            
            if LMStudioClient is None:
                raise ImportError("LM Studio client module not found.")
                
            self.llm_client = LMStudioClient(
                api_url=api_url, 
                model_name=model_name,
                timeout=timeout
            )
        else:
            print(f"Loading LLM model: {llm_cfg['model_id']}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_cfg["model_id"], trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_cfg["model_id"],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"LLM loaded on device: {device}")


        self.llm_config = llm_cfg

        #-----Comment out this section if you want to use espeak-ng instead.-----
        '''
        # --- TTS LOADING (Parler-TTS Multilingual - Apache 2.0) ---
        tts_cfg = self.config["tts"]
        model_id = tts_cfg["model_id"]
        print(f"Loading Commercial-Safe TTS: {model_id} (Speaker: {tts_cfg['speaker_name']})")
        
        # Multilingual requires two tokenizers
        self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tts_tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Specifically for the description (Jessica's voice...)
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.tts_model.config.text_encoder._name_or_path
        )
        
        # Build the final Jessica description
        self.speaker_description = tts_cfg["description_template"].format(name=tts_cfg["speaker_name"])
        self.audio_playback = None
        '''
         #----------------------------------------------------------------------

        # --- STT Loading (Whisper MIT) ---
        stt_cfg = self.config["stt"]
        self.whisper_model = whisper.load_model(stt_cfg["model_id"])

        # 1. Initialize the storage for loaded voices (This fixes the error)
        self.loaded_voices = {} 
        
        # 2. Set the directory where models will be saved
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

        # 3. Add the mapping for the languages you want to support
        self.voice_map = {
                            "ar": "ar_JO-kamel-medium",
                            "ca": "ca_ES-upc_ona-medium",
                            "cs": "cs_CZ-jirka-medium",
                            "cy": "cy_GB-gwann-medium", # Using Welsh medium
                            "da": "da_DK-toke_andersen-medium",
                            "de": "de_DE-thorsten-medium",
                            "el": "el_GR-raphael-medium",
                            "en": "en_US-kristin-medium", # Default high-quality US
                            "es": "es_ES-mls_9972-low",
                            "fa": "fa_IR-amir-medium",
                            "fi": "fi_FI-harri-medium",
                            "fr": "fr_FR-siwis-medium",
                            "hi": "hi_IN-priyamvada-medium",
                            "hu": "hu_HU-anna-medium",
                            "is": "is_IS-bui-medium",
                            "it": "it_IT-elsa-medium",
                            "ka": "ka_GE-natia-medium",
                            "kk": "kk_KZ-issai-high",
                            "lb": "lb_LU-mary-medium",
                            "lv": "lv_LV-peters-medium",
                            "ml": "ml_IN-meera-medium",
                            "ne": "ne_NP-chitwan-medium",
                            "nl": "nl_BE-rdh-medium", # Your Dutch target
                            "no": "no_NO-talesyntese-medium",
                            "pl": "pl_PL-mls_6892-low",
                            "pt": "pt_PT-soul_and_co-medium",
                            "ro": "ro_RO-mihai-medium",
                            "ru": "ru_RU-denis-medium",
                            "sk": "sk_SK-lili-medium",
                            "sl": "sl_SI-artur-medium",
                            "sr": "sr_RS-serbski_institut-medium",
                            "sv": "sv_SE-lisa-medium",
                            "sw": "sw_CD-lanfrica-medium",
                            "tr": "tr_TR-dfki-medium",
                            "uk": "uk_UA-ukrainian_tts-medium",
                            "vi": "vi_VN-vais1000-medium",
                            "zh": "zh_CN-huayan-medium"
                        }
        print(f"Mapped Languages: {list(self.voice_map.keys())}")

        # 4. Initialize the sentence splitter
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

        print(f"✅ Multilingual system ready for: {self.voice_map}")

        # Setup VAD
        self.vad = webrtcvad.Vad(2)
        self.sample_rate = 16000
        self.frame_duration = 30  # milliseconds
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.audio_queue = queue.Queue()
                # ... (rest of your code above remains the same)

        # Setup VAD - Mode 1 is much better for headset mics
        self.vad = webrtcvad.Vad(1) 
        self.sample_rate = 16000
        self.frame_duration = 30  # milliseconds
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.audio_queue = queue.Queue()

        #Pause flag
        self.paused = False 
        self.forced_lang = None
         
        #Short-term memory storage for conversation
        self.history = []

        # Use ID 11 (Default) for both Input and Output
        # This lets the Linux System (Pulse/PipeWire) handle the routing
        try:
            sd.default.device = [11, 11] 
            print("✅ Audio routed through System Default (ID 11)")
        except Exception as e:
            sd.default.device = [None, None]

    def get_voice(self, lang_code):
        # Ensure it's lowercase for the map lookup
        short_code = lang_code[:2]
        
        # 1. Get model name from map
        model_name = self.voice_map.get(short_code)
        target_lang = short_code

        # 2. Linear Fallback (Fixes RuntimeError)
        if not model_name:
            print(f"⚠️ No native Piper model for '{short_code}'. Falling back to English.")
            target_lang = "en"
            model_name = "en_US-lessac-medium"

        # 3. Check RAM cache using the full model name
        if model_name in self.loaded_voices:
            return self.loaded_voices[model_name]

        onnx_path = os.path.join(self.model_dir, f"{model_name}.onnx")
        
        # 4. Download if missing
        # 1. THE CRITICAL FIX: Use .isfile() instead of .exists()
        # This forces a download if the file is missing, even if a folder/ghost exists
        if not os.path.isfile(onnx_path):
            print(f"📡 {model_name}.onnx is missing. Triggering downloader...")
            self.download_piper_model(model_name, short_code)

        # 5. Load and cache
        print(f"📂 Loading Piper Voice: {model_name}")
        voice = PiperVoice.load(onnx_path)
        self.loaded_voices[model_name] = voice
        return voice

    def download_piper_model(self, model_full_name, lang_code):
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
        
        parts = model_full_name.split('-')
        region = parts[0]
        voice_name = parts[1]
        quality = parts[2]
        
        folder_path = f"{lang_code}/{region}/{voice_name}/{quality}"
        
        for ext in [".onnx", ".onnx.json"]:
            filename = f"{model_full_name}{ext}"
            
            # REMOVED the extra / between base_url and folder_path
            url = f"{base_url}{folder_path}/{filename}"
            local_path = os.path.join(self.model_dir, filename)
            
            print(f"📥 Downloading: {url}")
            try:
                r = requests.get(url, allow_redirects=True, timeout=30)
                r.raise_for_status() 
                with open(local_path, 'wb') as f:
                    f.write(r.content)
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
                if os.path.exists(local_path):
                    os.remove(local_path)
                raise e




    def generate_response(self, messages):
        if self.use_lm_studio:
            # Use LM Studio API client
            return self.llm_client.generate_response(
                messages,
                temperature=self.llm_config.get("temperature", 0.7),
                top_p=self.llm_config.get("top_p", 0.9),
                max_tokens=self.llm_config.get("max_new_tokens", 150)
            )
        else:

            # 1. Apply the chat template to turn dicts into a single string the model understands
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            # Use local Hugging Face model
            inputs = self.tokenizer(
                formatted_prompt, return_tensors="pt").to(self.model.device)
            
            input_length = inputs.input_ids.shape[1] # Count how many tokens went IN

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.llm_config.get("max_new_tokens", 150),
                temperature=self.llm_config.get("temperature", 0.7),
                top_p=self.llm_config.get("top_p", 0.9),
                do_sample=self.llm_config.get("do_sample", True),
            )
            # 2. Slice the output IDs to keep only the NEW tokens
            new_tokens = outputs[0][input_length:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
    #----------espeak-ng only ----------------------------------------------
    '''
    def speak(self, text):
        sd.stop()
        
        # 1. Strip formatting (Your original logic)
        for char in "*_#`>~":
            text = text.replace(char, "")

        # 2. Smart segment (Your original logic)
        sentences = self.segmenter.segment(text)
        
        print(f"🔊 Rule-based synthesis for {len(sentences)} segments...")

        for sentence in sentences:
            if not sentence.strip():
                continue

            # 3. Detect Language (FastText or py3langid)
            # We need the ISO code (nl, en, de, etc.)
            detected, _ = py3langid.classify(sentence)
            
            # 4. Execute eSpeak-ng (Instant, Low-Power)
            # -v: voice/language (e.g., nl, en-us)
            # -s: speed (words per minute)
            # -p: pitch (0-99)
            # -a: amplitude/volume
            try:
                # We use a subprocess to keep the main thread light
                subprocess.run([
                    "espeak-ng", 
                    f"-v{detected}", 
                    "-s160", 
                    "-p55", 
                    sentence
                ], check=True)
            except Exception as e:
                print(f"⚠️ eSpeak failed for {detected}: {e}")
    '''
    #------------------------------------------------------------------------
    
    #-----Comment out this section if you want to use espeak-ng instead.-----
    #'''

    def speak(self, text):
        sd.stop()
        
        # 1. Your original format stripping logic
        for char in "*_#`>~":
            text = text.replace(char, "")
        if not text.strip(): return

        # 2. DRIVE CHECK
        available_files = [f for f in os.listdir(self.model_dir) if f.endswith(".onnx")]
    
        # Match the highest-ranked detection to an actual file on disk
        predictions = py3langid.rank(text)

        # 3. IDENTIFY TOP GUESS vs. BEST LOCAL OPTION
        top_lang, top_score = predictions[0]

        # Find the highest-ranked language that is ALREADY on the drive
        best_local_match = next(
            ((lang, score) for lang, score in predictions 
             if any(f.startswith(lang[:2]) for f in available_files)), 
            (None, 0.0)
        )
        local_lang, local_score = best_local_match

        # 4. DECISION: If the top guess is better than our local options, DOWNLOAD IT
        # This triggers if 'nl' (0.9) > 'en' (0.1)

        # 4. DECIDE TARGET FILE (Scope Corrected)
        target_file = None

        if top_lang != local_lang and top_lang in self.voice_map:
            model_name = self.voice_map[top_lang]
            target_file = f"{model_name}.onnx"
            onnx_path = os.path.join(self.model_dir, target_file)
            
            # THE CRITICAL FIX: Use .isfile to ensure we aren't blocked by empty folders
            if not os.path.isfile(onnx_path):
                print(f"📈 Better match found: {top_lang} ({top_score:.2f}) > {local_lang} ({local_score:.2f})")
                print(f"📡 Downloading {model_name}...")
                self.download_piper_model(model_name, top_lang)
        else:
            # Otherwise, use the best thing we already have on the drive
            target_file = next(
                (f for lang, _ in predictions for f in available_files if f.startswith(lang[:2])), 
                available_files[0] if available_files else None
            )

        # 5. LOAD & PLAY
        if not target_file:
            print("❌ No model available to speak.")
            return

        # 3. Load/Get Voice
        model_key = target_file.replace(".onnx", "")
        if model_key not in self.loaded_voices:
            self.loaded_voices[model_key] = PiperVoice.load(os.path.join(self.model_dir, target_file))
        voice = self.loaded_voices[model_key]

        print(f"🚀 Piper speaking ({target_file})...")

        # 4. THE VERIFIED FIX: Use 'audio_int16_bytes' (Type: bytes)
        # This matches the attribute discovered in your DEBUG log
        try:
            audio_bytes = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text))
            
            # Convert to float32 for sounddevice (Normalizing signed 16-bit)
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            if audio_data.size > 0:
                target_sr = 44100
                original_sr = voice.config.sample_rate
                num_samples = int(len(audio_data) * target_sr / original_sr)
                final_audio = resample(audio_data, num_samples)
                
                sd.play(final_audio, target_sr)
                sd.wait() # Your required pause
                
        except Exception as e:
            print(f"❌ Playback Failure: {e}")



    #'''
    #------------------------------------------------------------------------


    def animate_typing(self, stop_event):
        animation = ["|", "/", "-", "\\"]
        idx = 0
        while not stop_event.is_set():
            sys.stdout.write(
                f"Bot is typing... {animation[idx % len(animation)]}\r")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.2)
        sys.stdout.write(" " * 30 + "\r")
        sys.stdout.flush()

    def is_speech(self, data):
        return self.vad.is_speech(data, self.sample_rate)

    def record_audio(self):
        print("🎤 Listening... Speak now.")
        buffer = bytes()
        silence_duration = 0
        speaking = False

        def callback(indata, frames, time_info, status):
            if status:
                print(status, file=sys.stderr)
            self.audio_queue.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                frame = self.audio_queue.get()
                if self.is_speech(frame):
                    buffer += frame
                    silence_duration = 0
                    speaking = True
                else:
                    if speaking:
                        silence_duration += self.frame_duration
                        if silence_duration > 800:
                            break
            return buffer

    def transcribe(self, audio_bytes, force_lang=None):
        # 1. Prepare audio (Standard 16-bit PCM to float32)
        audio_np = np.frombuffer(audio_bytes, dtype="int16").astype("float32") / 32768.0
        
        # 2. DECISION: Autodetect or Force?
        if force_lang:
            # Skip LID pass, go straight to the forced language dictionary
            # This is the "One-Shot" fix for heavy accents
            result = self.whisper_model.transcribe(audio_np, language=force_lang)
            detected_lang = force_lang
        else:
            # AGNOSTIC MODE: Let the model guess
            # Flipped Logic: Check probabilities first for better debug data
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio_np)).to(self.whisper_model.device)
            _, probs = self.whisper_model.detect_language(mel)
            acoustic_guess = max(probs, key=probs.get)

            result = self.whisper_model.transcribe(audio_np)
            # Use py3langid to verify the written text characters
            detected_lang, _ = py3langid.classify(result["text"])
            
            print(f"DEBUG | Autodetect -> Acoustic: {acoustic_guess} | Text: {detected_lang}")

        text = result["text"].strip()
        
        # Return the text and the final confirmed language code
        return text, detected_lang


    def handle_input(self, user_input):
        bot_response = None
        self.history.append({"role": "user", "content": f"{user_input} /no_think"})

        # 2. Optional: Summary/Truncation (Keep last 10 messages to save context/VRAM)
        #if len(self.history) > 10:
        #    self.history = self.history[-10:]

        def generate():
            nonlocal bot_response

            system_msg = self.llm_config.get("prompt_behavior", "You are a sensual assistant.")
            messages = [{"role": "system", "content": system_msg}] + self.history


            bot_response = self.generate_response(messages)

        gen_thread = threading.Thread(target=generate)
        stop_event = threading.Event()
        animation_thread = threading.Thread(
            target=self.animate_typing, args=(stop_event,))

        gen_thread.start()
        animation_thread.start()

        gen_thread.join()
        stop_event.set()
        animation_thread.join()

        return bot_response

    def chat(self):
        print("🟢 Full Voice ChatBot ready. Press any Ctrl+c to pause.\n")
        print("Wait for the '🎤 Listening...' prompt and just start speaking.")
        print("(To stop the bot, press Ctrl+C in the terminal)\n")
        
        while True:
            # 1. THE HYBRID MENU (Triggered by Ctrl+C)
            if self.paused:
                mode_str = f"FORCED: {self.forced_lang}" if self.forced_lang else "AUTODETECT"
                print("\n" + "—"*30)
                print(f"Current Mode: {mode_str}")
                print("📝 [Text] | [u] Unpause | [nl, en] Lock Lang | [auto] Reset | [q] Quit")
                user_cmd = input("You (Manual): ").strip().lower()

                if not user_cmd: continue
                
                # --- COMMAND LOGIC ---
                if user_cmd == 'u':
                    self.paused = False
                    print("▶️  Resuming Voice Mode...")
                    continue
                elif user_cmd == 'q':
                    print("Goodbye.")
                    break
                elif user_cmd in ['nl', 'en', 'de', 'fr', 'es', 'it', 'pt', 'pl']:
                    self.forced_lang = user_cmd
                    print(f"🔒 Locked voice detection to: {user_cmd.upper()}")
                    continue
                elif user_cmd == 'auto':
                    self.forced_lang = None
                    print("🌐 Switched to Autodetect mode.")
                    continue
                
                # ELSE: If it's not 'u' or 'q', it's a message!
                # We process it exactly like a voice message
                print("🧠 Thinking...")
                response = self.handle_input(user_cmd)
                print(f"Bot: {response}")
                self.speak(response)
                
                # Note: We stay in the 'if self.paused' block so you can type again
                continue

            try:
                # 1. Listen for voice (this is now the FIRST thing it does)
                audio_bytes = self.record_audio()
                
                # 2. Turn voice into text (FIX: Unpack the tuple)
                user_text, detected_lang = self.transcribe(audio_bytes, force_lang=self.forced_lang)
                
                # 3. Process if Whisper caught meaningful text
                if user_text and len(user_text) > 1:
                    print(f"\nUser (Voice - {detected_lang}): {user_text}")
                    
                    # 4. Generate AI response
                    response = self.handle_input(user_text)
                    print(f"Bot: {response}")
                    
                    # 5. Speak the response (with sd.wait() inside speak)
                    self.speak(response)
                else:
                    # If it was just background noise, loop back to listening
                    continue

            except KeyboardInterrupt:
                # 3. CATCH THE INTERRUPT
                # Instead of crashing, we toggle the pause state
                print("\n\n🛑  MANUAL MODE ENABLED.")
                self.paused = True
                # Give the user a moment to see the message before the menu pops up
                time.sleep(0.5) 

            except Exception as e:
                print(f"\n⚠️ An error occurred: {e}")
                time.sleep(1)


if __name__ == "__main__":
    chatbot = ChatBotSpeech()
    chatbot.chat()
