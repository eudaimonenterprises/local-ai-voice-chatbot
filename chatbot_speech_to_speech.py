'''
    *** In The Name Of GOD ***
    author:  AliBinary
    Email: AliGhanbariCs@gmail.com
    GitHub: https://github.com/AliBinary
    created: 07.07.2025 19:14:24
'''


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from TTS.api import TTS
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
import numpy as np

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

        # Load TTS
        tts_cfg = self.config["tts"]
        print(
            f"Loading TTS model: {tts_cfg['model_id']} with speaker '{tts_cfg.get('speaker', 'default')}'")
        self.tts = TTS(model_name=tts_cfg["model_id"])
        self.speaker = tts_cfg.get("speaker")
        self.audio_playback = None

        # Load STT (Whisper)
        stt_cfg = self.config["stt"]
        print(f"Loading STT model: {stt_cfg['model_id']}")
        self.whisper_model = whisper.load_model(stt_cfg["model_id"])
        self.stt_language = stt_cfg.get("language", "en")

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

        self.history = []

        # Use ID 11 (Default) for both Input and Output
        # This lets the Linux System (Pulse/PipeWire) handle the routing
        try:
            sd.default.device = [11, 11] 
            print("✅ Audio routed through System Default (ID 11)")
        except Exception as e:
            sd.default.device = [None, None]


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
        
    def speak(self, text):
        import numpy as np
        from scipy.signal import resample
        
        sd.stop()
        #if "</think>" in text:
        #    text = text.split("</think>")[-1].strip()
        #if not text: return

        # 1. Remove ONLY the symbols, keeping all the words
        # This removes *, _, #, `, and > without deleting the text between them
        forbidden_chars = "*_#`>~"
        for char in forbidden_chars:
            text = text.replace(char, "")

        # TTS Generation
        wav = self.tts.tts(text, speaker=self.speaker)
        audio_data = np.array(wav).astype(np.float32)
        
        # Resampling
        target_sr = 44100
        original_sr = self.tts.synthesizer.output_sample_rate
        num_samples = int(len(audio_data) * target_sr / original_sr)
        resampled_audio = resample(audio_data, num_samples)

        # PLAYBACK
        sd.play(resampled_audio, target_sr)
        
        # --- THE FIX ---
        sd.wait() # This forces the bot to finish talking before it starts listening again


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

    def transcribe(self, audio_bytes):
        audio_np = np.frombuffer(
            audio_bytes, dtype="int16").astype("float32") / 32768.0
        result = self.whisper_model.transcribe(
            audio_np, language=self.stt_language)
        return result["text"].strip()

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
                print("\n" + "—"*30)
                print("📝 [Type a message] | [u] Unpause/Listen | [q] Quit")
                user_cmd = input("You (Manual): ").strip()

                if not user_cmd:
                    continue
                
                # Check for control commands
                if user_cmd.lower() == 'u':
                    self.paused = False
                    print("▶️  Resuming Voice Mode...")
                    continue
                elif user_cmd.lower() == 'q':
                    print("Goodbye.")
                    break
                
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
                
                # 2. Turn voice into text
                user_voice_text = self.transcribe(audio_bytes)
                
                # 3. If Whisper caught something, process it
                if user_voice_text and len(user_voice_text) > 1:
                    print(f"\nUser (Voice): {user_voice_text}")
                    
                    # 4. Generate AI response
                    response = self.handle_input(user_voice_text)
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
