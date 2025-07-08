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


class ChatBotSpeech:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Load LLM
        llm_cfg = self.config["llm"]
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

    def generate_prompt(self, user_input):
        return f"User: {user_input}\nAssistant:"

    def generate_response(self, user_input):
        prompt = self.generate_prompt(user_input)
        inputs = self.tokenizer(
            prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.llm_config.get("max_new_tokens", 150),
            temperature=self.llm_config.get("temperature", 0.7),
            top_p=self.llm_config.get("top_p", 0.9),
            do_sample=self.llm_config.get("do_sample", True),
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip() if response.startswith(prompt) else response

    def speak(self, text):
        if self.audio_playback and self.audio_playback.is_playing():
            self.audio_playback.stop()
        wav_bytes = self.tts.tts_to_bytes(text, speaker=self.speaker)
        audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
        self.audio_playback = _play_with_simpleaudio(audio)

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

        def generate():
            nonlocal bot_response
            bot_response = self.generate_response(user_input)

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
        print("🟢 Full Voice ChatBot is ready. Type or speak. Type 'exit' to quit.\n")
        while True:
            print("Waiting for input (type or speak)...")
            user_input = input("You (text): ").strip()

            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            if user_input:
                response = self.handle_input(user_input)
                print("Bot:", response)
                self.speak(response)
                continue

            # No text, wait for voice
            audio_bytes = self.record_audio()
            user_input = self.transcribe(audio_bytes)
            if not user_input:
                print("Didn't catch that. Please try again.")
                continue

            print("You (voice):", user_input)
            response = self.handle_input(user_input)
            print("Bot:", response)
            self.speak(response)


if __name__ == "__main__":
    chatbot = ChatBotSpeech()
    chatbot.chat()
