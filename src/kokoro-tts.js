import { KokoroTTS } from "kokoro-js";

const model_id = "onnx-community/Kokoro-82M-ONNX";
const tts = await KokoroTTS.from_pretrained(model_id, {
  dtype: "q8", // Options: "fp32", "fp16", "q8", "q4", "q4f16"
});
console.time("generate");
const text = "Life is like a box of chocolates. You never know what you're gonna get.";
const audio = await tts.generate(text, {
  // Use `tts.list_voices()` to list all available voices
  voice: "af_bella",
});
console.timeEnd("generate");
audio.save("data/audio.wav");