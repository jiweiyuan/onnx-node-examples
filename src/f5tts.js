import ort from 'onnxruntime-node';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import nodejieba from 'nodejieba';
import pinyin from 'pinyin';
import wav from 'node-wav';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Constants and Configs
const CONFIG = {
    HOP_LENGTH: 256,
    SAMPLE_RATE: 24000,
    RANDOM_SEED: 9527,
    NFE_STEP: 32,
    SPEED: 1.0
};

const PATHS = {
    ONNX_MODEL_A: path.join(__dirname, '../models/F5_Preprocess.onnx'),
    ONNX_MODEL_B: path.join(__dirname, '../models/F5_Transformer.onnx'),
    ONNX_MODEL_C: path.join(__dirname, '../models/F5_Decode.onnx'),
    REF_AUDIO: path.join(__dirname, '../data/elevenlab-reference.wav'),
    GEN_AUDIO: path.join(__dirname, '../data/f5tts-generated.wav')
};

// Load vocabulary
const loadVocab = () => {
    const vocabPath = path.join(__dirname, '../data/vocab.txt');
    const vocab = {};
    const lines = fs.readFileSync(vocabPath, 'utf-8').split('\n');
    lines.forEach((line, i) => vocab[line.trim()] = i);
    return vocab;
};

const isChineseChar = (c) => {
    const cp = c.charCodeAt(0);
    return (
        (0x4E00 <= cp && cp <= 0x9FFF) ||
        (0x3400 <= cp && cp <= 0x4DBF) ||
        (0x20000 <= cp && cp <= 0x2A6DF) ||
        (0x2A700 <= cp && cp <= 0x2B73F) ||
        (0x2B740 <= cp && cp <= 0x2B81F) ||
        (0x2B820 <= cp && cp <= 0x2CEAF) ||
        (0xF900 <= cp && cp <= 0xFAFF) ||
        (0x2F800 <= cp && cp <= 0x2FA1F)
    );
};

const convertToPinyin = (text) => {
    const segments = nodejieba.cut(text);
    const pinyinResults = [];
    
    segments.forEach(seg => {
        if (/^[\x00-\x7F]*$/.test(seg)) {
            pinyinResults.push(...seg.split(''));
        } else {
            const pinyinArray = pinyin(seg, {
                style: pinyin.STYLE_TONE3,
                heteronym: true
            });
            pinyinResults.push(...pinyinArray.flat());
        }
    });
    
    return pinyinResults;
};


const loadAudio = async (audioPath) => {
    
    const buffer = fs.readFileSync(audioPath);
    const result = wav.decode(buffer);
    
    // Get mono channel and convert to specified sample rate
    const audioData = new Float32Array(result.channelData[0]);
    
    // Normalize audio (similar to Python's division by 32768.0)
    const normalizedAudio = audioData.map(x => x / 32768.0);
    
    // Get audio length
    const audioLen = normalizedAudio.length;
    
    // Reshape to [1, 1, -1] format
    return {
        data: normalizedAudio,
        length: audioLen,
        sampleRate: CONFIG.SAMPLE_RATE,
        dims: [1, 1, audioLen]
    };
};


async function initializeONNXSessions() {
    const sessionOptions = {
        logSeverityLevel: 3,
        interOpNumThreads: 0,
        intraOpNumThreads: 0,
    };

    const sessionA = await ort.InferenceSession.create(PATHS.ONNX_MODEL_A, sessionOptions);
    const sessionB = await ort.InferenceSession.create(PATHS.ONNX_MODEL_B, sessionOptions);
    const sessionC = await ort.InferenceSession.create(PATHS.ONNX_MODEL_C, sessionOptions);

    console.log('sessionA:', sessionA.outputNames);

    // const inputMetadataA = sessionA.inputNames.map(name => {
    //     const meta = sessionA.input(name);
    //     return {
    //       name,
    //       type: meta.type,
    //       dimensions: meta.dimensions
    //     };
    //   });
    //   console.log("sessionA inputs metadata:", inputMetadataA);

    return { sessionA, sessionB, sessionC };
}

// Add vocabulary mapping
const vocab = loadVocab();

async function main(refText, genText) {
    try {
        console.log('Initializing ONNX sessions...');
        const sessions = await initializeONNXSessions();
        
        console.log('Loading reference audio...');
        const { data: audio } = await loadAudio(PATHS.REF_AUDIO);

        // Shape: [1, 1, audio_length]
        const audioTensor = new ort.Tensor('float32', audio, [1, 1, audio.length]);
        console.log('Audio tensor shape:', audioTensor.dims);
        
        // Convert text to pinyin tokens
        const pinyinTokens = convertToPinyin(refText + genText);

        // Map tokens to vocab IDs
        const textIds = pinyinTokens.map(token => vocab[token] || 0);
        const textBigInts = new Int32Array(
            textIds.map(id => Number.parseInt(id) || 0)
        );

        // >>> FIX: shape => [batch_size=1, seq_len=textIds.length]
        const textTensor = new ort.Tensor('int32', textBigInts, [1, textIds.length]);

        console.log('Text tensor shape:', textTensor.dims);

        // Chinese punctuation pattern
        const zhPausePunc = /[。，、；：？！]/g;

        // Calculate text lengths with punctuation weights
        const getTextLength = (text) => {
            const utf8Length = Buffer.from(text).length;
            const puncCount = (text.match(zhPausePunc) || []).length;
            return utf8Length + (3 * puncCount);
        };

        // Calculate durations
        const refTextLen = getTextLength(refText);
        const genTextLen = getTextLength(genText);
        const refAudioLen = Math.floor(audio.length / CONFIG.HOP_LENGTH) + 1;

        // Calculate max duration with same formula as Python
        const maxDurationValue = BigInt(
            refAudioLen + 
            Math.floor(refAudioLen / refTextLen * genTextLen / CONFIG.SPEED)
        );

        const maxDurationArray = new BigInt64Array([maxDurationValue]);
        const maxDuration = new ort.Tensor('int64', maxDurationArray, [1]);

        console.log('Running initial inference...');
        const { noise, ropeCos, ropeSin } = await sessions.sessionA.run({
            audio: audioTensor,
            max_duration: maxDuration,
            text_ids: textTensor
        });

        console.log('Running NFE steps...');
        // NFE steps
        let currentNoise = noise;
        for (let step = 0; step < CONFIG.NFE_STEP; step++) {
            console.log(`NFE Step: ${step}`);
            const result = await sessions.sessionB.run({
                noise: currentNoise,
                rope_cos: ropeCos,
                rope_sin: ropeSin
            });
            currentNoise = result.output;
        }

        // Generate final audio
        const generatedSignal = await sessions.sessionC.run({
            noise: currentNoise
        });

        // Save generated audio
        const wavData = wav.encode([new Float32Array(generatedSignal.data)], {
            sampleRate: CONFIG.SAMPLE_RATE,
            float: true
        });
        fs.writeFileSync(PATHS.GEN_AUDIO, wavData);
        
        console.log('Audio generation complete!');
        return PATHS.GEN_AUDIO;
    } catch (error) {
        console.error('Error in speech generation:', error);
        throw error;
    }
}

// Example usage
const refText = "Computer scientists and traditional engineers need to speak the same language, a language rooted in real analysis, linear algebra, probability and physics.";
const genText = "You are a highly knowledgeable professor, with expertise spanning computer science, artificial intelligence, mathematics, and physics, biology, learning science. I";

main(refText, genText).catch(console.error);