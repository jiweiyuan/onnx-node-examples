/**
 * End-of-turn detection JavaScript implementation
 * 
 * This is a JavaScript port of the Python implementation from:
 * https://github.com/livekit/agents/blob/main/livekit-plugins/livekit-plugins-turn-detector/livekit/plugins/turn_detector/eou.py
 * 
 * Original source: LiveKit Agents Project
 * License: Apache License 2.0
 */
import { AutoTokenizer } from '@huggingface/transformers';
import ort from 'onnxruntime-node';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// unlikely_threshold
const UNLIKELY_THRESHOLD = 0.15;

const chatExample1 = [
    { role: "user", content: "What's the weather like today?" },
    { role: "assistant", content: "It's sunny and warm." },
    { role: "user", content: "I like the weather. but" },
    { role: "user", content: "I'm not sure what to" }
];

const chatExample2 = [
    { role: "user", content: "What's the weather like today?" },
    { role: "assistant", content: "It's sunny and warm." },
    { role: "user", content: "I like the weather. but" },
    { role: "user", content: "I'm not sure what to do?" }
];

const chatExample3 = [
    { role: "user", content: "What's the weather like today?" },
    { role: "assistant", content: "It's sunny and warm." },
    { role: "user", content: "I like the weather. but" },
    { role: "user", content: "I'm not sure what to do? But maybe" }
];

/**
 * Model Setup Instructions:
 * 1. Download ONNX model from Hugging Face:
 *    https://huggingface.co/livekit/turn-detector/blob/main/model_quantized.onnx
 * 2. Save model to: ../models/model_quantized.onnx
 * 
 * Note: Only this specific quantized model version is compatible with this implementation.
 */
async function initializeModel() {
    try {
        console.time('initializeModel');
        const localPathOnnx = path.resolve(__dirname, '../models/model_quantized.onnx');
        const session = await ort.InferenceSession.create(localPathOnnx);
        const tokenizer = await AutoTokenizer.from_pretrained('livekit/turn-detector');
        const eou_index = tokenizer.encode("<|im_end|>")[0];
        console.timeEnd('initializeModel');
        
        return { tokenizer, session, eou_index };
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

function normalize(text) {
    const PUNCS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'; // Punctuation without single quote
    // Remove punctuation and normalize whitespace
    const stripped = text.replace(new RegExp(`[${PUNCS}]`, 'g'), '');
    return stripped.toLowerCase().trim().split(/\s+/).join(' ');
}

async function formatChatContext(chatContext, tokenizer) {
    const normalizedContext = chatContext
        .map(msg => {
            const content = normalize(msg.content);
            return content ? { ...msg, content } : null;
        })
        .filter(Boolean);

    const convoText = tokenizer.apply_chat_template(normalizedContext, {
        add_generation_prompt: false,
        add_special_tokens: false,
        tokenize: false,
        add_generation_prompt: true,
      });
    
    const eouToken = "<|im_end|>";
    const lastEOUIndex = convoText.lastIndexOf(eouToken);
    return lastEOUIndex >= 0 ? convoText.slice(0, lastEOUIndex) : convoText;
}

function softmax(logits) {
    const arr = Array.isArray(logits) ? logits : Array.from(logits);
    const maxLogit = Math.max(...arr);
    const expLogits = arr.map(x => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(x => x / sumExp);
}

async function predictEndOfTurn(chatContext, { tokenizer, session, eou_index }) {
    const formattedText = await formatChatContext(chatContext, tokenizer);
    
    const inputs = await tokenizer(formattedText, {
        return_tensors: 'np',
        dtype: 'int64'
    });
    
    const output = await session.run({
        input_ids: inputs.input_ids,
        attention_mask: inputs.attention_mask,
        past_key_values: inputs.past_key_values,
        dtype : 'int64'
    });
    const data = output.logits.data;
    const dims = output.logits.dims;
    const lastTokenLogits = data.slice(-dims[2]);
    const probs = softmax(lastTokenLogits);
    const probability = probs[eou_index];
    return probability;
}

initializeModel().then(async (model) => {
    // count the time
    console.time('predictEndOfTurn');
    const probability1 = await predictEndOfTurn(chatExample1, model);
    console.log('End of turn probability1:', probability1);
    const probability2 = await predictEndOfTurn(chatExample2, model);
    console.log('End of turn probability2:', probability2);
    const probability3 = await predictEndOfTurn(chatExample3, model);
    console.log('End of turn probability3:', probability3);
    console.log(`If probability is less than ${UNLIKELY_THRESHOLD}, the model predicts that the user hasn't finished speaking.`);
    console.timeEnd('predictEndOfTurn');
}).catch(console.error);